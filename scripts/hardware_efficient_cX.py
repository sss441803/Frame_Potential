import torch
import numpy as np

import qtensor_ai
from qtensor_ai.OpFactory import ParallelParametricGate, ParallelTorchFactory
from qtensor_ai import ParallelComposer, HybridModule, DefaultOptimizer, TamakiOptimizer
from qtensor_ai.TensorNet import ParallelTensorNet
from qtensor_ai.Hybrid_Module import circuit_optimization

from trace_eval_circuit_composer import TraceEvaluationComposer


ansatze_name = 'Hardware_Efficient_cX'
n_qubits_list = [2,4,6,8,10,12,14,16,18,20,24,28,32,36,40,44,48,50]
n_layers_list = [[4,5,6,7,8,9,10,11,12,13,14]]*max(n_qubits_list)

ansatze_config = {
                'ansatze_name': ansatze_name,
                'n_qubits_list': n_qubits_list,
                'n_layers_list': n_layers_list
                }


# Gate that implements x, y or z rotations depending on parameters
class PauliPhase(ParallelParametricGate):
    name = 'PauliPhase'
    _changes_qubits = (0, )
    parameter_count = 2

    def __init__(self, *qubits, **parameters):
        device = parameters['device']
        is_placeholder = parameters['is_placeholder']
        if not is_placeholder:
            self.ct = torch.tensor([[1,0],[0,1]]).to(device)
            self.st = torch.tensor([
                                    [[0, -1j], [-1j, 0]],
                                    [[0, -1], [1, 0]],
                                    [[-1j, 0],[0, 1j]]
                                    ]).to(device)
        super().__init__(*qubits, **parameters)
    
    def _gen_tensor(self, **parameters):
        # rotation angles
        alpha = parameters['alpha']
        # gate types
        gates = parameters['gates']
        c = torch.cos(np.pi*alpha/2).unsqueeze(1)[:,None]*self.ct
        s = torch.sin(np.pi*alpha/2).unsqueeze(1)[:,None]*torch.index_select(self.st, 0, gates)
        return c + s


# Adding the random pauli rotation gate to the family of gates in the factory
ParallelTorchFactory.PauliPhase = PauliPhase


# Circuit composer to automatically generate random parameterized circuits for the 
# hardware efficient ansatze according to https://arxiv.org/pdf/1803.11173.pdf'''
class HardwareEfficientComposer(ParallelComposer):

    def __init__(self, n_qubits, n_layers):
        super().__init__(n_qubits)
        self.n_layers = n_layers

    def layer_of_Hadamards(self):
        for i in range(self.n_qubits):
            self.apply_gate(self.operators.H, i)

    def variational_layer(self, layer_gate_types, layer_params):
        for i in range(self.n_qubits):
            qubit = self.qubits[i]
            self.apply_gate(self.operators.PauliPhase, qubit, alpha=layer_params[:, i], gates=layer_gate_types[:, i])

    def entangling_layer(self):
        for i in range(self.n_qubits//2):
            control_qubit = 2*i
            target_qubit = 2*i+1
            self.apply_gate(self.operators.cX, control_qubit, target_qubit)
        for i in range((self.n_qubits+1)//2-1):
            control_qubit = 2*i+1
            target_qubit = 2*i+2
            self.apply_gate(self.operators.cX, control_qubit, target_qubit)

    # adding circuit that needs to be measured to self.builder.circuit'''
    def circuit(self, gate_types, params):
        '''gate_types is a integer array that has dimension (n_batch, n_qubits, layers). It contains the type of rotation gates to be used'''
        '''params is a np.ndarray that has dimension (n_batch, n_qubits, layers). It stores rotation angles'''
        self.n_batch = params.shape[0]
        assert self.n_layers == params.shape[2], "The number of layers in the parameter is different from that specified for the circuit during initialization"
        self.layer_of_Hadamards()
        for layer in range(self.n_layers):
            layer_gate_types = gate_types[:, :, layer]
            layer_params = params[:, :, layer]
            self.variational_layer(layer_gate_types, layer_params)
            self.entangling_layer()

    # Required method for ParallelComposers. Returns the circuit
    def updated_full_circuit(self, **parameters):
        params1, params2, gate_types1, gate_types2 = parameters['params1'], parameters['params2'], parameters['gate_types1'], parameters['gate_types2']
        self.builder.reset()
        self.device = params1.device
        self.circuit(gate_types1, params1)
        first_part = self.builder.circuit
        self.builder.reset()
        self.circuit(gate_types2, params2)
        self.builder.inverse()
        second_part = self.builder.circuit
        return first_part + second_part

    def name():
        return 'Hardware_Efficient_cX'

    # class method for generating random parameters and basis for the random pauli rotation gates
    @staticmethod
    def param_generator(device, n_batch, n_qubits, n_layers):

        # initializing random pauli rotation gate parameters
        params1 = 2*torch.rand(n_batch, n_qubits, n_layers, requires_grad=False).to(device)-1
        params2 = 2*torch.rand(n_batch, n_qubits, n_layers, requires_grad=False).to(device)-1
        # initializing random pauli basis for rotation (x, y, z are encoded as integers)
        gate_types1 = torch.randint(3, (n_batch, n_qubits, n_layers)).to(device)
        gate_types2 = torch.randint(3, (n_batch, n_qubits, n_layers)).to(device)

        parameters = {}
        parameters['params1'], parameters['params2'], parameters['gate_types1'], parameters['gate_types2'] = params1, params2, gate_types1, gate_types2
        return parameters


# HybridModule is a wrapper for quantum circuits as a torch.nn.Module. This naturally supports backpropagation, but we don't need it.
class HE_trace_mod(HybridModule):

    def __init__(self, n_qubits, n_layers, optimizer=DefaultOptimizer):
        circuit_name = 'HE_cX_trace_n_{}_l_{}'.format(n_qubits, n_layers)
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        HE_composer = HardwareEfficientComposer(n_qubits, n_layers)
        composer = TraceEvaluationComposer(n_qubits, HE_composer)
        super(HE_trace_mod, self).__init__(circuit_name=circuit_name, composer=composer, optimizer=optimizer)

    def forward(self, parameters):
        out = self.parent_forward(**parameters)
        return (2**self.n_qubits)*out.abs()


# Function for sampling random circuits and calculating their trace values.
def trace_gen(device, n_batch, n_qubits, n_layers):
    # Need to initialize new trace circuit for every new n_qubits and n_layers
    # It is okay to initalize a new duplicate one each time
    global previous_n_qubits, previous_n_layers, mod
    need_new_mod = False
    if not 'previous_n_qubits' in globals():
        need_new_mod = True
    elif previous_n_qubits != n_qubits or previous_n_layers!= n_layers:
        need_new_mod = True
    if need_new_mod:
        # Initialize quantum circuit trace evaluation module
        mod = HE_trace_mod(n_qubits, n_layers, optimizer=TamakiOptimizer(wait_time=min(2**min(n_qubits,n_layers), 1500)))
        previous_n_qubits = n_qubits
        previous_n_layers = n_layers

    # random unitaries used in the ansatze
    parameters = HardwareEfficientComposer.param_generator(device, n_batch, n_qubits, n_layers)
    # a context manager where tensors are not stored for backpropagation, 
    # and instead cleared for memory efficient
    with qtensor_ai.forward_only():
        iteration_results = mod(parameters).cpu()

    return iteration_results


def peo_finder():
    for n_qubits in n_qubits_list:
        for n_layers in n_layers_list[n_qubits]:
            optimizer = TamakiOptimizer(wait_time=min(2**min(n_qubits, n_layers), 1500))
            circuit_name = 'HE_cX_trace_n_{}_l_{}'.format(n_qubits, n_layers)
            HE_composer = HardwareEfficientComposer(n_qubits, n_layers)
            composer = TraceEvaluationComposer(n_qubits, HE_composer)
            parameters = HardwareEfficientComposer.param_generator('cpu', 1, n_qubits, n_layers)
            composer.produce_circuit(**parameters) # Update circuit variational parameters
            tn, _, _ = ParallelTensorNet.from_qtree_gates(composer.final_circuit)
            _ = circuit_optimization(circuit_name, tn, optimizer, composer)