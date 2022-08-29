import torch

import qtensor_ai
from qtensor_ai.OpFactory import ParallelParametricGate, ParallelTorchFactory
from qtensor_ai import ParallelComposer, HybridModule, DefaultOptimizer, TamakiOptimizer
from qtensor_ai.TensorNet import ParallelTensorNet
from qtensor_ai.Hybrid_Module import circuit_optimization

from trace_eval_circuit_composer import TraceEvaluationComposer

# This is the library for parameterized random unitary generation.
# Method is based on Jing, Li, et al. "Tunable efficient unitary neural networks (eunn) and their application to rnns." International Conference on Machine Learning. PMLR, 2017.
# Implementation is provided Floris Laporte https://github.com/flaport/torch_eunn.git
# Uniform sampling of parameters gives the Haar measure.
# This is also differentiable and trainable, but we don't care about that functionality here.
from eunn import Unitary

ansatze_name = 'Parallel_Random'
n_qubits_list = [2,4,6,8,10,12,14,16,18,20,24,28,32,36,40,44,48,50]
n_layers_list = [[4,5,6,7,8,9,10,11,12,13,14]]*max(n_qubits_list)

ansatze_config = {
                'ansatze_name': ansatze_name,
                'n_qubits_list': n_qubits_list,
                'n_layers_list': n_layers_list
                }


'''The implementation principle of the code below is best explained in the QTensorAI library https://github.com/sss441803/QTensorAI.git'''


# Random 2-qubit unitary gate
class RandU(ParallelParametricGate):
    name = 'Rand'
    _changes_qubits=(0, 1)

    @staticmethod
    def _gen_tensor(**parameters):
        return parameters['alpha']

    def gen_tensor(self, **parameters):
        if len(parameters) == 0:
            tensor = self._gen_tensor(**self._parameters)
        # implementing the adjoint (adjoint operation is usually taken care of
        # by the ParallelParametricGate class but here the default way doesn't work
        # and had to be implemented here because 4x4 unitaries are first used here)
        if self.is_inverse:
            tensor = torch.permute(tensor, [0,2,1,4,3]).conj()
        return tensor

    def __str__(self):
        return ("{}".format(self.name) +
                "({})".format(','.join(map(str, self._qubits)))
        )


# Adding the random unitary gate to the family of gates in the factory
ParallelTorchFactory.RandU = RandU


# Class to build parallel random unitary circuits automatically
class ParallelRandomUnitaryComposer(ParallelComposer):

    def __init__(self, n_qubits, n_layers):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        super().__init__(n_qubits)

    # adds an even number layer to self.builder.circuit
    def random_circuit_even_layer(self, layer_unitaries):
        for i in range(self.n_qubits//2):
            qubit1 = self.qubits[2*i]
            qubit2 = self.qubits[2*i+1]
            unitary = layer_unitaries[:, i]
            self.apply_gate(self.operators.RandU, qubit1, qubit2, alpha=unitary)

    # adds an odd number layer to self.builder.circuit
    def random_circuit_odd_layer(self, layer_unitaries):
        for i in range((self.n_qubits+1)//2-1):
            qubit1 = self.qubits[2*i+1]
            qubit2 = self.qubits[2*i+2]
            unitary = layer_unitaries[:, i]
            self.apply_gate(self.operators.RandU, qubit1, qubit2, alpha=unitary)

    # Required method for ParallelComposers. Returns the circuit
    def updated_full_circuit(self, **parameters):
        even_layer_unitaries1 = parameters['even_layer_unitaries1']
        odd_layer_unitaries1 = parameters['odd_layer_unitaries1']
        even_layer_unitaries2 = parameters['even_layer_unitaries2']
        odd_layer_unitaries2 = parameters['odd_layer_unitaries2']
        self.builder.reset()
        for i in range(self.n_layers):
            if i % 2 == 0:
                self.random_circuit_even_layer(even_layer_unitaries1[:, i//2])
            else:
                self.random_circuit_odd_layer(odd_layer_unitaries1[:, i//2])
        first_part = self.builder.circuit
        self.builder.reset()
        for i in range(self.n_layers):
            if i % 2 == 0:
                self.random_circuit_even_layer(even_layer_unitaries2[:, i//2])
            else:
                self.random_circuit_odd_layer(odd_layer_unitaries2[:, i//2])
        self.builder.inverse()
        second_part = self.builder.circuit
        self.builder.reset()
        return first_part + second_part
    
    def name(self):
        return 'Parallel_Random'

    '''The following class methods are uniquely needed for parallel random unitaries to generate random unitaries as input parameters'''
    @staticmethod
    def random_unitary_generator(n_batch, counts):
        results = Unitary(n_batch * counts)().reshape(n_batch, 1, counts, 2,2,2,2).detach()
        results = torch.permute(results, (0,1,2,3,5,4,6))
        return results

    @classmethod
    def even_layer_random_unitaries(cls, n_batch, n_qubits):
        count = n_qubits//2
        return cls.random_unitary_generator(n_batch, count)

    @classmethod
    def odd_layer_random_unitaries(cls, n_batch, n_qubits):
        count = (n_qubits-1)//2
        return cls.random_unitary_generator(n_batch, count)

    # Generating all random unitaries needed for the ansatze
    @classmethod
    def circuit_random_unitaries(cls, device, n_batch, n_qubits, n_layers):
        even_layer_unitaries1 = torch.cat([cls.even_layer_random_unitaries(n_batch, n_qubits).to(device) for _ in range((n_layers+1)//2)], axis=1)
        even_layer_unitaries2 = torch.cat([cls.even_layer_random_unitaries(n_batch, n_qubits).to(device) for _ in range((n_layers+1)//2)], axis=1)
        odd_layer_unitaries1 = None
        odd_layer_unitaries2 = None
        if n_layers > 1:
            odd_layer_unitaries1 = torch.cat([cls.odd_layer_random_unitaries(n_batch, n_qubits).to(device) for _ in range(n_layers//2)], axis=1)
            odd_layer_unitaries2 = torch.cat([cls.odd_layer_random_unitaries(n_batch, n_qubits).to(device) for _ in range(n_layers//2)], axis=1)
        
        unitaries = {}
        unitaries['even_layer_unitaries1'] = even_layer_unitaries1
        unitaries['odd_layer_unitaries1'] = odd_layer_unitaries1
        unitaries['even_layer_unitaries2'] = even_layer_unitaries2
        unitaries['odd_layer_unitaries2'] = odd_layer_unitaries2
        return unitaries


# HybridModule is a wrapper for quantum circuits as a torch.nn.Module. This naturally supports backpropagation, but we don't need it.
class PRU_trace_mod(HybridModule):

    def __init__(self, n_qubits, n_layers, optimizer=DefaultOptimizer):
        circuit_name = 'PRU_trace_n_{}_l_{}'.format(n_qubits, n_layers)
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        PRU_composer = ParallelRandomUnitaryComposer(n_qubits, n_layers)
        composer = TraceEvaluationComposer(n_qubits, PRU_composer)
        super(PRU_trace_mod, self).__init__(circuit_name=circuit_name, composer=composer, optimizer=optimizer)

    def forward(self, unitaries):
        out = self.parent_forward(**unitaries)
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
        mod = PRU_trace_mod(n_qubits, n_layers, optimizer=TamakiOptimizer(wait_time=min(2**min(n_qubits,n_layers), 1500)))
        previous_n_qubits = n_qubits
        previous_n_layers = n_layers

    # random unitaries used in the ansatze
    parameters = ParallelRandomUnitaryComposer.circuit_random_unitaries(device, n_batch, n_qubits, n_layers)
    # a context manager where tensors are not stored for backpropagation, 
    # and instead cleared for memory efficient
    with qtensor_ai.forward_only():
        iteration_results = mod(parameters).cpu()

    return iteration_results


# Optimizes the contraction order BEFORE running the simulation.
# The GPU time should be used for only doing numerical calculations, instead of CPU based
# contraction order optimization. Optimizing before means that the GPU doesn't have to
# wait for the CPU to find the contraction order.
# See QTensorAI for how to implement contraction order optimization
def peo_finder():
    for n_qubits in n_qubits_list:
        for n_layers in n_layers_list[n_qubits]:
            optimizer = TamakiOptimizer(wait_time=min(2**min(n_qubits, n_layers), 1500))
            circuit_name = 'PRU_trace_n_{}_l_{}'.format(n_qubits, n_layers)
            PRU_composer = ParallelRandomUnitaryComposer(n_qubits, n_layers)
            composer = TraceEvaluationComposer(n_qubits, PRU_composer)
            parameters = ParallelRandomUnitaryComposer.circuit_random_unitaries('cpu', 1, n_qubits, n_layers)
            composer.produce_circuit(**parameters) # Update circuit variational parameters
            tn, _, _ = ParallelTensorNet.from_qtree_gates(composer.final_circuit)
            _ = circuit_optimization(circuit_name, tn, optimizer, composer)