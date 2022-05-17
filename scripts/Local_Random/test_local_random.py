import qtree
from qtensor.optimisation.TensorNet import QtreeTensorNet
from qtensor.optimisation.Optimizer import WithoutOptimizer, TamakiExactOptimizer, GreedyOptimizer, TamakiOptimizer
from qtensor import CircuitComposer, QtreeSimulator, TorchBuilder
from qtensor.OpFactory import TorchBuilder
from qtree.operators import Gate, ParametricGate
from qtensor.contraction_backends import TorchBackend

import torch
import torch.nn as nn
import torch.multiprocessing as mp

import numpy as np
import itertools
from functools import partial
import time



def bucket_elimination(buckets, process_bucket_fn,
                       n_var_nosum=0):

    n_var_contract = len(buckets) - n_var_nosum

    result = None
    for n, bucket in enumerate(buckets[:n_var_contract]):
        if len(bucket) > 0:
            tensor = process_bucket_fn(bucket)
            i = 0
            for used_tensor in bucket:
                used_tensor._data = None
                i+=1
            if len(tensor.indices) > 0:
                # tensor is not scalar.
                # Move it to appropriate bucket
                first_index = int(tensor.indices[0])
                buckets[first_index].append(tensor)
            else:   # tensor is scalar
                if result is not None:
                    result *= tensor
                else:
                    result = tensor
            del tensor
            torch.cuda.empty_cache()

    # form a single list of the rest if any
    rest = list(itertools.chain.from_iterable(buckets[n_var_contract:]))
    if len(rest) > 0:
        # only multiply tensors
        tensor = process_bucket_fn(rest, no_sum=True)
        if result is not None:
            result *= tensor
        else:
            result = tensor
    return result

qtree.optimizer.bucket_elimination = bucket_elimination


class M(Gate):
    name = 'M'
    _changes_qubits = (0, )
    """
    Measurement gate. This is essentially the identity operator, but
    it forces the introduction of a variable in the graphical model
    """
    @staticmethod
    def gen_tensor():
        return torch.tensor([[1, 0], [0, 1]])


class RandU(ParametricGate):
    name = 'Rand'
    _changes_qubits=(0, 1)

    @staticmethod
    def _gen_tensor(**parameters):
        return parameters['unitary']

    def __str__(self):
        return ("{}".format(self.name) +
                "({})".format(','.join(map(str, self._qubits)))
        )


class LocalRandomUnitaryComposer(CircuitComposer):

    def __init__(self, n_qubits, n_layers):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        super().__init__()

    def _get_builder(self):
        return self._get_builder_class()(self.n_qubits)

    def _get_builder_class(self):
        return TorchBuilder

    def random_circuit(self, unitaries):
        self.builder.reset()
        for qubit in range(self.n_qubits):
            self.apply_gate(M, qubit)
        for layer in range(self.n_layers):
            qubit = np.random.randint(self.n_qubits-1)
            self.apply_gate(RandU, qubit, qubit+1, unitary=unitaries[layer])
        return self.builder.circuit
    
    def name():
        return 'Local_Random'


'''Compose the circuit to evaluate the trace of the target circuit'''
class TraceEvaluationCircuitComposer(CircuitComposer):
    
    def __init__(self, n_qubits, target_name):
        self.n_target_qubits = n_qubits
        self.n_qubits = n_qubits*2
        self.target_name = target_name
        super().__init__(n_qubits*2)

    def _get_builder(self):
        return self._get_builder_class()(self.n_qubits)
    
    def _get_builder_class(self):
        return TorchBuilder

    def added_circuit(self):
        for target_qubit in range(self.n_target_qubits):
            control_qubit = target_qubit + self.n_target_qubits
            self.apply_gate(self.operators.H, control_qubit)
            self.apply_gate(self.operators.cX, control_qubit, target_qubit)

    '''Building circuit whose first amplitude is the expectation value of the measured circuit wrt to the cost_operator'''
    def expectation_circuit(self, circuit):
        self.builder.reset()
        self.added_circuit()
        first_part = self.builder.circuit
        self.builder.inverse()
        second_part = self.builder.circuit
        self.builder.reset()
        self.static_circuit = first_part + circuit + second_part
        self.expectation_circuit_initialized = True

    def name(self):
        return 'TraceEvaluation' + self.target_name




def random_unitary_generator(n_batch, n_layers, n_qubits):
    module = nn.Linear(1,1)
    module.unitary = nn.Parameter(torch.rand(n_batch, n_layers, 2**n_qubits, 2**n_qubits, dtype=torch.cfloat))
    orthmod = nn.utils.parametrizations.orthogonal(module, name='unitary')
    results = orthmod.unitary.reshape(n_batch, n_layers, 2,2,2,2).detach()
    return results

def get_val(com, trace, sim, unitary):
    peo = None
    circuit = com.random_circuit(unitary)
    trace.expectation_circuit(circuit)
    trace_circuit = trace.static_circuit
    result = sim.simulate_batch(trace_circuit, peo=peo)
    return result.detach()

def get_vals(n_qubits, n_layers, num, device):

    com = LocalRandomUnitaryComposer(n_qubits, 2*n_layers)
    trace = TraceEvaluationCircuitComposer(n_qubits, LocalRandomUnitaryComposer.name())
    sim = QtreeSimulator(backend=TorchBackend())
    get_val_map_fn = partial(get_val, com, trace, sim)
    pool = mp.Pool(num)
    #unitaries = [random_unitary_generator(1, 2*n_layers, 2)[0] for i in range(num)]
    unitaries = random_unitary_generator(num, 2*n_layers, 2).to(device)
    result = pool.map(get_val_map_fn, unitaries)
    return torch.abs(torch.tensor(result))






device = 'cuda'
n_qubits = 5
n_layers = 5
num = 300

com = LocalRandomUnitaryComposer(n_qubits, 2*n_layers)
trace = TraceEvaluationCircuitComposer(n_qubits, LocalRandomUnitaryComposer.name())
sim = QtreeSimulator(backend=TorchBackend())

start = time.time()
print(get_val(com,trace,sim,random_unitary_generator(num, 2*n_layers, 2)[0].to(device)))
stop = time.time()
print('Time taken for a single evaluation: ', stop - start)
start = time.time()
print(get_vals(n_qubits, n_layers, num, device))
stop = time.time()
print('Time taken for multiprocess evaluation: ', stop - start)