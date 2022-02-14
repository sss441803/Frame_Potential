import torch
import math

from QTensorAI.qtensor_ai.ParallelQTensor import TraceEvaluationCircuitComposer, RPQCComposer, ParallelQtreeSimulator, ParallelQtreeTensorNet, ParallelTorchBackend
from QTensorAI.qtensor_ai.Quantum_Neural_Net import circuit_optimization
from qtensor.optimisation.Optimizer import TamakiOptimizer




def RPQC_estimate(k, n_qubits, n_layers, n_batch, device, peo=None):

    rpqc1 = RPQCComposer(n_qubits)
    rpqc2 = RPQCComposer(n_qubits)
    trace = TraceEvaluationCircuitComposer(n_qubits, RPQCComposer.name())
    sim = ParallelQtreeSimulator(backend=ParallelTorchBackend())
    optimizer = TamakiOptimizer(wait_time=20)

    init_params1 = torch.rand(n_batch, n_qubits, n_layers, requires_grad=False).to(device)
    init_params2 = torch.rand(n_batch, n_qubits, n_layers, requires_grad=False).to(device)
    gate_types1 = torch.randint(3, (n_batch*n_qubits*n_layers,)).reshape((n_batch, n_qubits, n_layers)).to(device)
    gate_types2 = torch.randint(3, (n_batch*n_qubits*n_layers,)).reshape((n_batch, n_qubits, n_layers)).to(device)

    rpqc1.expectation_circuit(gate_types1, init_params1)
    rpqc2.expectation_circuit(gate_types2, init_params2)
    trace.expectation_circuit(rpqc1.static_circuit + trace.inverse(rpqc2.static_circuit))

    if peo == None:
        tn, _, _ = ParallelQtreeTensorNet.from_qtree_gates(trace.static_circuit)
        '''peo is the tensor network contraction order'''
        peo = circuit_optimization(n_qubits, n_layers, tn, optimizer=optimizer, composer=trace)

    return ((2**n_qubits)*sim.simulate_batch(trace.static_circuit, peo=peo).abs())**(2*k), peo




def main():
    
    k=2
    n_qubits = 5
    n_layers = 10
    n_batch = 100
    device = 'cuda'

    iterations = 5
    results = torch.empty((0)).to(device)
    peo = None
    for i in range(iterations):
        iteration_results, peo = RPQC_estimate(k, n_qubits, n_layers, n_batch, device, peo=peo)
        results = torch.cat((results, iteration_results))
        print('On the ', i, 'th iteration, memory allocated: ', torch.cuda.memory_allocated('cuda'), '; reserved: ', torch.cuda.memory_reserved('cuda'))

    mean = results.mean()
    std = results.std()
    print('Results have a mean of {}, and a standard deviation of {}. The Frame potential of a {} design is {}'.format(mean, std, k, math.factorial(k)))




if __name__ == "__main__":
    main()