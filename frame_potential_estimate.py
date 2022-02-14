import torch
import math

from QTensorAI.qtensor_ai.ParallelQTensor import TraceEvaluationCircuitComposer, RPQCComposer, ParallelQtreeSimulator, ParallelQtreeTensorNet, ParallelTorchBackend
from QTensorAI.qtensor_ai.Quantum_Neural_Net import circuit_optimization
from qtensor.optimisation.Optimizer import TamakiOptimizer




def RPQC_estimate(n_qubits, n_layers, n_batch, device, peo=None):

    rpqc1 = RPQCComposer(n_qubits)
    rpqc2 = RPQCComposer(n_qubits)
    trace = TraceEvaluationCircuitComposer(n_qubits, RPQCComposer.name())
    sim = ParallelQtreeSimulator(backend=ParallelTorchBackend())
    optimizer = TamakiOptimizer(wait_time=2*min(n_qubits, n_layers*2))

    init_params1 = torch.rand(n_batch, n_qubits, n_layers, requires_grad=False).to(device)
    init_params2 = torch.rand(n_batch, n_qubits, n_layers, requires_grad=False).to(device)
    gate_types1 = torch.randint(3, (n_batch*n_qubits*n_layers,)).reshape((n_batch, n_qubits, n_layers)).to(device)
    gate_types2 = torch.randint(3, (n_batch*n_qubits*n_layers,)).reshape((n_batch, n_qubits, n_layers)).to(device)

    rpqc1.expectation_circuit(gate_types1, init_params1)
    rpqc2.expectation_circuit(gate_types2, init_params2)
    trace.expectation_circuit(rpqc1.static_circuit + trace.inverse(rpqc2.static_circuit))

    tn, _, _ = ParallelQtreeTensorNet.from_qtree_gates(trace.static_circuit)
    '''peo is the tensor network contraction order'''
    peo = circuit_optimization(n_qubits, n_layers, tn, optimizer=optimizer, composer=trace)

    return ((2**n_qubits)*sim.simulate_batch(trace.static_circuit, peo=peo).abs())




def main():
    
    k=2
    n_qubits = 5
    n_layers = 10
    n_batch = 100
    device = 'cuda'

    iterations = 5
    results = torch.empty((0)).to(device)
    for i in range(iterations):
        iteration_results = RPQC_estimate(k, n_qubits, n_layers, n_batch, device)
        results = torch.cat((results, iteration_results))
        print('On the ', i, 'th iteration, memory allocated: ', torch.cuda.memory_allocated('cuda'), '; reserved: ', torch.cuda.memory_reserved('cuda'))

    mean = results.mean()
    std = results.std()
    print('Results have a mean of {}, and a standard deviation of {}. The Frame potential of a {} design is {}'.format(mean, std, k, math.factorial(k)))


def main():
    device = 'cuda'
    k=4
    for n_qubits in range(2, 24, 2):
        for n_layers in [1,2,3,4,5,6,7,8,9,10,20,30,40,50,75,100,150,200,250,300,400,500]:

            '''calculating what the batch size should be based on memory usage'''
            torch.cuda.empty_cache()
            before_memory = torch.cuda.memory_allocated('cuda')
            iteration_results = RPQC_estimate(n_qubits, n_layers, 1, device)
            after_memory = torch.cuda.memory_allocated('cuda')
            memory_per_batch = after_memory - before_memory
            n_batch = max(10000, 18000000000/memory_per_batch)

            results = torch.empty((0)).to('cpu')
            haar_frame_potential = math.factorial(k)
            mean = 0
            var = 100000000000
            std = math.sqrt(var)
            iteration = 0
            while (std>math.sqrt(haar_frame_potential)/10) and ((haar_frame_potential>mean-2*std) or (haar_frame_potential<mean+2*std))):
                iteration_results = RPQC_estimate(n_qubits, n_layers, n_batch, device).to('cpu')
                results = torch.cat((results, iteration_results))
                mean = ( mean*iteration + iteration_results**(2*k).mean() )  /  (iteration+1)
                var =  ( var*iteration + iteration_results**(2*k).var() )  /  (iteration+1)
                std = math.sqrt(var)/math.sqrt((iteration+1)*n_batch)
                iteration += 1
                print('At iteration {}, the mean is {}, and the standard deviation of the mean is {}. The frame potential of a {} design is {}'.format(iteration, mean, std, k, haar_frame_potential))
            print('Saving results.\nAt the end of experiment for n={} and l={}, the mean is {}, and the standard deviation of the mean is {}. The frame potential of a {} design is {}'.format(n_qubits, n_layers, mean, std, k, haar_frame_potential))
            print('Memory used in this experiment: {}.'.format(torch.cuda.memory_allocated('cuda')))
            torch.save(results, '/home/hl8967/Frame_Potential/results/n_{}_l_{}.pt'.format(n_qubits, n_layers))




if __name__ == "__main__":
    main()