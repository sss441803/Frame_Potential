import torch
import numpy as np
import math
import resource
import sys
import gc
import os
import re
import time

from QTensorAI.qtensor_ai.ParallelQTensor import TraceEvaluationCircuitComposer, RPQCComposer, ParallelQtreeSimulator, ParallelQtreeTensorNet, ParallelTorchBackend
from QTensorAI.qtensor_ai.Quantum_Neural_Net import circuit_optimization
from qtensor.optimisation.Optimizer import DefaultOptimizer, TamakiOptimizer, TamakiExactOptimizer

pi = np.pi



def p_finder(n_qubits, n_layers):
    device = 'cpu'
    n_batch = 10000
    ratio = 0
    small_p = 0
    large_p = 1
    p = 0
    while (ratio>100) or (ratio<1/100):
        _, _, _, _, w = param_generator(n_batch, n_qubits, n_layers, device, p)
        ratio = w.mean().detach().cpu().numpy()/(2**(8*n_qubits))
        if p>0.9:
            p=0.9
            break
        elif ratio>100:
            large_p = p
            p = (small_p + p)/2
        elif ratio<1/100:
            small_p = p
            p = (large_p + p)/2
        else:
            break
    return p



def param_generator(n_batch, n_qubits, n_layers, device, p):

    sigma=1/3
    init_params1 = 2*torch.rand(n_batch, n_qubits, n_layers, requires_grad=False).to(device)-1
    init_params2_shift_uniform = 2*torch.rand(n_batch, n_qubits, n_layers, requires_grad=False).to(device)-1
    init_params2_shift_normal = torch.randn(n_batch, n_qubits, n_layers, requires_grad=False).to(device)*sigma
    choice = torch.tensor(np.random.choice(2, n_batch, p=[1-p, p]).astype(bool), requires_grad=False).to(device).reshape(-1, 1, 1).expand(-1, n_qubits, n_layers)
    init_params2_shift = init_params2_shift_normal * choice + init_params2_shift_uniform * (~choice)
    init_params2 = init_params1 + init_params2_shift
    init_params2 = torch.remainder(init_params2, 1)

    gate_types1 = torch.randint(3, (n_batch, n_qubits, n_layers)).to(device)
    gate_types2_change_biased = torch.zeros(n_batch, n_qubits, n_layers).to(device)
    gate_types2_change_uniform = torch.randint(3, (n_batch, n_qubits, n_layers)).to(device)-1
    gate_types2 = gate_types1 + gate_types2_change_biased * choice + gate_types2_change_uniform * (~choice)
    gate_types2 = torch.remainder(gate_types2, 3).int()

    # total_uniform pd is scaled by 2**(n_qubits*n_layers)
    # normal_pd is scaled by 2, which means total_normal_pd is scaled by 2**(n_qubits*n_layers)
    # The result is not altered
    total_uniform_pd = 1
    normal_pd = 2*torch.exp(-0.5*(init_params2_shift/sigma)**2)/(sigma*np.sqrt(2*pi))
    total_normal_pd = torch.prod(torch.prod(normal_pd, 2), 1)
    total_biased_pd = p*total_normal_pd + (1-p)*total_uniform_pd
    w = total_uniform_pd/total_biased_pd

    return init_params1, init_params2, gate_types1, gate_types2, w



def RPQC_estimate(n_qubits, n_layers, n_batch, device, p, qtensor_object_dict=None):

    init_params1, init_params2, gate_types1, gate_types2, w = param_generator(n_batch, n_qubits, n_layers, device, p)

    if qtensor_object_dict == None:
        rpqc1 = RPQCComposer(n_qubits)
        rpqc2 = RPQCComposer(n_qubits)
        trace = TraceEvaluationCircuitComposer(n_qubits, RPQCComposer.name())
        sim = ParallelQtreeSimulator(backend=ParallelTorchBackend())
        qtensor_object_dict = {}
        qtensor_object_dict['rpqc1'], qtensor_object_dict['rpqc2'], qtensor_object_dict['trace'], qtensor_object_dict['sim'] = rpqc1, rpqc2, trace, sim
    else:
        rpqc1, rpqc2, trace, sim = qtensor_object_dict['rpqc1'], qtensor_object_dict['rpqc2'], qtensor_object_dict['trace'], qtensor_object_dict['sim']
        
    optimizer = TamakiOptimizer(wait_time=(2**min(2*n_qubits, n_layers*2+2))//500)
    #optimizer = TamakiExactOptimizer()
    #optimizer = DefaultOptimizer()

    rpqc1.expectation_circuit(gate_types1, init_params1)
    rpqc2.expectation_circuit(gate_types2, init_params2)
    #rpqc1.expectation_circuit(gate_types, init_params1)
    #rpqc2.expectation_circuit(gate_types, init_params2)
    trace.expectation_circuit(rpqc1.static_circuit + trace.inverse(rpqc2.static_circuit))

    tn, _, _ = ParallelQtreeTensorNet.from_qtree_gates(trace.static_circuit)
    '''peo is the tensor network contraction order'''
    peo = circuit_optimization(n_qubits, n_layers, tn, optimizer=optimizer, composer=trace)

    results = (2**n_qubits)*torch.abs(sim.simulate_batch(trace.static_circuit, peo=peo))

    return results, w, qtensor_object_dict



def trace_generation(directory, k):

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    haar_frame_potential = math.factorial(k)
    for n_qubits in [50]:
        n_batch = 16777216//n_qubits
        mean = haar_frame_potential
        std = haar_frame_potential
        for n_layers in [9,10]:#,150,200,250,300,400,500]:
            if (std<haar_frame_potential/20 and haar_frame_potential>mean-2*std and haar_frame_potential<mean+2*std):
                break
            
            #p = p_finder(n_qubits, n_layers)
            iteration = 0
            trying = True
            #if device == 'cuda':

            qtensor_object_dict = None

            while trying:
                try:
                    '''calculating what the batch size should be based on memory usage'''
                    results = torch.empty((0)).to('cpu')
                    
                    try:
                        results = torch.cat( (results, torch.load(directory+'n_{}_l_{}.pt'.format(n_qubits, n_layers))) )
                        print('Found results in ', directory)
                    except FileNotFoundError:
                        print('No results in ', directory)

                    samples = results.shape[0]
                    power = results**(2*k)
                    mean = torch.mean(power)
                    std = torch.std(power)/np.sqrt(1+samples)

                    if results.shape[0] == 0:
                        mean = haar_frame_potential
                        std = haar_frame_potential

                    p = 0
                    iteration = 0
                    while (std>haar_frame_potential/20 and haar_frame_potential>mean-2*std and haar_frame_potential<mean+2*std and iteration<150) or (results.shape[0]<5000):
                        qtensor_object_dict = None
                        iteration_results, w, qtensor_object_dict = RPQC_estimate(n_qubits, n_layers, n_batch, device, p, qtensor_object_dict)
                        iteration_results, w = iteration_results.to('cpu'), w.to('cpu')
                        results = torch.cat((results, iteration_results))
                        samples = results.shape[0]
                        power = results**(2*k)
                        mean = torch.mean(power)
                        std = torch.std(power)/np.sqrt(1+samples)
                        #summand = w*(iteration_results**(2*k))
                        #mean = ( mean*iteration + torch.mean(summand) )  /  (iteration+1)
                        #var =  ( var*iteration + torch.var(summand) )  /  (iteration+1)
                        #std = math.sqrt(var)/math.sqrt((iteration+1)*n_batch)
                        iteration += 1
                        if iteration == 1:
                            print('Batch size for n={} and l={} is {}'.format(n_qubits, n_layers, n_batch))
                        print('Saving. At iteration {}, the mean is {}, and the standard deviation of the mean is {}. The frame potential of a {} design is {}'.format(iteration, mean, std, k, haar_frame_potential))
                        torch.save(results, directory + '/n_{}_l_{}.pt'.format(n_qubits, n_layers))

                    print('Experiment for n={} and l={} is over.'.format(n_qubits, n_layers))
                    print('Saving. At iteration {}, the mean is {}, and the standard deviation of the mean is {}. The frame potential of a {} design is {}'.format(iteration, mean, std, k, haar_frame_potential))
                    torch.save(results, directory + '/n_{}_l_{}.pt'.format(n_qubits, n_layers))
                    trying = False

                except RuntimeError:
                    gc.collect()
                    torch.cuda.empty_cache()
                    n_batch //= 2
                    if n_batch == 0:
                        print('Cannot fit a single circuit.')
                        raise

            



def frame_potential(results_dir: str):
    max_ns = 75
    max_l = 20
    max_k = 7
    frame_potential = torch.zeros((max_ns, max_l, max_k, 2))
    frame_potential[:,:,:,:] = np.nan

    files = os.listdir(results_dir)
    n_pattern = re.compile(r'n_\d*')
    l_pattern = re.compile(r'l_\d*')
    for file in files:
        n_match = n_pattern.findall(file)
        if len(n_match) != 0:
            n = int(n_match[0][2:])
            l_match = l_pattern.findall(file)
            l = int(l_match[0][2:])
            tensor = torch.load(results_dir + file)
            samples = tensor.shape[0]
            for k in range(1, max_k+1):
                print(n,l,k)
                power = tensor**(2*k)
                mean, std = power.mean(), power.std()/np.sqrt(samples)
                try:
                    frame_potential[n-1][l-1][k-1][0] = mean
                    frame_potential[n-1][l-1][k-1][1] = std
                except IndexError:
                    print('File skipped')
    torch.save(frame_potential, results_dir + '/frame_potential.pt')
            


def memory_limit():
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    print(get_memory())
    resource.setrlimit(resource.RLIMIT_AS, (get_memory() * 1024 // 2, hard))

def get_memory():
    with open('/proc/meminfo', 'r') as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                free_memory += int(sline[1])
    return free_memory

if __name__ == '__main__':
    memory_limit() # Limitates maximun memory usage to half
    try:
        trace_generation(directory = './results/k=5/', k=5)
        frame_potential(results_dir = './results/k=5/')
    except MemoryError:
        sys.stderr.write('\n\nERROR: Memory Exception\n')
        sys.exit(1)