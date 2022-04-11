import torch
import numpy as np
import math
import resource
import sys
import gc
import os
import re
import time


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

    from QTensorAI.qtensor_ai.ParallelQTensor import TraceEvaluationCircuitComposer, RPQCComposer, ParallelQtreeSimulator, ParallelQtreeTensorNet, ParallelTorchBackend
    from QTensorAI.qtensor_ai.Quantum_Neural_Net import circuit_optimization
    from qtensor.optimisation.Optimizer import DefaultOptimizer, TamakiOptimizer, TamakiExactOptimizer

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
        
    optimizer = TamakiOptimizer(wait_time=(20*min(2*n_qubits, n_layers*2)))
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



'''class RPQC(nn.Module):
    def __init__(self, n_qubits, n_layers) -> None:
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        init_params1 = 2*torch.rand(1, n_qubits, n_layers, requires_grad=False)-1
        init_params2 = 2*torch.rand(1, n_qubits, n_layers, requires_grad=False)-1
        gate_types1 = torch.randint(3, (1, n_qubits, n_layers))
        gate_types2 = torch.randint(3, (1, n_qubits, n_layers))
        rpqc1 = RPQCComposer(n_qubits)
        rpqc2 = RPQCComposer(n_qubits)
        trace = TraceEvaluationCircuitComposer(n_qubits, RPQCComposer.name())
        sim = ParallelQtreeSimulator(backend=ParallelTorchBackend())
        optimizer = TamakiOptimizer(wait_time=(20*min(2*n_qubits, n_layers*2)))
        rpqc1.expectation_circuit(gate_types1, init_params1)
        rpqc2.expectation_circuit(gate_types2, init_params2)
        trace.expectation_circuit(rpqc1.static_circuit + trace.inverse(rpqc2.static_circuit))
        tn, _, _ = ParallelQtreeTensorNet.from_qtree_gates(trace.static_circuit)
        ''''''peo is the tensor network contraction order''''''
        self.peo = circuit_optimization(n_qubits, n_layers, tn, optimizer=optimizer, composer=trace)
        super().__init__()

    def forward(self, gate_types1, init_params1, gate_types2, init_params2):
        return None'''



def RPQC_estimate(n_qubits, n_layers, n_batch, device):

    from QTensorAI.qtensor_ai.ParallelQTensor import TraceEvaluationCircuitComposer, RPQCComposer, ParallelQtreeSimulator, ParallelQtreeTensorNet, ParallelTorchBackend
    from QTensorAI.qtensor_ai.Quantum_Neural_Net import circuit_optimization
    from qtensor.optimisation.Optimizer import DefaultOptimizer, TamakiOptimizer, TamakiExactOptimizer
    
    init_params1 = 2*torch.rand(n_batch, n_qubits, n_layers, requires_grad=False).to(device)-1
    init_params2 = 2*torch.rand(n_batch, n_qubits, n_layers, requires_grad=False).to(device)-1
    gate_types1 = torch.randint(3, (n_batch, n_qubits, n_layers)).to(device)
    gate_types2 = torch.randint(3, (n_batch, n_qubits, n_layers)).to(device)

    rpqc1 = RPQCComposer(n_qubits)
    rpqc2 = RPQCComposer(n_qubits)
    trace = TraceEvaluationCircuitComposer(n_qubits, RPQCComposer.name())
    sim = ParallelQtreeSimulator(backend=ParallelTorchBackend())
        
    optimizer = TamakiOptimizer(wait_time=(5*min(2*n_qubits, n_layers*2+2)))

    rpqc1.expectation_circuit(gate_types1, init_params1)
    rpqc2.expectation_circuit(gate_types2, init_params2)
    trace.expectation_circuit(rpqc1.static_circuit + trace.inverse(rpqc2.static_circuit))

    tn, _, _ = ParallelQtreeTensorNet.from_qtree_gates(trace.static_circuit)
    '''peo is the tensor network contraction order'''
    peo = circuit_optimization(n_qubits, n_layers, tn, optimizer=optimizer, composer=trace)

    results = (2**n_qubits)*torch.abs(sim.simulate_batch(trace.static_circuit, peo=peo))

    return results



def trace_generation(directory, k):

    ngpus = torch.cuda.device_count()
    print('number of gpus', ngpus)

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    haar_frame_potential = math.factorial(k)
    for n_qubits in [50,20]:
        n_batch = 16777216//n_qubits
        mean = haar_frame_potential
        std = haar_frame_potential
        for n_layers in [10]:#,150,200,250,300,400,500]:
            #if (std<haar_frame_potential/100 and haar_frame_potential>mean-std):
            #    break
            
            #p = p_finder(n_qubits, n_layers)
            iteration = 0
            trying = True
            mostly_settled = False
            settled = False

            results = torch.empty((0)).to('cpu')
            file = directory+'n_{}_l_{}.pt'.format(n_qubits, n_layers)
            if os.path.isfile(file):
                while True:
                    try:
                        results = torch.load(file)
                        break
                    except EOFError:
                        time.sleep(1)

            while trying:
                
                try:
                    '''calculating what the batch size should be based on memory usage'''

                    samples = results.shape[0]
                    power = results**(2*k)
                    mean = torch.mean(power)
                    std = torch.std(power)/np.sqrt(1+samples)

                    if results.shape[0] == 0:
                        mean = haar_frame_potential
                        std = haar_frame_potential

                    p = 0
                    iteration = 0
                    
                    while True:
                        if device == 'cuda' and ngpus>1:
                            futures = [torch.jit.fork(RPQC_estimate, n_qubits, n_layers, n_batch, 'cuda:'+str(gpu)) for gpu in range(ngpus)]
                            iteration_results = [torch.jit.wait(fut).to('cpu') for fut in futures]
                            iteration_results = torch.cat(iteration_results, -1)
                        else:
                            iteration_results = RPQC_estimate(n_qubits, n_layers, n_batch, device).to('cpu')
                        
                        file = directory+'n_{}_l_{}.pt'.format(n_qubits, n_layers)
                        if os.path.isfile(file):
                            while True:
                                try:
                                    results = torch.load(file)
                                    break
                                except EOFError:
                                    time.sleep(1)

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
                        print('Saving {} samples. At iteration {}, the mean is {}, and the standard deviation of the mean is {}. The frame potential of a {} design is {}'.format(samples, iteration, mean, std, k, haar_frame_potential))
                        torch.save(results, directory + '/n_{}_l_{}.pt'.format(n_qubits, n_layers))
                        if not mostly_settled:
                            mostly_settled = True
                        if not settled:
                            old_n_batch = n_batch
                            n_batch = int(n_batch*1.1)

                        if (mean-2*std > haar_frame_potential):
                            break

                    print('Experiment for n={} and l={} is over.'.format(n_qubits, n_layers))
                    print('Saving. At iteration {}, the mean is {}, and the standard deviation of the mean is {}. The frame potential of a {} design is {}'.format(iteration, mean, std, k, haar_frame_potential))
                    torch.save(results, directory + '/n_{}_l_{}.pt'.format(n_qubits, n_layers))
                    trying = False

                except RuntimeError:
                    gc.collect()
                    torch.cuda.empty_cache()
                    if not mostly_settled:
                        n_batch //= 2
                    else:
                        n_batch = old_n_batch
                        settled = True
                        print('Settled batch size is ', n_batch)
                    if n_batch == 0:
                        print('Cannot fit a single circuit.')
                        raise

            



def frame_potential(results_dir: str):
    max_ns = 75
    max_l = 20
    max_k = 5
    frame_potential = torch.zeros((max_ns, max_l, max_k, 2))
    frame_potential[:,:,:,:] = np.nan

    files = os.listdir(results_dir)

    #files1 = os.listdir('./results/k=5/')
    #files2 = os.listdir('./results/gpu_large_k=5/')
    #files = set(files1+files2)

    n_pattern = re.compile(r'n_\d*')
    l_pattern = re.compile(r'l_\d*')
    for file in files:
        n_match = n_pattern.findall(file)
        if len(n_match) != 0:
            n = int(n_match[0][2:])
            l_match = l_pattern.findall(file)
            l = int(l_match[0][2:])
            #tensor1 = torch.zeros(0)
            #tensor2 = torch.zeros(0)
            #try:
            #    tensor1 = torch.load('./results/k=5/'+file)
            #    print('In k=5')
            #except FileNotFoundError:
            #    pass
            #try:
            #    tensor1 = torch.load('./results/gpu_large_k=5/'+file)
            #    print('In gpu_large_k=5')
            #except FileNotFoundError:
            #    pass
            #tensor = torch.cat((tensor1, tensor2))
            tensor = torch.load(results_dir + file)
            samples = tensor.shape[0]
            print('n={}, l={}, {} samples'.format(n,l,samples))
            torch.save(tensor, results_dir + '/n_{}_l_{}.pt'.format(n, l))
            for k in range(1, max_k+1):
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
        #trace_generation(directory = './results/combined/', k=5)
        frame_potential(results_dir = './results/combined/')
    except MemoryError:
        sys.stderr.write('\n\nERROR: Memory Exception\n')
        sys.exit(1)