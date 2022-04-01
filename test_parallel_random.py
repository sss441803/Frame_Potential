import torch
import torch.nn as nn

import numpy as np
import math
import os
import re
import time
import gc
import argparse

import qtensor_ai
from qtensor_ai.OpFactory import ParallelParametricGate, ParallelTorchFactory
from qtensor_ai import ParallelComposer, HybridModule, DefaultOptimizer, TamakiOptimizer
from qtensor_ai.TensorNet import ParallelTensorNet
from qtensor_ai.Hybrid_Module import circuit_optimization


class RandU(ParallelParametricGate):
    name = 'Rand'
    _changes_qubits=(0, 1)

    @staticmethod
    def _gen_tensor(**parameters):
        return parameters['alpha']

    def gen_tensor(self, **parameters):
        if len(parameters) == 0:
            tensor = self._gen_tensor(**self._parameters)
        if self.is_inverse:
            tensor = torch.permute(tensor, [0,2,1,4,3]).conj()
        return tensor

    def __str__(self):
        return ("{}".format(self.name) +
                "({})".format(','.join(map(str, self._qubits)))
        )


ParallelTorchFactory.RandU = RandU


class ParallelRandomUnitaryComposer(ParallelComposer):

    def __init__(self, n_qubits, n_layers):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        super().__init__(n_qubits)

    def random_circuit_even_layer(self, layer_unitaries):
        for i in range(self.n_qubits//2):
            qubit1 = self.qubits[2*i]
            qubit2 = self.qubits[2*i+1]
            unitary = layer_unitaries[:, i]
            self.apply_gate(self.operators.RandU, qubit1, qubit2, alpha=unitary)

    def random_circuit_odd_layer(self,layer_unitaries):
        for i in range((self.n_qubits+1)//2-1):
            qubit1 = self.qubits[2*i+1]
            qubit2 = self.qubits[2*i+2]
            unitary = layer_unitaries[:, i]
            self.apply_gate(self.operators.RandU, qubit1, qubit2, alpha=unitary)

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


'''Compose the circuit to evaluate the trace of the target circuit'''
class TraceEvaluationComposer(ParallelComposer):
    
    def __init__(self, n_qubits, com):
        self.n_target_qubits = n_qubits
        self.n_qubits = n_qubits*2
        self.com = com
        super().__init__(n_qubits*2)

    def added_circuit(self):
        for target_qubit in range(self.n_target_qubits):
            control_qubit = target_qubit + self.n_target_qubits
            self.apply_gate(self.operators.H, control_qubit)
            self.apply_gate(self.operators.cX, control_qubit, target_qubit)

    '''Building circuit whose first amplitude is the expectation value of the measured circuit wrt to the cost_operator'''
    def updated_full_circuit(self, **parameters):
        self.com.n_batch = self.n_batch
        circuit = self.com.updated_full_circuit(**parameters)
        self.builder.reset()
        self.added_circuit()
        first_part = self.builder.circuit
        self.builder.inverse()
        second_part = self.builder.circuit
        self.builder.reset()
        result_circuit = first_part + circuit + second_part
        return result_circuit

    def name(self):
        return 'TraceEvaluation'


class PRU_trace(HybridModule):

    def __init__(self, n_qubits, n_layers, optimizer=DefaultOptimizer):
        circuit_name = 'PRU_trace_n_{}_l_{}'.format(n_qubits, n_layers)
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        PRU_composer = ParallelRandomUnitaryComposer(n_qubits, n_layers)
        composer = TraceEvaluationComposer(n_qubits, PRU_composer)
        super(PRU_trace, self).__init__(circuit_name=circuit_name, composer=composer, optimizer=optimizer)

    @staticmethod
    def random_unitary_generator(n_batch, counts):
        module = nn.Linear(1,1)
        module.unitary = nn.Parameter(torch.rand(n_batch, counts, 2**2, 2**2, dtype=torch.cfloat))
        orthmod = nn.utils.parametrizations.orthogonal(module, name='unitary')
        results = orthmod.unitary.reshape(n_batch, 1, counts, 2,2,2,2).detach()
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

    def layer_random_unitaries(self, device, n_batch):
        even_layer_unitaries1 = torch.cat([self.even_layer_random_unitaries(n_batch, self.n_qubits).to(device) for _ in range((self.n_layers+1)//2)], axis=1)
        even_layer_unitaries2 = torch.cat([self.even_layer_random_unitaries(n_batch, self.n_qubits).to(device) for _ in range((self.n_layers+1)//2)], axis=1)
        odd_layer_unitaries1 = None
        odd_layer_unitaries2 = None
        if self.n_layers > 1:
            odd_layer_unitaries1 = torch.cat([self.odd_layer_random_unitaries(n_batch, self.n_qubits).to(device) for _ in range(self.n_layers//2)], axis=1)
            odd_layer_unitaries2 = torch.cat([self.odd_layer_random_unitaries(n_batch, self.n_qubits).to(device) for _ in range(self.n_layers//2)], axis=1)
        
        results = {}
        results['even_layer_unitaries1'] = even_layer_unitaries1
        results['odd_layer_unitaries1'] = odd_layer_unitaries1
        results['even_layer_unitaries2'] = even_layer_unitaries2
        results['odd_layer_unitaries2'] = odd_layer_unitaries2
        return results

    def forward(self, unitaries):
        out = self.parent_forward(**unitaries)
        return (2**self.n_qubits)*out.abs()


def trace_generation(directory, k):

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    haar_frame_potential = math.factorial(k)
    for n_qubits in [14,16,18,20,24,28,32,36,40,44,48]:
        n_batch = 1600000//n_qubits
        mean = haar_frame_potential
        std = haar_frame_potential
        for n_layers in [4,5,6,7,8,9,10,12,14]:#,150,200,250,300,400,500]:
            if (mean+2*std < 1.5*haar_frame_potential):
                break
            pru = PRU_trace(n_qubits, n_layers, optimizer=TamakiOptimizer(wait_time=2**min(n_qubits, n_layers)))

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

                    iteration = 0
                    
                    while True:
                        if (mean-2*std > haar_frame_potential):
                            break
                        
                        unitaries = pru.layer_random_unitaries(device, n_batch)
                        with qtensor_ai.forward_only():
                            iteration_results = pru(unitaries).cpu()
                        
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
    max_ns = 200
    max_l = 10
    max_k = 5
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


def peo_finder():
    for n_layers in [12,14,16,18,20]:
        for n_qubits in [24,28]:
            optimizer = TamakiOptimizer(wait_time=1500)
            circuit_name = 'PRU_trace_n_{}_l_{}'.format(n_qubits, n_layers)
            PRU_composer = ParallelRandomUnitaryComposer(n_qubits, n_layers)
            composer = TraceEvaluationComposer(n_qubits, PRU_composer)
            pru = PRU_trace(n_qubits, n_layers)
            unitaries = pru.layer_random_unitaries(device='cpu', n_batch=1)
            composer.produce_circuit(**unitaries) # Update circuit variational parameters
            tn, _, _ = ParallelTensorNet.from_qtree_gates(composer.final_circuit)
            _ = circuit_optimization(circuit_name, tn, optimizer, composer)


def test():

    from qtensor_ai.Simulate import ParallelSimulator
    from qtensor_ai.Backend import ParallelTorchBackend

    n_qubits = 10
    n_layers = 3
    n_batch = 10

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    pru = PRU_trace(n_qubits, n_layers, optimizer=DefaultOptimizer)
    com = ParallelRandomUnitaryComposer(n_qubits, n_layers)
    sim = ParallelSimulator(backend=ParallelTorchBackend()) 
    for _ in range(1):
        unitaries = pru.layer_random_unitaries(device, n_batch)
        with qtensor_ai.forward_only():
            com.produce_circuit(**unitaries) # Update circuit variational parameters
            out = sim.simulate_batch(com.final_circuit)
        print(out)
        time.sleep(1)


parser = argparse.ArgumentParser()
parser.add_argument('--mode', help="Mode")
args = vars(parser.parse_args())


if __name__ == '__main__':
    if args['mode'] == 'trace_generation':
        trace_generation(directory = './results/PRU/', k=3)
    elif args['mode'] == 'frame_potential':
        frame_potential(results_dir = './results/PRU/')
    elif args['mode'] == 'test':
        test()
    elif args['mode'] == 'peo_finder':
        peo_finder()
    else:
        print("Invalid Argument")