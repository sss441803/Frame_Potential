import torch
import torch.nn as nn

import numpy as np
import math
import os
import re
import time
import gc

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

    def random_circuit_odd_layer(self, layer_unitaries):
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


def retrieve_batch_sizes(max_n, max_l, n, l, directory):
    '''Initializing batch size file'''
    if os.path.isfile(directory):
        while True:
            try:
                batch_sizes = np.load(directory)
                assert batch_sizes.shape == (max_n, max_l, 2), "Retrieved batch size array shape is unexpected: {}".format(batch_sizes.shape)
                break
            except (EOFError, RuntimeError):
                time.sleep(1)
                print("Failed to load batch_sizes.np. Retrying.")
    else:
        '''Stores the batch sizes. Dims: qubits, layers, batch size and status (0: still halving; 1: still growing; 2: settled)'''
        batch_sizes = np.zeros((max_n, max_l, 2), dtype=int)
        for n in range(1, max_n+1):
            for l in range(1, max_l+1):
                batch_sizes[n-1, l-1, 0] = int(160000//min(n, l))
        np.save(directory, batch_sizes)
            
    '''Decide what batch size to try'''
    n_batch = batch_sizes[n-1, l-1, 0]
    status = batch_sizes[n-1, l-1, 1]

    return int(n_batch), int(status)


def update_batch_sizes(n_batch, status, n, l, directory):
    '''Initializing batch size file'''
    assert os.path.isfile(directory), "File for batch sizes not found. Must call retrieve_batch_sizes first."
    while True:
        try:
            batch_sizes = np.load(directory)
            break
        except (EOFError, RuntimeError):
            time.sleep(1)
            print("Failed to load batch_sizes.np. Retrying.")

    assert status in (0,1,2), "Status {} ill-defined.".format(status)

    '''Only update if the updating status supercedes or equals to the saved status.'''
    if status >= batch_sizes[n-1, l-1, 1]:
        '''Save the smaller batch size when still halving'''
        if status == 0:
            batch_sizes[n-1, l-1, 0] = int(min(n_batch//2, batch_sizes[n-1, l-1, 0]))
        '''Save the larger batch size when still growing or settled'''
        if status == 1:
            batch_sizes[n-1, l-1, 0] = int(max(n_batch*1.1, batch_sizes[n-1, l-1, 0]))
        if status == 2:
            if batch_sizes[n-1, l-1, 1] == 2:
                batch_sizes[n-1, l-1, 0] = batch_sizes[n-1, l-1, 0]
            elif batch_sizes[n-1, l-1, 1] == 1:
                batch_sizes[n-1, l-1, 0] = int(batch_sizes[n-1, l-1, 0]//1.1)
                print('Settled batch size is ', batch_sizes[n-1, l-1, 0])
            else:
                print("Something overwrote the batch size file from status 1 to 0.", batch_sizes[n-1, l-1, 1])
                raise
        batch_sizes[n-1, l-1, 1] = status

    '''Saving updated file'''
    np.save(directory, batch_sizes)


def trace_generation(directory, k, threshold):

    batch_sizes_dir = "results/PRU/batch_sizes.npy"

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    haar_frame_potential = math.factorial(k)

    for n_qubits in [4,6,8,10,12,14,16,18,20,24,28,32,36,40,44,48]:

        mean = None
        std = None
        previous_mean = None
        previous_std = None

        for n_layers in [4,5,6,7,8,9,10,11,12,13,14]:#,150,200,250,300,400,500]:
            '''Stop if error is smaller than threshold'''
            if mean != None:
                if (mean+2*std < threshold*haar_frame_potential):
                    break
            '''Initialize quantum circuit'''
            pru = PRU_trace(n_qubits, n_layers, optimizer=TamakiOptimizer(wait_time=min(2**min(n_qubits,n_layers), 1500)))
            '''Update saved batch_sizes'''

            iteration = 0
            experiment_completed = False

            results = torch.empty((0)).to('cpu')
            file = directory+'n_{}_l_{}.pt'.format(n_qubits, n_layers)
            if os.path.isfile(file):
                while True:
                    try:
                        results = torch.load(file)
                        results = results[~results.isinf()]
                        break
                    except (EOFError, RuntimeError):
                        time.sleep(1)

            while not experiment_completed:
                
                try:
                    '''calculating what the batch size should be based on memory usage'''
                    mean = None
                    std = None
                    samples = results.shape[0]
                    if samples != 0:
                        power = results**(2*k)
                        mean = torch.mean(power)
                        std = torch.std(power)/np.sqrt(1+samples)
                  
                    while True:
                        '''Conditions for exiting the loop for refining the frame potential value'''
                        if (iteration == 200):
                            break
                        if mean != None:
                            '''Break if we have bad values'''
                            if (mean == np.inf):
                                break
                            if (mean == np.nan):
                                break
                            if (std == np.inf):
                                break
                            if (std == np.nan):
                                break
                            '''Break only if current mean is less than previous mean, or:'''
                            if previous_mean != None:
                                if mean+2*std < previous_mean+2*previous_std:
                                    if mean > haar_frame_potential:
                                        if (mean-haar_frame_potential>2*std):
                                            break
                                        if (mean+2*std < threshold*haar_frame_potential):
                                            break
                            '''Can break if no previous mean recorded.'''
                            if previous_mean == None:
                                if mean > haar_frame_potential:
                                    if (mean-haar_frame_potential>2*std):
                                        break
                                    if (mean+2*std < threshold*haar_frame_potential):
                                        break

                        n_batch, status = retrieve_batch_sizes(48, 14, n_qubits, n_layers, batch_sizes_dir)
                        if status == 1:
                            print('Trying batch size ', n_batch)
                        
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

                        iteration += 1
                        if iteration == 1:
                            print('Batch size for n={} and l={} is {}'.format(n_qubits, n_layers, n_batch))
                        print('Saving {} samples. At iteration {}, the mean is {}, and the standard deviation of the mean is {}. The frame potential of a {} design is {}'.format(samples, iteration, mean, std, k, haar_frame_potential))
                        torch.save(results, directory + '/n_{}_l_{}.pt'.format(n_qubits, n_layers))
                        if status == 0:
                            status = 1
                        if status in (1, 2):
                            update_batch_sizes(n_batch, status, n_qubits, n_layers, batch_sizes_dir)

                    print('Experiment for n={} and l={} is over.'.format(n_qubits, n_layers))
                    torch.save(results, directory + '/n_{}_l_{}.pt'.format(n_qubits, n_layers))
                    previous_mean = mean
                    previous_std = std
                    experiment_completed = True

                except RuntimeError:
                    gc.collect()
                    torch.cuda.empty_cache()
                    if status == 2:
                        print("The settled batch size is still exceeding the memory.")
                        raise
                    if status == 1:
                        status = 2
                    update_batch_sizes(n_batch, status, n_qubits, n_layers, batch_sizes_dir)
                    if n_batch == 0:
                        print('Cannot fit a single circuit.')
                        raise


def frame_potential(results_dir: str):
    max_ns = 48
    max_l = 14
    max_k = 4
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
            tensor = tensor[~tensor.isinf()]
            samples = tensor.shape[0]
            for k in range(1, max_k+1):
                power = tensor**(2*k)
                mean, std = power.mean(), power.std()/np.sqrt(samples)
                print(n,l,k, mean, std)
                try:
                    frame_potential[n-1][l-1][k-1][0] = mean
                    frame_potential[n-1][l-1][k-1][1] = std
                except IndexError:
                    print('File skipped')
    frame_potential = frame_potential.numpy()
    np.save(results_dir + '/frame_potential', frame_potential)


def peo_finder():
    for n_layers in [13]:
        for n_qubits in [2,4,6,8,10,12,14,16,18,20,24,28,32,36,40,44,48]:
            optimizer = TamakiOptimizer(wait_time=min(2**min(n_qubits,n_layers), 1500))
            circuit_name = 'PRU_trace_n_{}_l_{}'.format(n_qubits, n_layers)
            PRU_composer = ParallelRandomUnitaryComposer(n_qubits, n_layers)
            composer = TraceEvaluationComposer(n_qubits, PRU_composer)
            pru = PRU_trace(n_qubits, n_layers)
            unitaries = pru.layer_random_unitaries(device='cpu', n_batch=1)
            composer.produce_circuit(**unitaries) # Update circuit variational parameters
            tn, _, _ = ParallelTensorNet.from_qtree_gates(composer.final_circuit)
            _ = circuit_optimization(circuit_name, tn, optimizer, composer)


