import qtensor_ai
from qtensor_ai.OpFactory import ParallelParametricGate, ParallelTorchFactory
from qtensor_ai import ParallelComposer
from qtensor_ai.Simulate import ParallelSimulator
from qtensor_ai.Backend import ParallelTorchBackend

import torch
import torch.multiprocessing as mp

import numpy as np
import pickle
from functools import partial
import time
import math
import re
import os

from ..eunn import Unitary


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



        '''This line of code where we used to have self.com.n_batch = self.n_batch is inconsistent with the HybridModule implementation'''
        


        circuit = self.com.updated_full_circuit(**parameters)


        self.n_batch = self.com.n_batch


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


class LocalRandomUnitaryComposer(ParallelComposer):

    def __init__(self, n_qubits, n_layers):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        super().__init__(n_qubits)

    def _get_builder(self):
        return self._get_builder_class()(self.n_qubits)

    def updated_full_circuit(self, **parameters):
        unitaries = parameters['unitaries']
        self.n_batch = unitaries.shape[0]
        self.builder.reset()
        for qubit in range(self.n_qubits):
            self.apply_gate(self.operators.M, qubit)
        for layer in range(self.n_layers):
            qubit = np.random.RandomState().randint(0, self.n_qubits-1)
            #print(unitaries[:, layer].shape)
            self.apply_gate(self.operators.RandU, qubit, qubit+1, alpha=unitaries[:, layer])
        return self.builder.circuit
    
    def name():
        return 'Local_Random'


def random_unitary_generator(n_batch, counts):
    #return Unitary(1 * counts)().unsqueeze(0).expand(n_batch, counts, 4, 4).reshape(n_batch, counts, 2,2,2,2).detach()
    results = Unitary(n_batch * counts)().reshape(n_batch, counts, 2,2,2,2).detach()
    results = torch.permute(results, (0,1,2,4,3,5))
    return results

def get_val(com, trace, sim, unitary):
    trace_circuit = trace.updated_full_circuit(unitaries=unitary)
    result = sim.simulate(trace_circuit).detach()
    return (2**com.n_qubits)*torch.abs(torch.tensor(result))

def get_vals(n_qubits, n_layers, num):
    com = LocalRandomUnitaryComposer(n_qubits, 2*n_layers)
    trace = TraceEvaluationComposer(n_qubits, com)
    sim = ParallelSimulator(backend=ParallelTorchBackend())
    get_val_map_fn = partial(get_val, com, trace, sim)
    unitaries = random_unitary_generator(num, 2*n_layers).unsqueeze(1)
    with mp.Pool(num) as pool:
        result = pool.map(get_val_map_fn, unitaries)
    return result

def get_combined_stat(k, n_qubits, n_layers, nproc, directory):
    samples, mean, std = 0, None, None
    for i in range(1, nproc + 1):
        descriptor_file = directory+'/k_{}_n_{}_l_{}_id_{}.pickle'.format(k, n_qubits, n_layers, i)
        if os.path.isfile(descriptor_file):
            while True:
                try:
                    with open(descriptor_file, 'rb') as infile:
                        descriptor = pickle.load(infile)
                        i_samples, i_mean, i_std =  int(descriptor['samples']), float(descriptor['mean']), float(descriptor['std'])
                        if samples == 0:
                            samples, mean, std = i_samples, i_mean, i_std
                        else:
                            mean = (mean*samples + i_mean*i_samples) / (samples + i_samples)
                            var, i_var =  np.square(std), np.square(i_std)
                            var = (var*samples + i_var*i_samples) / (samples + i_samples)
                            std = np.sqrt(var)
                            samples += i_samples
                        break
                except EOFError:
                    #print('EOFError')
                    time.sleep(1+np.random.random(1)[0])
                    break

        sterr = std/np.sqrt(1+samples) if samples != 0 else None
    
    return samples, mean, std, sterr


def simulation(directory, k, threshold, nproc, id):

    sim = ParallelSimulator(backend=ParallelTorchBackend())
    haar_frame_potential = math.factorial(k)

    for n_qubits in [50]:#[24,28,32,36,40,44,48,50]:

        mean = None
        std = None
        sterr = None
        previous_mean = None
        previous_sterr = None

        for n_layers in [3,3.25,3.5,3.75,4,4.25,4.5,4.75,5]:#,5.25,5.5,5.75,6,6.25,6.5,6.75,7]:#,7.25,7.5,7.75,8,8.25,8.5,8.75,9,9.25,9.5,9.75,10]:#,6,7,8,9,10,11,12,13,14]:
            print('Experiment for n={} and l={} began.'.format(n_qubits, n_layers))
            '''Stop if error is smaller than threshold'''
            if mean != None:
                if (mean+2*sterr < threshold*haar_frame_potential):
                    break
            '''Initialize quantum circuit'''
            com = LocalRandomUnitaryComposer(n_qubits, int(2*n_layers*n_qubits))
            trace = TraceEvaluationComposer(n_qubits, com)

            iteration = 0
            results = torch.empty((0)).to('cpu')
            file = directory + '/n_{}_l_{}_id_{}.pt'.format(n_qubits, n_layers, id)
            if os.path.isfile(file):
                while True:
                    try:
                        results = torch.load(file)
                        results = results[~results.isinf()]
                        break
                    except EOFError:
                        time.sleep(1)

            samples, mean, std, sterr = get_combined_stat(k, n_qubits, n_layers, nproc, directory)
                        
            while True:
                '''Conditions for exiting the loop for refining the frame potential value'''
                if (iteration == 20000):
                    break
                if (samples >= 500000):
                    break
                if mean != None:
                    '''Break if we have bad values'''
                    if (mean == np.inf):
                        break
                    if (mean == np.nan):
                        break
                    if (sterr == np.inf):
                        break
                    if (sterr == np.nan):
                        break
                    '''Break only if current mean is less than previous mean, or:'''
                    #if previous_mean != None:
                    #    if mean+2*sterr < previous_mean+2*previous_sterr:
                    #        if mean > haar_frame_potential:
                    #            if (mean-haar_frame_potential>2*sterr):
                    #                break
                    #            if (mean+2*sterr < threshold*haar_frame_potential):
                    #                break
                    '''Can break if no previous mean recorded.'''
                    #if previous_mean == None:
                    if True:
                        if mean > haar_frame_potential:
                            if (mean-haar_frame_potential>5*sterr):
                                break
                            if (mean+2*sterr < threshold*haar_frame_potential):
                                break
                
                with qtensor_ai.forward_only():
                    iteration_results = get_val(com,trace,sim,random_unitary_generator(1, int(2*n_layers*n_qubits)))

                results = torch.cat((results, iteration_results))

                iteration += 1
                if iteration%100 == 99:
                    id_samples = results.shape[0]
                    power = results**(2*k)
                    id_mean = float(torch.mean(power))
                    id_std = float(torch.std(power))
                    descriptor = {'samples': id_samples, 'mean': id_mean, 'std': id_std}
                    with open(directory + '/k_{}_n_{}_l_{}_id_{}.pickle'.format(k, n_qubits, n_layers, id), 'wb') as outfile:
                        pickle.dump(descriptor, outfile)
                        time.sleep(1)

                    samples, mean, std, sterr = get_combined_stat(k, n_qubits, n_layers, nproc, directory)

                    print('Saving {} samples. At iteration {}, the mean is {}, and the standard error of the mean is {}. The frame potential of a {} design is {}'.format(samples, iteration, mean, sterr, k, haar_frame_potential))
                    torch.save(results, directory + '/n_{}_l_{}_id_{}.pt'.format(n_qubits, n_layers, id))
                    

            print('Experiment for n={} and l={} is over with {} samples. Mean: {}; sterr: {}.'.format(n_qubits, n_layers, samples, mean, sterr))
            torch.save(results, directory + '/n_{}_l_{}_id_{}.pt'.format(n_qubits, n_layers, id))
            previous_mean = mean
            previous_sterr = sterr


def frame_potential(results_dir: str, k: int):
    max_ns = 50
    max_l = 14
    max_k = k
    frame_potential = np.zeros((max_ns, max_l*4, max_k, 3))
    frame_potential[:,:,:,:] = np.nan

    files = os.listdir(results_dir)
    n_pattern = re.compile(r'n_(?:\d*\.\d+|\d+)')
    l_pattern = re.compile(r'l_(?:\d*\.\d+|\d+)')
    for file in files:
        if file.endswith(".pt"):
            n_match = n_pattern.findall(file)
            if len(n_match) != 0:
                n = int(n_match[0][2:])
                l_match = l_pattern.findall(file)
                l = int(float(l_match[0][2:])*4)
                tensor = torch.load(results_dir + file)
                tensor = tensor[~tensor.isinf()]
                this_samples = tensor.shape[0]
                if this_samples == 0:
                    print(file, 'has no samples')
                    continue
                for k in range(1, max_k+1):
                    power = tensor**(2*k)
                    this_mean, this_std = power.mean(), power.std()/np.sqrt(this_samples)
                    this_var = np.square(this_std)
                    try:
                        if np.isnan(frame_potential[n-1][l-1][k-1][0]):
                            frame_potential[n-1][l-1][k-1] = np.array([0,0,0], dtype=float)
                        previous_mean = frame_potential[n-1][l-1][k-1][0]
                        previous_std = frame_potential[n-1][l-1][k-1][1]
                        previous_var = np.square(previous_std)
                        previous_samples = frame_potential[n-1][l-1][k-1][2]
                        samples = this_samples + previous_samples
                        if samples == 0:
                            print(n,l,k, this_samples, previous_samples)
                        frame_potential[n-1][l-1][k-1][2] = samples
                        mean = (previous_mean*previous_samples + this_mean*this_samples)/samples
                        frame_potential[n-1][l-1][k-1][0] = mean
                        var = (previous_var*previous_samples + this_var*this_samples)/samples
                        std = np.sqrt(var)
                        frame_potential[n-1][l-1][k-1][1] = std
                    except IndexError:
                        print('File skipped')
                    #print(n,l,k,mean,std)
            else:
                print('failed with ', file)

    np.save(results_dir + '/frame_potential', frame_potential)


def test():
    n_qubits = 10
    n_layers = n_qubits * 6
    num = 20

    com = LocalRandomUnitaryComposer(n_qubits, 2*n_layers)
    trace = TraceEvaluationComposer(n_qubits, com)
    sim = ParallelSimulator(backend=ParallelTorchBackend())

    start = time.time()
    for _ in range(num):
        print(get_val(com,trace,sim,random_unitary_generator(1, 2*n_layers)[0].unsqueeze(0)))
    stop = time.time()
    print('Time taken for a single evaluation: ', stop - start)
    #start = time.time()
    #for _ in range(10):
    #    print(get_vals(n_qubits, n_layers, num))
    #stop = time.time()
    #print('Time taken for multiprocess evaluation: ', (stop - start)/10)