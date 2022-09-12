import torch
from sklearn.utils import resample

import numpy as np
import math
import os
import re
import time
import gc

from batch_size_management import retrieve_batch_sizes, update_batch_sizes



# Evaluates trace values for random unitaries, upto the precision determined by the k-th frame potential
# Higher the k, harder for the circuit to converge to the Haar value
# The argument 'threshold' determines how close the frame potential needs to be to the Haar value for the simulation to stop
def simulation(ansatze_config, trace_gen, k, threshold):

    # A list of integers that is the number of qubits of the circuit
    n_qubits_list = ansatze_config['n_qubits_list']
    # An array of integers that is a list of number of layers for each number of qubits
    n_layers_list = ansatze_config['n_layers_list']
    # Ansatze name. Determines the directories of generated files
    ansatze_name = ansatze_config['ansatze_name']
    directory = "../results/" + ansatze_name
    if not os.path.isdir(directory):
        os.mkdir(directory)
    batch_sizes_file = directory + "/batch_sizes.npy"

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    haar_frame_potential = math.factorial(k)

    for n_qubits in n_qubits_list:

        # we expect the frame potential to decrease as l increases. We keep track of the previous found values
        # to compare to see the the current estimate is sensible and to sufficient precision
        mean = None
        std = None
        previous_mean = None
        previous_std = None

        for n_layers in n_layers_list[n_qubits]:
            
            # knowing what batch size to use and whether or not to try larger or smaller batch sizes
            n_batch, status = retrieve_batch_sizes(50, 14, n_qubits, n_layers, batch_sizes_file)
            # Stop if error is smaller than threshold. This is when the estimate is confidently close enough to the Haar value
            if mean != None:
                if (mean+2*std < threshold*haar_frame_potential):
                    break

            # We keep track of iterations in case it takes forever to get to a desirable precision
            iteration = 0
            # Flag to keep track whether the simulated layer last time is good enough to stop the simulation for this n
            experiment_completed = False

            results = torch.empty((0)).to('cpu')
            # load previously computed trace values
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
                
                # with the chosen batch size, try if it fits in the memory
                try:
                    mean = None
                    std = None
                    samples = results.shape[0]
                    if samples != 0:
                        power = results**(2*k)
                        mean = torch.mean(power)
                        std = torch.std(power)/np.sqrt(1+samples)
                  
                    while True:
                        '''Conditions for exiting the loop for refining the frame potential value'''
                        # If it took too long
                        if (iteration == 200):
                            break
                        # If if will take too long
                        if n_batch != 0:
                            if (2*samples/n_batch >= 1000):
                                break
                        # If cannot fit in memory
                        else:
                            print('Batch size is 0.')
                            exit()
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
                            '''Break only if current mean is less than previous mean, or no previous mean'''
                            if previous_mean != None:
                                # because we know frame potential must decrease, we only stop if the larger layer
                                # frame potential is smaller (here we compare the upper 2*standard error)
                                if mean+2*std < previous_mean+2*previous_std:
                                    # must be greater than the Haar value
                                    if mean > haar_frame_potential:
                                        # when it is definitely larger than the Haar value
                                        if (mean-haar_frame_potential>2*std):
                                            break
                                        # when it is definitely closer to Haar value than the threshold
                                        # but must be greater than the Haar value
                                        if mean+2*std < threshold*haar_frame_potential and mean > haar_frame_potential:
                                            break
                            # Can break if no previous mean recorded
                            else:
                                if (mean-haar_frame_potential>2*std):
                                    break
                                if mean+2*std < threshold*haar_frame_potential and mean > haar_frame_potential:
                                    break
                        
                        # update what batch size to try. May have changed from other jobs or previous attempts
                        n_batch, status = retrieve_batch_sizes(50, 14, n_qubits, n_layers, batch_sizes_file)
                        if status == 1:
                            print('Trying batch size ', n_batch)
                        
                        iteration_results = trace_gen(device, n_batch, n_qubits, n_layers)
                        
                        # read the updated calculated trace values (potentially updated by other jobs)
                        file = directory+'n_{}_l_{}.pt'.format(n_qubits, n_layers)
                        if os.path.isfile(file):
                            while True:
                                try:
                                    results = torch.load(file)
                                    break
                                except EOFError:
                                    time.sleep(1)

                        # updating the trace values and calculate the new frame potential estimate
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
                            update_batch_sizes(n_batch, status, n_qubits, n_layers,batch_sizes_file)

                    print('Experiment for n={} and l={} is over with {} samples. Mean: {}; std: {}.'.format(n_qubits, n_layers, samples, mean, std))
                    torch.save(results, directory + '/n_{}_l_{}.pt'.format(n_qubits, n_layers))
                    previous_mean = mean
                    previous_std = std
                    experiment_completed = True

                # If memory is exceeded, retry with different batch sizes (unless already settled and still failed)
                except RuntimeError:
                    gc.collect()
                    torch.cuda.empty_cache()
                    if status == 2:
                        print("The settled batch size is still exceeding the memory.")
                        raise
                    # If tried to increase and failed, go to settled
                    if status == 1:
                        status = 2
                    if n_batch == 0:
                        print('Cannot fit a single circuit.')
                        raise
                    # Unless cannot fit a single circuit, update the batch size and status
                    else:
                        update_batch_sizes(n_batch, status, n_qubits, n_layers,batch_sizes_file)


# Function that calculates frame potentials for different k values and saves the results
def frame_potential(ansatze_config):

    ansatze_name = ansatze_config['ansatze_name']
    results_dir = "../results/" + ansatze_name + '/'
    
    max_ns = 50
    max_l = 14
    max_k = 5
    # Since not all qubit numbers are calculated, such entries will have value nan
    frame_potential = torch.zeros((max_ns, max_l, max_k, 2))
    frame_potential[:,:,:,:] = np.nan

    # regular expression matching all files that contain trace values
    files = os.listdir('../results/' + ansatze_name)
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
                # If the file is outside what is asked for by the frame potential array
                except IndexError:
                    print('File skipped')
    frame_potential = frame_potential.numpy()
    np.save(results_dir + '/frame_potential', frame_potential)


# (With bootstrapping) Function that calculates frame potentials for different k values and saves the results
def bootstrapped_frame_potential(ansatze_config):

    ansatze_name = ansatze_config['ansatze_name']
    results_dir = "../results/" + ansatze_name + '/'
    
    max_ns = 50
    max_l = 14
    max_k = 5
    bootstrap_samples = 300
    # Since not all qubit numbers are calculated, such entries will have value nan
    frame_potential = np.zeros((max_ns, max_l, max_k, bootstrap_samples, 2))
    frame_potential[:,:,:,:,:] = np.nan
    samples_array = np.zeros((max_ns, max_l))

    # regular expression matching all files that contain trace values
    files = os.listdir('../results/' + ansatze_name)
    n_pattern = re.compile(r'n_\d*')
    l_pattern = re.compile(r'l_\d*')
    for file in files:
        n_match = n_pattern.findall(file)
        if len(n_match) != 0:
            n = int(n_match[0][2:])
            l_match = l_pattern.findall(file)
            l = int(l_match[0][2:])
            loaded_tensor = torch.load(results_dir + file)
            tensor = loaded_tensor[:min(loaded_tensor.shape[0], 100000)]
            del loaded_tensor
            tensor = tensor[~tensor.isinf()]
            samples = tensor.shape[0]
            samples_array[n-1][l-1] = samples
            for k in range(1, max_k+1):
                power = tensor**(2*k)
                #mean, std = power.mean(), power.std()/np.sqrt(samples)
                #print(n,l,k, mean, std)
                print("n, l, k: ", n, l, k)
                try:
                    for i in range(bootstrap_samples):
                        sampled_power = resample(power.numpy(), n_samples=samples)
                        mean = sampled_power.mean().item()
                        std = sampled_power.std().item()/np.sqrt(samples)
                        frame_potential[n-1][l-1][k-1][i][0] = mean
                        frame_potential[n-1][l-1][k-1][i][1] = std
                        #print('Mean, std: ', mean, std)
                # If the file is outside what is asked for by the frame potential array
                except IndexError:
                    print('File skipped')
    np.save(results_dir + '/bootstrapped_frame_potential', frame_potential)
    np.save(results_dir + '/samples_array', samples_array)