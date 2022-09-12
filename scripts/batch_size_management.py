import numpy as np
import os
import time


# Batch sizes must be dynamically determined depending on the GPU memory usage for each circuit size.
# The algorithm starts from a large batch size depending on the number of qubits and layers,
# then iteratively halves the size each time it has a memory error.
# When succeeded once without memory error, the batch size is then increased by 10 percent each time until failure.
# The size before failure is the settled batch size.

# During the process, there are THREE batch size status: 0 (halving), 1 (increasing), and 2 (settled).
# Assuming the same configuration and memory for each job with different GPUs or nodes,
# different jobs can use the same batch size for the same circuit size.

# Because multiple jobs can run at the same time, each may fail at different time or succeed,
# some might move on to the next circuit size before others, the jobs communicate what batch sizes they tested by
# writing to local files, indicating the circuit size, tested batch size and status.

# This allows the algorithm to reuse the same found batch sizes, or continue the search
# instead of starting over when the simulation is restarted.


# Read the local batch size file
# The structure of the file depends on the max number of qubits and layers.
def retrieve_batch_sizes(max_n, max_l, n, l, directory):
    # Initializing batch size file
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
        # Stores the batch sizes. Dims: qubits, layers, batch size and status (0: still halving; 1: still growing; 2: settled)
        batch_sizes = np.zeros((max_n, max_l, 2), dtype=int)
        for n in range(1, max_n+1):
            for l in range(1, max_l+1):
                batch_sizes[n-1, l-1, 0] = int(1600000//min(n, l))
        np.save(directory, batch_sizes)
            
    # Decide what batch size to try
    n_batch = batch_sizes[n-1, l-1, 0]
    status = batch_sizes[n-1, l-1, 1]

    return int(min(n_batch, 1000000)), int(status)


# Update the batch size, status
def update_batch_sizes(n_batch, status, n, l, directory):
    # Initializing batch size file
    assert os.path.isfile(directory), "File for batch sizes not found. Must call retrieve_batch_sizes first."
    while True:
        try:
            batch_sizes = np.load(directory)
            break
        except (EOFError, RuntimeError):
            time.sleep(1)
            print("Failed to load batch_sizes.np. Retrying.")

    assert status in (0,1,2), "Status {} ill-defined.".format(status)

    # Only update if the updating status supercedes or equals to the saved status.
    if status >= batch_sizes[n-1, l-1, 1]:
        # Save the smaller batch size when still halving
        if status == 0:
            temp_size = int(min(n_batch//2, batch_sizes[n-1, l-1, 0]))
            if temp_size == 0:
                print('Batch size 0.')
                batch_sizes[n-1, l-1, 0] = int(1600000//min(n, l))
                batch_sizes[n-1, l-1, 0] = 0
                np.save(directory, batch_sizes)
                exit()
            else:
                batch_sizes[n-1, l-1, 0] = temp_size
        # Save the larger batch size when still growing or settled
        if status == 1:
            temp_size = int(max(n_batch*1.1, batch_sizes[n-1, l-1, 0]))
            if temp_size == n_batch:
                temp_size += 1
            batch_sizes[n-1, l-1, 0] = temp_size
        if status == 2:
            if batch_sizes[n-1, l-1, 1] == 2:
                batch_sizes[n-1, l-1, 0] = batch_sizes[n-1, l-1, 0]
            elif batch_sizes[n-1, l-1, 1] == 1:
                temp_size = int(batch_sizes[n-1, l-1, 0]//1.15)
                if temp_size == batch_sizes[n-1, l-1, 0]:
                    temp_size -= 1
                batch_sizes[n-1, l-1, 0] = temp_size
                print('Settled batch size is ', temp_size)
            else:
                print("Something overwrote the batch size file from status 1 to 0.", batch_sizes[n-1, l-1, 1])
                raise
        batch_sizes[n-1, l-1, 1] = status

    #Saving updated file
    np.save(directory, batch_sizes)