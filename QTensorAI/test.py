import torch
from qtensor_ai import QConv1D, QNN, InnerProduct, TamakiOptimizer
import time

in_channels =  1
out_channels = 1
kernel_size = 5
n_batch = 25
n_qubits = 5
n_layers = 2
sequence_length = kernel_size + 100

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
print(device)

# Defining quantum neural network, quantum convolutioinal layer, and classical convolutional layer
optimizer=TamakiOptimizer(wait_time=5) # If you do not want to use this, remove the optimizer keyword below. wait_time=5 is default

qconv = QConv1D(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, variational_layers=3, optimizer=optimizer).to(device)
qnn = QNN(kernel_size, out_channels).to(device)
innerproduct = InnerProduct(n_qubits=n_qubits, n_layers=n_layers, optimizer=optimizer).to(device)
conv = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size).to(device)

torch.manual_seed(0)

params1 = torch.ones(n_batch, n_qubits, n_layers)
params2 = torch.ones(n_batch, n_qubits, n_layers)
start = time.time()
y = innerproduct(params1, params2)
stop = time.time()
print(y, ' , time for innerproduct ', stop-start)

# Testing QNN
x = torch.rand(n_batch*1, kernel_size, requires_grad=True).to(device)
start = time.time()
qy = qnn(x)
stop = time.time()
loss = qy.sum()
loss.backward()
print('time for qnn', stop-start)

# Testing QConv1D
x1 = torch.rand(n_batch, in_channels, sequence_length, requires_grad=True).to(device)
start = time.time()
qy = qconv(x1)
stop = time.time()
print('time for qconv ', stop-start)
print(qy.shape, qy)
loss = qy.sum()
loss.backward()

# Testing QConv1D with different input
x2 = torch.rand(n_batch, in_channels, sequence_length, requires_grad=True).to(device)
start = time.time()
qy = qconv(x2)
stop = time.time()
print('time for qconv ', stop-start)
print(qy.shape, qy)
loss = qy.sum()
loss.backward()

# Testing QConv1D with the same input as the first time to confirm the output is the same
start = time.time()
qy = qconv(x1)
stop = time.time()
print('time for qconv ', stop-start)
print(qy.shape, qy)
loss = qy.sum()
loss.backward()

# Testing classical convolution
x = torch.rand(n_batch, in_channels, sequence_length, requires_grad=True).to(device)
start = time.time()
cy = conv(x)
stop = time.time()
print('time for classical ', stop-start)
print(cy.shape)
