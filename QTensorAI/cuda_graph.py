import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.profiler import profile, record_function, ProfilerActivity

import numpy as np
import gc
import sys
sys.path.append("../../")
from utils import data_generator
from model import TCN
import time


parser = argparse.ArgumentParser(description='Sequence Modeling - Copying Memory Task')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size (default: 32)')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: True)')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='dropout applied to layers (default: 0.0)')
parser.add_argument('--clip', type=float, default=1.0,
                    help='gradient clip, -1 means no clip (default: 1.0)')
parser.add_argument('--epochs', type=int, default=50,
                    help='upper epoch limit (default: 50)')
parser.add_argument('--ksize', type=int, default=8,
                    help='kernel size (default: 8)')
parser.add_argument('--iters', type=int, default=100,
                    help='number of iters per epoch (default: 100)')
parser.add_argument('--levels', type=int, default=8,
                    help='# of levels (default: 8)')
parser.add_argument('--blank_len', type=int, default=1000, metavar='N',
                    help='The size of the blank (i.e. T) (default: 1000)')
parser.add_argument('--seq_len', type=int, default=10,
                    help='initial history size (default: 10)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='report interval (default: 50')
parser.add_argument('--lr', type=float, default=5e-4,
                    help='initial learning rate (default: 5e-4)')
parser.add_argument('--optim', type=str, default='RMSprop',
                    help='optimizer to use (default: RMSprop)')
parser.add_argument('--nhid', type=int, default=10,
                    help='number of hidden units per layer (default: 10)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
args = parser.parse_args()

torch.manual_seed(args.seed)

cuda = False
if torch.cuda.is_available():
    cuda = True


batch_size = args.batch_size
seq_len = args.seq_len    # The size to memorize
epochs = args.epochs
iters = args.iters
T = args.blank_len
n_steps = T + (2 * seq_len)
n_classes = 10  # Digits 0 - 9
n_train = 1000
n_test = 100

print(args)

channel_sizes = [args.nhid] * args.levels
kernel_size = args.ksize
dropout = args.dropout

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
print(device)

print("Preparing data...")
train_x, train_y = data_generator(T, seq_len, n_train)
test_x, test_y = data_generator(T, seq_len, n_test)
model = TCN(1, n_classes, channel_sizes, kernel_size, dropout=dropout).to(device)
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

print('Before stream s, allocated: ', torch.cuda.memory_allocated('cuda'), '; reserved: ', torch.cuda.memory_reserved('cuda'))
# warmup
# Uses static_input and static_target here for convenience,
# but in a real setting, because the warmup includes optimizer.step()
# you must use a few batches of real data.
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for i in range(3):
        inputs = train_x[batch_size*i:batch_size*(i+1)].unsqueeze(1).contiguous().to(device)
        labels = train_y[batch_size*i:batch_size*(i+1)].to(device)
        start = time.time()
        optimizer.zero_grad(set_to_none=True)
        y_pred = model(inputs)
        loss = criterion(y_pred.view(-1, n_classes), labels.view(-1))
        stop = time.time()
        #print('Time for iteration is ', stop - start, 'seconds.')
        start = time.time()
        loss.backward()
        optimizer.step()
        stop = time.time()
        #print('Time for backward pass is ', stop - start, 'seconds.')
torch.cuda.current_stream().wait_stream(s)

gc.collect()
torch.cuda.empty_cache()

print('After stream s before graph g, allocated: ', torch.cuda.memory_allocated('cuda'), '; reserved: ', torch.cuda.memory_reserved('cuda'))
print('Start capturing CUDA graph.')

start = time.time()
# capture
train_graph = torch.cuda.CUDAGraph()
#evaluate_graph = torch.cuda.CUDAGraph()
# Sets grads to None before capture, so backward() will create
# .grad attributes with allocations from the graph's private pool
static_inputs_train = train_x[:batch_size].unsqueeze(1).contiguous().to(device)
static_labels_train = train_y[:batch_size].to(device)
static_inputs_evaluate = train_x[:batch_size].unsqueeze(1).contiguous().to(device)
static_labels_evaluate = train_y[:batch_size].to(device)

with torch.cuda.graph(train_graph):
    optimizer.zero_grad(set_to_none=True)
    static_out_train = model(static_inputs_train)
    static_loss_train = criterion(static_out_train.view(-1, n_classes), static_labels_train.view(-1))
    static_loss_train.backward()
    optimizer.step()

#optimizer.zero_grad(set_to_none=True)
#with torch.cuda.graph(evaluate_graph):
#    static_out_evaluate = model(static_inputs_evaluate)
#    static_loss_evaluate = criterion(static_out_evaluate.view(-1, n_classes), static_labels_evaluate.view(-1))
    
stop = time.time()
print('Time for capturing CUDA graph is ', stop - start, 'seconds.')
print('After graph g before replay, allocated: ', torch.cuda.memory_allocated('cuda'), '; reserved: ', torch.cuda.memory_reserved('cuda'))

def train(ep):
    global batch_size, seq_len, iters, epochs
    total_loss = 0
    start_time = time.time()
    correct = 0
    counter = 0
    for batch_idx, batch in enumerate(range(0, n_train, batch_size)):
        start_ind = batch
        end_ind = start_ind + batch_size

        inputs = train_x[start_ind:end_ind].unsqueeze(1).contiguous().to(device)
        labels = train_y[start_ind:end_ind].to(device)
        static_inputs_train.copy_(inputs)
        static_labels_train.copy_(labels)
        train_graph.replay()
        out = torch.clone(static_out_train)
        pred = out.view(-1, n_classes).data.max(1, keepdim=True)[1]
        correct += pred.eq(labels.data.view_as(pred)).cpu().sum()
        counter += out.view(-1, n_classes).size(0)
        loss = torch.clone(static_loss_train)
        total_loss += loss.item()

        if batch_idx > 0 and batch_idx % args.log_interval == 0:
            avg_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| Epoch {:3d} | {:5d}/{:5d} batches | lr {:2.5f} | ms/batch {:5.2f} | '
                  'loss {:5.8f} | accuracy {:5.4f}'.format(
                ep, batch_idx, n_train // batch_size+1, args.lr, elapsed * 1000 / args.log_interval,
                avg_loss, 100. * correct / counter))
            start_time = time.time()
            total_loss = 0
            correct = 0
            counter = 0


def evaluate():
    global batch_size
    model.eval()
    with torch.no_grad():
        if cuda:
            loss = torch.zeros(1).cuda()
        else:
            loss = torch.zeros(1)
        correct = 0
        counter = 0
        for batch_idx, batch in enumerate(range(0, n_test, batch_size)):
            start_ind = batch
            end_ind = start_ind + batch_size

            x = test_x[start_ind:end_ind].to(device)
            y = test_y[start_ind:end_ind].to(device)
            out = model(x.unsqueeze(1).contiguous())
            loss += criterion(out.view(-1, n_classes), y.view(-1))
            pred = out.view(-1, n_classes).data.max(1, keepdim=True)[1]
            correct += pred.eq(y.data.view_as(pred)).cpu().sum()
            counter += out.view(-1, n_classes).size(0)

        print('\nTest set: Average loss: {:.8f}  |  Accuracy: {:.4f}\n'.format(
            loss.item()/(n_test//batch_size), 100. * correct / counter))
        return loss.item()
        

for ep in range(1, epochs + 1):
    train(ep)
    print('evaluating')
    evaluate()