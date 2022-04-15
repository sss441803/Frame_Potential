#   Copyright 2018 Floris Laporte
#   MIT License

#   Permission is hereby granted, free of charge, to any person obtaining a copy of this
#   software and associated documentation files (the "Software"), to deal in the Software
#   without restriction, including without limitation the rights to use, copy, modify,
#   merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
#   permit persons to whom the Software is furnished to do so, subject to the following
#   conditions:

#   The above copyright notice and this permission notice shall be included in all copies
#   or substantial portions of the Software.

#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
#   INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
#   PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
#   HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
#   CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR
#   THE USE OR OTHER DEALINGS IN THE SOFTWARE.

""" An Efficient Unitary Neural Network implementation for PyTorch
based on https://arxiv.org/abs/1612.05231
"""

import torch
from math import pi


def _permute(x):
    """ Pairwise permutation of tensor elements along the feature dimension """
    return torch.stack([x[:, 1::2], x[:, ::2]], 2).view(*x.shape)


class Unitary(torch.nn.Module):

    def __init__(self, n_batch):

        self.n_batch = n_batch
        self.identity = torch.eye(4, 4).repeat((n_batch,1,1)) # n_batch x 4 x 4
        super(Unitary, self).__init__()
        self.angles = torch.nn.Parameter(2*pi*torch.rand(self.n_batch, 4, 4))

    def forward(self):
        
        x = self.identity
        # phis and thetas
        phi = self.angles[:, ::2]
        theta = self.angles[:, 1::2]

        # calculate the sin and cos of rotation angles
        cos_phi = torch.cos(phi).unsqueeze(2)
        sin_phi = torch.sin(phi).unsqueeze(2)
        cos_theta = torch.cos(theta).unsqueeze(2)
        sin_theta = torch.sin(theta).unsqueeze(2)

        diag = torch.cat([(cos_phi+1j*sin_phi)*cos_theta, cos_theta], 2).view(-1, 4, 4) # n_batch x 4 x 4
        offdiag = torch.cat([-(cos_phi+1j*sin_phi)*sin_theta, sin_theta], 2).view(-1, 4, 4)

        # loop over sublayers
        for i in range(4):
            layer_diag = diag[:,:,i].repeat((4,1,1)).permute(1,2,0) # n_batch x 4 x 4
            layer_offdiag = offdiag[:,:,i].repeat((4,1,1)).permute(1,2,0)
            x = layer_diag*x + layer_offdiag*_permute(x)
            x = torch.roll(x, 2 * (i % 2) - 1, 1)

        return x