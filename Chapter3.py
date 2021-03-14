##    LINK ====>   https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
## NEURAL NETWORKS

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)


"""

You just have to define the forward function, and the backward function (where gradients are 
computed) is automatically defined for you using autograd. You can use any of the Tensor 
operations in the forward function.

The learnable parameters of a model are returned by net.parameters()

"""

params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight

print('------------------------------------------------------------------------------------------')

"""
            ************************                    THIS IS IMPORTANT                      *************************
"""
for x in params:
    print(x.size())
print('------------------------------------------------------------------------------------------')

input = torch.randn(1, 1, 32, 32)
out = net(input)
print(input)
print(out)

## Zero the gradient buffers of all parameters and backprops with random gradients:
net.zero_grad()
out.backward(torch.randn(1, 10))


"""

Recap:

torch.Tensor - A multi-dimensional array with support for autograd operations like backward(). 
Also holds the gradient w.r.t. the tensor.

nn.Module - Neural network module. Convenient way of encapsulating parameters, with helpers for 
moving them to GPU, exporting, loading, etc.

nn.Parameter - A kind of Tensor, that is automatically registered as a parameter when assigned 
as an attribute to a Module.

autograd.Function - Implements forward and backward definitions of an autograd operation. 
Every Tensor operation creates at least a single Function node that connects to functions that 
created a Tensor and encodes its history.

"""

## Loss Function

output = net(input)
target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)

print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

net.zero_grad()     # zeroes the gradient buffers of all parameters

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)


import torch.optim as optim

# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update