###############     LINK  -->  https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
## A GENTLE INTRODUCTION TO TORCH.AUTOGRAD

import torch, torchvision
model = torchvision.models.resnet18(pretrained=True)
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)

# print(model)
# print(data)
# print(labels)
('------------------------------------------------------------------------------------------')


prediction = model(data) # forward pass

loss = (prediction - labels).sum()
loss.backward() # backward pass

optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

optim.step() #gradient descent


a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)

## We create another tensor Q from a and b.
Q = 3*a**3 - b**2

external_grad = torch.tensor([1., 1.])
Q.backward(gradient=external_grad)

# print(a)
# print(b)
# print(Q)

# check if collected gradients are correct
print(9*a**2 == a.grad)
print(-2*b == b.grad)

x = torch.rand(5, 5)
y = torch.rand(5, 5)
z = torch.rand((5, 5), requires_grad=True)

## Exclusion from the DAG
## torch.autograd tracks operations on all tensors which have their requires_grad flag set to True. For tensors that 
# don’t require gradients, setting this attribute to False excludes it from the gradient computation DAG.

## The output tensor of an operation will require gradients even if only a single input tensor has requires_grad=True.
a = x + y
print(f"Does `a` require gradients? : {a.requires_grad}")
b = x + z
print(f"Does `b` require gradients?: {b.requires_grad}")


"""

In a NN, parameters that don’t compute gradients are usually called frozen parameters. 
It is useful to “freeze” part of your model if you know in advance that you won’t need the 
gradients of those parameters (this offers some performance benefits by reducing autograd 
computations).

Another common usecase where exclusion from the DAG is important is for finetuning a pretrained 
network

In finetuning, we freeze most of the model and typically only modify the classifier layers to 
make predictions on new labels. Let’s walk through a small example to demonstrate this. 
As before, we load a pretrained resnet18 model, and freeze all the parameters.



"""

from torch import nn, optim

model = torchvision.models.resnet18(pretrained=True)

# Freeze all the parameters in the network
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(512, 10)

# Optimize only the classifier
optimizer = optim.SGD(model.fc.parameters(), lr=1e-2, momentum=0.9)