import torch
import numpy as np
from Mask1 import CustomizedLinear
from torch import nn, optim
# define mask matrix to customize linear
m1=torch.randint(0, 3, (15, 7))
m2=torch.randint(0, 3, (7, 3))
mask=[]
mask.append(m1)
mask.append(m2)
# define size of layers.
# this architect is [INPUT, HIDDEN(masked(customized) linear), OUTPUT]-layers.
Dim_INPUT  = mask[0].shape[0]
Dim_HIDDEN = mask[1].shape[1]
Dim_OUTPUT = 1

# create randomly input:x, output:y as train dataset.
batch = 1
x = torch.randn(batch, Dim_INPUT)
y = torch.randn(batch, Dim_OUTPUT)

# pipe as model
model = torch.nn.Sequential(
        CustomizedLinear(mask[0], bias=None), # dimmentions is set from mask.size
        CustomizedLinear(mask[1], bias=None),  # dimmentions is set from mask.size
        torch.nn.Linear(Dim_HIDDEN, Dim_OUTPUT, bias=None),
        )

# backward pass
print('=== mask matrix ===')
print(mask)
print('===================')
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
loss_fn = torch.nn.CrossEntropyLoss()
learning_rate=0.1
for t in range(15):
    # forward
    y_pred = model(x)

    # loss
    loss = (y_pred - y).abs().mean()

    # Zero the gradients before running the backward pass.
    optimizer.zero_grad()

    # Use autograd to compute the backward pass
    loss.backward()

    optimizer.step()
    # Update the weights
    with torch.no_grad():
        for param in model.parameters():
            # mask is also saved in param, but mask.requires_grad=False
            #if param.requires_grad:
                #param -= learning_rate * param.grad
                # check masked param.grad
                if np.array(param.grad).size == np.array(mask[1]).size:
                    print('--- epoch={}, loss={} ---'.format(t,loss.item()))
                    print('↓↓↓masked weight↓↓↓')
                    print(param.t())
                    print('↓↓↓masked grad of weight↓↓↓')
                    print(param.grad.t())