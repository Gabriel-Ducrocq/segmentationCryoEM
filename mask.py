import torch
import numpy as np
from torch.optim.lr_scheduler import ExponentialLR
from torch import nn
from torch.nn.functional import softmax
import torch.nn.functional as F
import matplotlib.pyplot as plt


N_residue = 1550
N_systems = 20
denom = N_systems*(N_systems-1)
B = 400
S = 100

if B%S != 0:
    print("Not divisible !!!")

nb_per_res = int(B/S)
balance = B - S + 1
bs_per_res = np.empty((N_residue, nb_per_res), dtype=int)
count = np.zeros(N_residue, dtype=int)
for i in range(N_residue):
    print(i)
    start = max(i + balance - B, 0)//S ##Find the smallest window number such that the residue i is inside

    if max(i + balance - B, 0)%S != 0:
        start +=1 ## If there is a rest, means we have to go one more window

    bs_per_res[i, :] = np.arange(start, start+nb_per_res)
    #while i + balance > S*k:
    #    if i + balance > S*k + B:
    #        print("Error")
    #        break

    #    print(k)
    #    count[i] += 1

    #    k += 1

    #print("\n\n")

nb_windows = bs_per_res[-1, -1] + 1
list_weights = []
alpha = torch.randn((nb_windows, N_systems))
weights = torch.nn.Parameter(data=alpha, requires_grad=True)

m = torch.nn.ReLU()

def multiply_windows_weights(weights):
    weights_per_residues = torch.empty((N_residue, N_systems))
    for i in range(N_residue):
        windows_set = bs_per_res[i]  # Extracting the indexes of the windows for the given residue
        weights_per_residues[i, :] = torch.prod(weights[windows_set, :], axis=0)  # Muliplying the weights of all the windows, for each subsystem

    return weights_per_residues

def func(weights):
    #weights2 = F.softmax(weights, dim=0)
    weights_per_residues = multiply_windows_weights(weights)

    attention_softmax = F.softmax(weights_per_residues, dim = 1)
    #attention_softmax = torch.nn.functional.normalize(m(attention_softmax-0.1), eps=0.00001)
    attention_softmax_log = torch.log(attention_softmax)
    prod = attention_softmax_log*attention_softmax
    loss = -torch.sum(prod)

    return loss/N_residue


"""
def func(x):
    loss = 0
    #attention_softmax = softmax(x, dim=0)
    #for i in range(N_systems):
    #    for j in range(N_systems):
    #        if j != i:
    #            loss += 0.5*torch.sum(torch.abs(torch.select(attention_softmax, 1, i) - torch.select(attention_softmax, 1, j)))

    #y = torch.cumsum(x**2, dim=0)

    attention_softmax = softmax(x, dim=1)
    y = 2*(attention_softmax - 1/2)
    loss_signs = -torch.sum((y[1:, :]*y[:-1, :]))
    attention_softmax_log = torch.log(attention_softmax)
    prod = attention_softmax_log*attention_softmax
    loss = -torch.sum(prod)
    return loss /N_residue + 100*loss_signs
"""





#x0 = torch.randn(size=(N_residue, N_systems),requires_grad=True)
x = weights
#optimizer = torch.optim.SGD([x], lr=0.1)
optimizer = torch.optim.Adam([x], lr=0.001)
scheduler = ExponentialLR(optimizer, gamma=0.5)
step = 1000
losses = []
for epoch in range(10):
    print("Epoch:", epoch)
    for i in range(step):
        print("Step:", i)
        optimizer.zero_grad()
        f = func(x)
        losses.append(f.detach().numpy())
        f.backward()
        optimizer.step()

    print("\n\n")
    scheduler.step()



w = multiply_windows_weights(weights)
plt.imshow(w.detach().numpy())
plt.show()



y = torch.softmax(w, dim=1)

plt.imshow(y.detach().numpy(), extent=[1, N_systems, 1, N_residue])#, aspect="auto")
plt.colorbar()
plt.ylabel("Residue number")
plt.xlabel("Domain number")
plt.xticks([1, 2, 3, 4, 5], [1,2,3,4,5])
plt.show()

plt.plot(losses)
plt.show()

