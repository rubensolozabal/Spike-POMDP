import torch, torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from snntorch import utils

num_steps = 25 # number of time steps
batch_size = 3
beta = 0.5  # neuron decay rate
spike_grad = surrogate.fast_sigmoid() # surrogate gradient

net = nn.Sequential(
      nn.Conv2d(1, 8, 5),
      nn.MaxPool2d(2),
      snn.Leaky(beta=beta, init_hidden=True, spike_grad=spike_grad),
      nn.Conv2d(8, 16, 5),
      nn.MaxPool2d(2),
      snn.Leaky(beta=beta, init_hidden=True, spike_grad=spike_grad),
      nn.Flatten(),
      nn.Linear(16 * 4 * 4, 10),
      snn.Leaky(beta=beta, init_hidden=True, spike_grad=spike_grad, output=True)
      )

data_in = torch.rand(num_steps, batch_size, 1, 28, 28) # random input data
spike_recording = [] # record spikes over time
utils.reset(net) # reset/initialize hidden states for all neurons

for step in range(num_steps): # loop over time
    spike, state = net(data_in[step]) # one time step of forward-pass
    spike_recording.append(spike) # record spikes in list

pass