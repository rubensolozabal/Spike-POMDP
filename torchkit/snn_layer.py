# from cv2 import mean
# from sympy import print_rcode
import torch
import torch.nn as nn
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MergeTemporalDim(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T

    def forward(self, x_seq: torch.Tensor):
        return x_seq.flatten(0, 1).contiguous() # x_seq shape [T, N, C, H, W] -> [T * N, C, H, W]. syntax of flatten, For example, if you have a tensor of shape [a, b, c, d], after applying flatten(0, 1), the shape will become [a*b, c, d]. It essentially merges the first two dimensions.

from typing import List, Tuple
# @torch.jit.script
def my_1d_tolist(x):
    result: List[int] = []
    for i in x:
        result.append(i.item())
    return result



class ExpandTemporalDim(nn.Module):
    def __init__(self, T, dim=0):
        super().__init__()
        self.T = T
        self.dim = dim

    def forward(self, x_seq: torch.Tensor):
        y_shape = []
        y_shape.extend(x_seq.shape[:self.dim])
        y_shape.extend([self.T, int(x_seq.shape[self.dim]/self.T)])
        y_shape.extend(x_seq.shape[self.dim+1:]) 
        return x_seq.view(y_shape) 
    
    # def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
    #     dim_size = x_seq.size(self.dim)
    #     y_shape = torch.tensor(x_seq.shape, dtype=torch.int64)  # Convert shape to tensor
    #     y_shape[self.dim] = self.T
    #     # Use torch.cat to concatenate tensors
    #     y_shape = torch.cat([y_shape[:self.dim+1], torch.tensor([dim_size // self.T], dtype=torch.int64), y_shape[self.dim+1:]])
    #     return x_seq.view(my_1d_tolist(y_shape))  # Convert tensor to tuple for view



class ZIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gama):
        out = (input >= 0).float()
        L = torch.tensor([gama])
        ctx.save_for_backward(input, out, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others) = ctx.saved_tensors
        gama = others[0].item()
        grad_input = grad_output
        tmp = (1 / gama) * (1 / gama) * ((gama - input.abs()).clamp(min=0))
        grad_input = grad_input * tmp
        return grad_input, None


class LIF(nn.Module):
    def __init__(self, T=0, thresh=1.0, tau=1., gama=1.0):
        super(LIF, self).__init__()
        self.act = ZIF.apply       
        # self.thresh = nn.Parameter(torch.tensor([thresh], device='cuda'), requires_grad=False, )
        self.thresh = torch.tensor([thresh], device='cuda', requires_grad=False)
        self.tau = tau
        self.gama = gama
        self.expand = ExpandTemporalDim(T)
        self.merge = MergeTemporalDim(T)
        self.T = T
        self.init_mem = 0.
        self.current_mem = 0.

    def forward(self, x, **kwargs):        
        # thre = self.thresh.data        
        
        mem = kwargs.get("mem", None)
    
    
        if mem is not None:
            mem = mem
        else:
            mem = self.init_mem

        if len(x.shape) == 3:
            steps = x.shape[0] # [steps=200, BS=32 * T=4, embed]
            episode_spike_pot = []
            for step in range(steps):

                x_step = x[step, ...]
                x_step = self.expand(x_step) # [T * N, C, H, W] -> [T, N, C, H, W]

                spike_pot = []
                for t in range(self.T):
                    
                    mem = self.tau*mem + x_step[t, ...]

                    # mem should be bigger than 0
                    # mem = torch.clamp(mem, min=0)
                    
                    # print(mem[0])

                    temp_spike = self.act(mem-self.thresh, self.gama)
                    spike = temp_spike * self.thresh # spike [N, C, H, W]

                    # print(spike[0])
                    
                    ### Soft reset ###
                    # mem = mem - spike
                    ### Hard reset ###
                    mem = mem*(1.-spike)

                    spike_pot.append(spike) # spike_pot[0].shape [N, C, H, W]

                x_step = torch.stack(spike_pot,dim=0) # dimension [T, N, C, H, W]  
                x_step = self.merge(x_step)  
                episode_spike_pot.append(x_step)

            x = torch.stack(episode_spike_pot, dim=0)    

            # print(spike[0])              

            # Store current state
            self.current_mem = mem.detach().clone()

        else:
            x = self.expand(x) # [T * N, C, H, W] -> [T, N, C, H, W]
            spike_pot = []
            for t in range(self.T):
                
                mem = self.tau*mem + x[t, ...] 
                
                # mem should be bigger than 0
                # mem = torch.clamp(mem, min=0)
                
                temp_spike = self.act(mem-self.thresh, self.gama)
                spike = temp_spike * self.thresh # spike [N, C, H, W]

                
                
                ### Soft reset ###
                # mem = mem - spike
                ### Hard reset ###
                mem = mem*(1.-spike)

                spike_pot.append(spike) # spike_pot[0].shape [N, C, H, W]

            x = torch.stack(spike_pot,dim=0) # dimension [T, N, C, H, W]               
            x = self.merge(x)  

            # Store current state
            self.current_mem = mem.detach().clone() 

        return x



class LIF_residue(nn.Module):
    def __init__(self, T=0, thresh=1.0, tau=1., gama=1.0, alpha=0.5):
        super(LIF_residue, self).__init__()
        self.act = ZIF.apply       
        # self.thresh = nn.Parameter(torch.tensor([thresh], device='cuda'), requires_grad=False, )
        self.thresh = torch.tensor([thresh], device='cuda', requires_grad=False)
        self.tau = tau
        self.gama = gama
        self.alpha = alpha
        self.expand = ExpandTemporalDim(T)
        self.merge = MergeTemporalDim(T)
        self.T = T
        self.init_mem = 0.
        self.current_mem = 0.
        self.current_spike_residue = 0.

    def forward(self, x, **kwargs):        
        # thre = self.thresh.data        
        
        mem = kwargs.get("mem", None)
        spike_residue = kwargs.get("spike_residue", None)
    
        if mem is not None:
            mem = mem
            spike_residue = spike_residue
            
        else:
            mem = self.init_mem
            spike_residue = self.init_mem

        if len(x.shape) == 3:
            steps = x.shape[0] # [steps=200, BS=32 * T=4, embed]
            episode_spike_pot = []
            for step in range(steps):

                x_step = x[step, ...]
                x_step = self.expand(x_step) # [T * N, C, H, W] -> [T, N, C, H, W]

                spike_pot = []
                for t in range(self.T):
                    
                    mem = self.tau*mem + x_step[t, ...]

                    # mem should be bigger than 0
                    # mem = torch.clamp(mem, min=0)
                    
                    # print(mem[0])

                    temp_spike = self.act(mem-self.thresh, self.gama)
                    spike = temp_spike * self.thresh # spike [N, C, H, W]
                    spike_residue = self.alpha * spike_residue + spike 

                    # print(spike_residue[0])
                    
                    ### Soft reset ###
                    # mem = mem - spike
                    ### Hard reset ###
                    mem = mem*(1.-spike)

                    spike_pot.append(spike_residue) # spike_pot[0].shape [N, C, H, W]

                x_step = torch.stack(spike_pot,dim=0) # dimension [T, N, C, H, W]  
                x_step = self.merge(x_step)  
                episode_spike_pot.append(x_step)

            x = torch.stack(episode_spike_pot, dim=0)    

            # print(spike[0])              

            # Store current state
            self.current_mem = mem.detach().clone()
            self.current_spike_residue = spike_residue.detach().clone()

        else:
            x = self.expand(x) # [T * N, C, H, W] -> [T, N, C, H, W]
            spike_pot = []
            for t in range(self.T):
                
                mem = self.tau*mem + x[t, ...] 
                
                # mem should be bigger than 0
                # mem = torch.clamp(mem, min=0)
                
                temp_spike = self.act(mem-self.thresh, self.gama)
                spike = temp_spike * self.thresh # spike [N, C, H, W]
                spike_residue = self.alpha * spike_residue + spike 

                # print(spike_residue[0])
                
                
                ### Soft reset ###
                # mem = mem - spike
                ### Hard reset ###
                mem = mem*(1.-spike)

                spike_pot.append(spike_residue) # spike_pot[0].shape [N, C, H, W]

            x = torch.stack(spike_pot,dim=0) # dimension [T, N, C, H, W]               
            x = self.merge(x)  

            # Store current state
            self.current_mem = mem.detach().clone() 
            self.current_spike_residue = spike_residue.detach().clone()

        return x



class STC_LIF(nn.Module):
    def __init__(self, T=0, thresh=1.0, tau=1., gama=1.0):
        super(STC_LIF, self).__init__()
        self.act = ZIF.apply       
        # self.thresh = nn.Parameter(torch.tensor([thresh], device='cuda'), requires_grad=False, )
        self.thresh = torch.tensor([thresh], device='cuda', requires_grad=False)
        self.tau = tau
        self.gama = gama
        self.expand = ExpandTemporalDim(T)
        self.merge = MergeTemporalDim(T)
        self.T = T
        self.init_mem = None
        self.current_mem = 0.
        self.last_spike = 0.

        # Internal weights
        self.W_gt = nn.Linear(128, 128).cuda()
        self.W_gs = nn.Linear(128, 128).cuda()

    def forward(self, x, **kwargs):        

        mem = kwargs.get("mem", None)
        spike = kwargs.get("spike", None)
        


        if len(x.shape) == 3:
            
            if mem is None or spike is None:
                BS = int(x.shape[1]/self.T)
                mem = torch.zeros((BS,128)).cuda()
                spike = torch.zeros((BS,128)).cuda()
            else:
                mem = mem
                spike = spike

        
            steps = x.shape[0] # [steps=200, BS=32 * T=4, embed]
            episode_spike_pot = []
            for step in range(steps):

                x_step = x[step, ...]
                x_step = self.expand(x_step) # [T * N, C, H, W] -> [T, N, C, H, W]

                spike_pot = []
                for t in range(self.T):
                    
                    beta = (1 + torch.tanh(self.W_gt(spike)))/2
                    gamma = (1 + torch.tanh(self.W_gs(spike)))/2

                    mem = mem  + x_step[t, ...] * gamma 

                    # print(mem[0])

                    temp_spike = self.act(mem-self.thresh, self.gama)
                    spike = temp_spike * self.thresh # spike [N, C, H, W]

                    # print(spike[0])
                    
                    ### Soft reset ###
                    # mem = mem - spike
                    ### Hard reset ###
                    mem = mem*(1.-spike)

                    spike_pot.append(spike) # spike_pot[0].shape [N, C, H, W]

                x_step = torch.stack(spike_pot,dim=0) # dimension [T, N, C, H, W]  
                x_step = self.merge(x_step)  
                episode_spike_pot.append(x_step)

            x = torch.stack(episode_spike_pot, dim=0)       

            # print(spike[0])       

            # Store current state
            self.current_mem = mem.detach().clone()
            self.last_spike = spike.detach().clone()
        else:

            if mem is None or spike is None:
                BS = int(x.shape[0]/self.T)
                mem = torch.zeros((BS,128)).cuda()
                spike = torch.zeros((BS,128)).cuda()
            else:
                mem = mem
                spike = spike

  
            x = self.expand(x) # [T * N, C, H, W] -> [T, N, C, H, W]
            spike_pot = []
            
            for t in range(self.T):
                
                beta = (1 + torch.tanh(self.W_gt(spike)))/2
                gamma = (1 + torch.tanh(self.W_gs(spike)))/2

                mem = mem * beta + x[t, ...] * gamma 
                
                temp_spike = self.act(mem-self.thresh, self.gama)
                spike = temp_spike * self.thresh # spike [N, C, H, W]
                
                ### Soft reset ###
                # mem = mem - spike
                ### Hard reset ###
                mem = mem*(1.-spike)

                spike_pot.append(spike) # spike_pot[0].shape [N, C, H, W]

            x = torch.stack(spike_pot,dim=0) # dimension [T, N, C, H, W]               
            x = self.merge(x)  

            # Store current state
            self.current_mem = mem.detach().clone()
            self.last_spike = spike.detach().clone()

        return x