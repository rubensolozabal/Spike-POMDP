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

    def forward(self, x, mem=None):        
        # thre = self.thresh.data        
        
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
                    temp_spike = self.act(mem-self.thresh, self.gama)
                    spike = temp_spike * self.thresh # spike [N, C, H, W]
                    
                    ### Soft reset ###
                    # mem = mem - spike
                    ### Hard reset ###
                    mem = mem*(1.-spike)

                    spike_pot.append(spike) # spike_pot[0].shape [N, C, H, W]

                x_step = torch.stack(spike_pot,dim=0) # dimension [T, N, C, H, W]  
                x_step = self.merge(x_step)  
                episode_spike_pot.append(x_step)

            x = torch.stack(episode_spike_pot, dim=0)              

            # Store current state
            self.current_mem = mem.detach().clone()

        else:
            x = self.expand(x) # [T * N, C, H, W] -> [T, N, C, H, W]
            spike_pot = []
            for t in range(self.T):
                
                mem = self.tau*mem + x[t, ...] 
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




# class LIF(nn.Module):
#     def __init__(self, T=0, thresh=1.0, tau=1., gama=1.0):
#         super(LIF, self).__init__()
#         self.act = ZIF.apply        
#         # self.thresh = nn.Parameter(torch.tensor([thresh], device='cuda'), requires_grad=False, )
#         self.thresh = torch.tensor([thresh], device='cuda', requires_grad=False)
#         self.tau = tau
#         self.gama = gama
#         self.expand = ExpandTemporalDim(T)
#         self.merge = MergeTemporalDim(T)
#         self.T = T
#         self.mem = 0.

#     def forward(self, x):        
#         # thre = self.thresh.data        
#         x = self.expand(x) # [T * N, C, H, W] -> [T, N, C, H, W]
#         spike_pot = []
#         mem = 0.
#         for t in range(self.T):
            
#             mem = self.tau*mem + x[t, ...] 
#             temp_spike = self.act(mem-self.thresh, self.gama)
#             spike = temp_spike * self.thresh # spike [N, C, H, W]
            
#             ### Soft reset ###
#             # mem = mem - spike
#             ### Hard reset ###
#             mem = mem*(1.-spike)

#             spike_pot.append(spike) # spike_pot[0].shape [N, C, H, W]

#         x = torch.stack(spike_pot,dim=0) # dimension [T, N, C, H, W]               
#         x = self.merge(x)        
#         return x
    

