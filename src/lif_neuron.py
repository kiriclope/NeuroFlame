import torch
from torch import nn

class LIFCell(nn.Module):
    def __init__(self):
        return self
    
    def update_spikes(self, step, volt):
        ''' Spike update'''
        spikes = volt>=self.V_THRESH
        volt[spikes] = self.V_REST
        
        self.isi[spikes] = step - self.spike_time[spikes]
        self.spike_time[spikes] = step
        
        if step >= self.N_STEADY:
            self.spike_count[spikes] += 1
        
        return volt, spikes
        
    def update_dynamics(self, volt, ff_input, rec_input, spikes):
        '''Dynamics of the recurrent inputs'''
        
        # update hidden state
        hidden = spikes @ self.Wab_T
        
        # update recurrent input
        if self.SYN_DYN:
            rec_input = self.EXP_DT_TAU_SYN * rec_input + self.DT_TAU_SYN * hidden
        else:
            rec_input = hidden

        # update net input 
        net_input = ff_input + rec_input

        # Update membrane voltage
        volt = volt * self.EXP_DT_TAU + self.DT * net_input
        
        return volt, rec_input
            
    def forward(self, ff_input):
        
