import os
import numpy as np

import torch
from torch import nn
from torch.distributions import Normal, MultivariateNormal

from yaml import safe_load
from time import perf_counter

from src.utils import set_seed, get_theta, clear_cache
from src.activation import Activation
from src.plasticity import STP_Model

import warnings
warnings.filterwarnings("ignore")

class LifNetwork(nn.Module):
    def __init__(self, conf_file, sim_name, repo_root, **kwargs):
        super().__init__()

        a = 0
    
    def update_voltage(self, volt, net_input):
        '''Dynamics of the membrane'''
        volt = volt * self.EXP_DT_TAU + self.DT * net_input
        return volt
    
    def update_spikes(self, step, volt):
        ''' Spike update'''
        spikes = volt>=self.V_THRESH
        volt[spikes] = self.V_REST
        
        self.isi[spikes] = step - self.spike_time[spikes]
        self.spike_time[spikes] = step
        
        if step >= self.N_STEADY:
            self.spike_count[spikes] += 1
        
        return spikes
        
    def update_rec_input(self, spikes, rec_input, rec_NMDA):
        '''Dynamics of the recurrent inputs'''
        
        A_u_x = 1.0
        if self.IF_STP:
            self.stp.mato_stp(self.isi * self.DT)
            A_u_x = self.stp.A_u_x_stp
        
        if self.SYN_DYN:
            # This also implies dynamics of the feedforward inputs
            rec_input = self.EXP_DT_TAU_SYN * rec_input + self.DT_TAU_SYN * (self.Wab(A_u_x * spikes))
            if self.IF_NMDA:
                rec_NMDA = self.EXP_DT_TAU_NMDA * rec_NMDA + self.DT_TAU_NMDA * (self.Wab(A_u_x * spikes))
        else:
            rec_input = self.Wab(A_u_x * spikes)
        
        return rec_input, rec_NMDA
        
    def update_net_input(self, rec_input, rec_NMDA, ff_input):
        '''Updating the net input into the neurons'''
        
        noise = 0
        if self.VAR_FF[0]>0:
            noise = self.ff_normal.sample((self.N_BATCH, self.N_NEURON,))
        
        net_input = ff_input + noise + rec_input
        
        if self.IF_NMDA:
            net_input += rec_NMDA
        
        return net_input
    
    def forward(self, step, volt, rec_input, rec_NMDA, ff_input):
        
        ff_input = self.update_stim(step)
        net_input = self.update_net_input(rec_input, rec_NMDA, ff_input)
        volt = self.update_voltage(volt, net_input)
        spikes = self.update_spikes(step, volt)
        rec_input, rec_NMDA = self.update_rec_input(spikes, rec_input, rec_NMDA)
        
        return volt, rec_input, rec_NMDA, ff_input
    
    def run(self):
        result = []
        
        with torch.no_grad():
            rates, rec_input, rec_NMDA, ff_input = self.initRates()
            for step in range(self.N_STEPS):
                volt, rec_input, rec_NMDA, ff_input = self.forward(step, volt, rec_input, rec_NMDA, ff_input)
                
                if step >= self.N_STEADY:
                    if step % self.N_WINDOW == 0:
                        rates = self.spike_count / self.N_WINDOW
                        self.spike_count *= 0
                        
                        if self.VERBOSE:
                            self.print_activity(step, rates)
                    
                        if self.REC_LAST_ONLY==0:
                            result.append(rates.cpu().detach().numpy())
        
        if self.REC_LAST_ONLY:
            result.append(rates.cpu().detach().numpy())
        
        result = np.array(result)
        
