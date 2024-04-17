import torch
from torch import nn

from src.configuration import Configuration
from src.connectivity import Connectivity
from src.activation import Activation
from src.plasticity import Plasticity
from src.ff_input import live_ff_input, init_ff_input

from src.utils import set_seed, clear_cache, print_activity
from src.lr_utils import initLR

import warnings
warnings.filterwarnings("ignore")

class LIFNetwork(nn.Module):

    def __init__(self, conf_name, repo_root, **kwargs):
        '''
        Class: LIFNetwork
        Creates a recurrent network of rate units with customizable connectivity and dynamics.
        The network can be trained with standard torch optimization technics.
        Parameters:
               conf_name: str, name of a .yml file contaning the model's parameters.
               repo_root: str, root path of the NeuroFlame repository.
               **kwargs: **dict, any parameter in conf_file can be passed here
                                 and will then be overwritten.
        Returns:
               rates: tensorfloat of size (N_BATCH, N_STEPS or 1, N_NEURON).
        '''

        super().__init__()

        # Load parameters from configuration file and create networks constants
        config = Configuration(conf_name, repo_root)(**kwargs)
        self.__dict__.update(config.__dict__)

        # Initialize weight matrix
        self.initWeights()

        # Initialize low rank connectivity for training
        if self.LR_TRAIN:
            initLR(self)
        
        # Reset the seed
        set_seed(0)
        clear_cache()

    def initWeights(self):
        '''
        Initializes the connectivity matrix self.Wab.
        Scales weights Jab and loops over blocks to create the full matrix.
        Relies on class Connectivity from connetivity.py
        '''

        # Scale synaptic weights as 1/sqrt(K) for sparse nets
        self.scaleWeights()

        # in pytorch, Wij is i to j.
        self.Wab_T = torch.zeros((self.N_NEURON, self.N_NEURON),  device=self.device)

        # Creates connetivity matrix in blocks
        for i_pop in range(self.N_POP):
            for j_pop in range(self.N_POP):

                weight_mat = Connectivity(self.Na[i_pop], self.Na[j_pop],
                                          self.Ka[j_pop], device=self.device)

                weights = weight_mat(self.CON_TYPE, self.PROBA_TYPE[i_pop][j_pop],
                                     kappa=self.KAPPA[i_pop][j_pop], phase=self.PHASE,
                                     sigma=self.SIGMA[i_pop][j_pop], lr_mean= self.LR_MEAN,
                                     lr_cov=self.LR_COV, ksi=self.PHI0)

                self.Wab_T[self.slices[i_pop], self.slices[j_pop]] = self.Jab[i_pop][j_pop] * weights

        del weights, weight_mat
        # take weights transpose for optim
        self.Wab_T = self.Wab_T.T
        
        # if self.CON_TYPE=='sparse':
        #     self.Wab_T = self.Wab_T.to_sparse()

    def initSTP(self):
        ''' Creates stp model for population 0'''
        self.J_STP = torch.tensor(self.J_STP,  device=self.device)

        self.stp = Plasticity(self.USE, self.TAU_FAC, self.TAU_REC,
                              self.DT, (self.N_BATCH, self.Na[0]),
                              STP_TYPE=self.STP_TYPE,
                              FLOAT=self.FLOAT, device=self.device)

        # NEED .clone() here otherwise BAD THINGS HAPPEN
        self.W_stp_T = self.Wab_T[self.slices[0], self.slices[0]].clone() / self.Jab[0, 0]

    def init_ff_input(self):
        return init_ff_input(self)        
    
    def initVolt(self, ff_input=None):
        if ff_input is None:
            if self.VERBOSE:
                print('Generating ff input')
            
            ff_input = init_ff_input(self)
        else:
            ff_input.to(self.device)
            self.N_BATCH = ff_input.shape[0]
        
        rec_input = torch.randn((self.IF_NMDA+1, self.N_BATCH, self.N_NEURON),  device=self.device)
        volts = torch.zeros((self.N_BATCH, self.N_NEURON),  device=self.device)
        
        # Update spikes
        spikes = (volts>=self.V_THRESH)
        volts[spikes] = self.V_REST
        spikes = spikes * 1.0
        
        return volts, ff_input, rec_input, spikes
    
    def scaleWeights(self):

        # scaling recurrent weights Jab as 1 / sqrt(Kb)
        if self.VERBOSE:
            print("Jab", self.Jab)

        self.Jab = torch.tensor(self.Jab,  device=self.device)
        self.Jab = self.Jab.reshape(self.N_POP, self.N_POP) * self.GAIN
        self.Jab.mul_(self.V_THRESH - self.V_REST)
        
        for i_pop in range(self.N_POP):
            self.Jab[:, i_pop] = self.Jab[:, i_pop] / torch.sqrt(self.Ka[0])
            self.Jab[:, i_pop] = self.Jab[:, i_pop] / self.Ka[i_pop] / self.TAU_SYN[i_pop]
        
        # scaling FF weights as sqrt(K0)
        if self.VERBOSE:
            print("Ja0", self.Ja0)
        
        self.Ja0 = torch.tensor(self.Ja0,  device=self.device)
        self.Ja0.mul_(self.V_THRESH - self.V_REST)
        self.Ja0 = self.Ja0.unsqueeze(0)  # add batch dim
        self.Ja0 = self.Ja0.unsqueeze(-1) # add neural dim
        
        # now inputs are scaled in init_ff_input unless live update
        if self.LIVE_FF_UPDATE:
            self.Ja0.mul_(self.M0 * torch.sqrt(self.Ka[0]))
        
        # scaling ff variance as 1 / sqrt(K0)
        self.VAR_FF = torch.sqrt(torch.tensor(self.VAR_FF,  device=self.device))
        self.VAR_FF.mul_(self.M0 / torch.sqrt(self.Ka[0]))
        self.VAR_FF = self.VAR_FF.unsqueeze(0)  # add batch dim
        self.VAR_FF = self.VAR_FF.unsqueeze(-1) # add neural dim
        
    def update_dynamics(self, volts, ff_input, rec_input, spikes):
        '''LIF Dynamics'''
        
        # update hidden state
        hidden = spikes @ self.Wab_T
        
        # update stp variables
        if self.IF_STP:
            Aux = self.stp(spikes[:, self.slices[0]])  # Aux is now u * x * rates
            hidden_stp = self.J_STP * Aux @ self.W_stp_T
            hidden[:, self.slices[0]].add_(hidden_stp)
        
        # update batched EtoE
        if self.IF_BATCH_J:
            hidden[:, self.slices[0]].add_(self.Jab_batch * spikes[:, self.slices[0]] @ self.W_batch_T)
        
        # update reccurent input
        if self.SYN_DYN:
            rec_input[0] = rec_input[0] * self.EXP_DT_TAU_SYN + hidden * self.DT_TAU_SYN
        else:
            rec_input[0] = hidden
        
        # compute net input
        net_input = ff_input + rec_input[0]
        
        if self.IF_NMDA:
            hidden = spikes[:, self.slices[0]] @ self.Wab_T[self.slices[0]]
            if self.IF_STP:
                hidden[:, self.slices[0]].add_(hidden_stp)

            rec_input[1] = rec_input[1] * self.EXP_DT_TAU_NMDA + self.R_NMDA * hidden * self.DT_TAU_NMDA
            net_input.add_(rec_input[1])

        # Update membrane voltage
        volts = volts * self.EXP_DT_TAU + self.DT_TAU * net_input
        
        # Update spikes
        spikes = volts>=self.V_THRESH
        volts[spikes] = self.V_REST
        spikes = spikes * 1.0
        
        # del hidden, net_input
        
        return volts, rec_input, spikes
    
    def forward(self, ff_input=None, REC_LAST_ONLY=0, RET_FF=0, RET_STP=0):
        '''
        Main method of Network class, runs networks dynamics over set of timesteps
        and returns rates at each time point or just the last time point.
        args:
        :param ff_input: float (N_BATCH, N_STEP, N_NEURONS), ff inputs into the network.
        :param REC_LAST_ONLY: bool, wether to record the last timestep only.
        rates_list:
        :param rates_list: float (N_BATCH, N_STEP or 1, N_NEURONS), rates of the neurons.
        '''

        # Initialization (if  ff_input is None, ff_input is generated)
        volts, ff_input, rec_input, spikes = self.initVolt(ff_input)
        
        # NEED .clone() here otherwise BAD THINGS HAPPEN
        if self.IF_BATCH_J:
            self.W_batch_T = self.Wab_T[self.slices[0], self.slices[0]].clone() / self.Jab[0, 0]
        
        # Add STP
        if self.IF_STP:
            self.initSTP()
            self.x_list, self.u_list = [], []
    
        if self.IF_BATCH_J or self.IF_STP:
            self.Wab_T[self.slices[0], self.slices[0]] = 0

        # Moving average
        
        mv_volts, mv_rates, mv_ff = 0, 0, 0
        volts_list, rates_list, spikes_list, ff_list = [], [], [], []
        
        # Temporal loop
        for step in range(self.N_STEPS):

            # update dynamics
            if self.LIVE_FF_UPDATE:
                ff_input, noise = live_ff_input(self, step, ff_input)
                volts, rec_input, spikes = self.update_dynamics(volts, ff_input + noise, rec_input, spikes)
            else:
                volts, rec_input, spikes = self.update_dynamics(volts, ff_input[:, step], rec_input, spikes)
                
            # update moving average
            mv_rates += spikes
            mv_volts += volts
            
            if self.LIVE_FF_UPDATE and RET_FF:
                mv_ff += ff_input + noise
            
            # Reset moving average to start at 0
            if step == self.N_STEADY-self.N_WINDOW-1:
                mv_volts = 0.0
                mv_rates = 0.0
                mv_ff = 0.0

            # update output every N_WINDOW steps
            if step >= self.N_STEADY:
                if step % self.N_WINDOW == 0:

                    spikes_list.append(spikes)
                    
                    if self.VERBOSE:
                        print_activity(self, step, mv_rates)
                        
                    if not REC_LAST_ONLY:
                        rates_list.append(mv_rates / self.N_WINDOW)
                        volts_list.append(mv_volts / self.N_WINDOW)
                        
                        if self.LIVE_FF_UPDATE and RET_FF:
                            ff_list.append(mv_ff / self.N_WINDOW)
                        if self.IF_STP and RET_STP:
                            self.x_list.append(self.stp.x_stp)
                            self.u_list.append(self.stp.u_stp)

                    # Reset moving average
                    mv_volts, mv_rates, mv_ff = 0, 0, 0

        # returns last step
        rates = mv_rates
        volts = mv_volts
        
        # returns full sequence
        if REC_LAST_ONLY==0:
            # Stack list on 1st dim so that output is (N_BATCH, N_STEPS, N_NEURON)
            rates = torch.stack(rates_list, dim=1)
            del rates_list

            volts = torch.stack(volts_list, dim=1)
            del volts_list

            spikes = torch.stack(spikes_list, dim=1)
            del spikes_list
            
            if self.LIVE_FF_UPDATE and RET_FF:  # returns ff input
                self.ff_input = torch.stack(ff_list, dim=1)
                del ff_list

            if self.IF_STP and RET_STP:  # returns stp u and x
                self.u_list = torch.stack(self.u_list, dim=1)
                self.x_list = torch.stack(self.x_list, dim=1)
        
        # Add Linear readout (N_BATCH, N_EVAL_WIN, 1) on last few steps
        if self.LR_TRAIN:
            y_pred = self.linear(rates[:, -self.lr_eval_win:])
            del rates
            return y_pred.squeeze(-1)
        
        if self.LIVE_FF_UPDATE == 0 and RET_FF:
            self.ff_input = ff_input[..., self.slices[0]]
        
        del ff_input, rec_input
        
        clear_cache()
        
        return rates, volts, spikes
