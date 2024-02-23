import os
import numpy as np

import torch
from torch import nn
from torch.distributions import MultivariateNormal

from yaml import safe_load
from time import perf_counter

from src.connectivity import Connectivity
from src.activation import Activation
from src.stimuli import Stimuli
from src.plasticity import Plasticity
from src.utils import set_seed, clear_cache

import warnings
warnings.filterwarnings("ignore")

class Network(nn.Module):
    
    def __init__(self, conf_file, sim_name, repo_root, **kwargs):
        '''
        Class: Network
        Creates a recurrent network of rate units with customizable connectivity and dynamics.
        The network can be trained with standard torch optimization technics.
        Parameters:
               conf_file: yml,
               name of the configuration file with network's parameters.
               sim_name: str,
               name of the output file for saving purposes.
               repo_root: str,
               root path for the NeuroTorch repository.
               **kwargs: **dict,
               any parameter in the configuration file can be passed and will then be overwritten.
        Returns:
               rates: tensorfloat of size (N_BATCH, N_SEQ_LEN, N_NEURON).
        '''
        
        super().__init__()
        
        # Load parameters from configuration file
        self.loadConfig(conf_file, sim_name, repo_root, **kwargs)
        
        # Set seed for the connectivity/input vectors
        set_seed(self.SEED)
        
        # Rescale some parameters (time steps, time constants, ...)
        self.initConst()
        
        # Rescale synaptic weights for balance state
        self.scaleParam()
        
        # Initialize network connectivity
        self.initWeights()
        
        # Train a low rank connectivity
        if self.LR_TRAIN:
            # Low rank vector
            self.U = nn.Parameter(torch.randn((self.N_NEURON, int(self.RANK)), device=self.device, dtype=self.FLOAT))
            # self.V = nn.Parameter(torch.randn((self.N_NEURON, int(self.RANK)), device=self.device, dtype=self.FLOAT))

            # Mask to train excitatory neurons only
            self.mask = torch.zeros((self.N_NEURON, self.N_NEURON), device=self.device, dtype=self.FLOAT)
        
            self.mask[self.csumNa[0] : self.csumNa[1],
                      self.csumNa[0] : self.csumNa[1]] = 1.0
            
            # Linear readout for supervised learning
            # self.linear = nn.Linear(self.N_NEURON, 1, device=self.device, dtype=self.FLOAT, bias=False)
            self.linear = nn.Linear(self.Na[0], 1, device=self.device, dtype=self.FLOAT, bias=False)
            
            self.lr_kappa = nn.Parameter(5 * torch.rand(1))
            
            # Window where to evaluate loss
            self.lr_eval_win = int(self.LR_EVAL_WIN / self.DT / self.N_WINDOW)
                        
        # Reset the seed
        set_seed(0)
        
    def initWeights(self):
        '''
        Initializes the connectivity matrix self.Wab.
        Loops over (pre, post) blocks to create the full matrix.
        Relies on class Connectivity from connetivity.py
        '''
        
        # in pytorch, Wij is i to j.
        self.Wab = torch.zeros((self.N_NEURON, self.N_NEURON), dtype=self.FLOAT, device=self.device)
        
        for i_pop in range(self.N_POP):
            for j_pop in range(self.N_POP):
                
                weights = Connectivity(self.Na[i_pop],
                                       self.Na[j_pop],
                                       self.Ka[j_pop])(self.CON_TYPE,
                                                       self.PROBA_TYPE[i_pop][j_pop],
                                                       kappa=self.KAPPA[i_pop][j_pop],
                                                       phase=self.PHASE,
                                                       sigma=self.SIGMA[i_pop][j_pop],
                                                       lr_mean= self.LR_MEAN,
                                                       lr_cov=self.LR_COV,
                                                       ksi=self.PHI0)
                
                self.Wab[self.csumNa[i_pop] : self.csumNa[i_pop + 1],
                         self.csumNa[j_pop] : self.csumNa[j_pop + 1]] = self.Jab[i_pop][j_pop] * weights
        
                del weights
    
    def update_dynamics(self, rates, ff_input, rec_input):
        '''Updates the dynamics of the model at each timestep'''
        
        # update stp variables
        A_u_x = 1.0
        if self.IF_STP:
            A_u_x = self.stp(rates[:, :self.Na[0]])
            stp_ee = (rates[:, :self.Na[0]] * A_u_x) @ self.W_stp.T
            
        if self.LR_TRAIN:
            lr = (1.0 + self.mask * self.KAPPA[0][0] * (self.U @ self.U.T) / torch.sqrt(self.Ka[0]))
            hidden = rates @ (self.Wab.T * lr.T)
        else:
            hidden = rates @ self.Wab.T
        
        if self.IF_STP:
            hidden[:, :self.Na[0]] = hidden[:, :self.Na[0]] + stp_ee
        
        # update thresholds
        # if self.THRESH_DYN:
        #     thresh = self.EXP_DT_TAU_THRESH * thresh + self.DT_TAU_THRESH * (thresh * rates)
        
        # update reccurent input
        if self.SYN_DYN:
            rec_input = self.EXP_DT_TAU_SYN * rec_input + self.DT_TAU_SYN * hidden
        else:
            rec_input = hidden

        del hidden
        
        # compute net input
        net_input = ff_input + rec_input
        non_linear = Activation()(net_input, func_name=self.TF_TYPE, thresh=self.THRESH[0])

        del net_input
        
        # update rates
        if self.RATE_DYN:
            rates = self.EXP_DT_TAU * rates + self.DT_TAU * non_linear
        else:
            rates = non_linear

        del non_linear
        
        return rates, rec_input
    
    def forward(self, ff_input=None, REC_LAST_ONLY=1):
        '''
        Main method of Network class, runs networks dynamics over set of timesteps and returns rates at each time point or just the last time point.
        args:
        :param ff_input: float (N_BATCH, N_TIME, N_NEURONS), ff inputs into the network.
        :param REC_LAST_ONLY: bool, wether to record the last timestep only.
        output:
        :param output: float (N_BATCH, N_TIME or 1, N_NEURONS), rates of the neurons. 
        '''
        
        if self.VERBOSE:
            start = perf_counter()
        
        output = []
        
        # Initialization (if  ff_input is None, ff_input is generated)
        rates, rec_input, self.ff_input = self.initialization(ff_input)

        ################################################
        # WARNING STP WAS NOT TESTED AND MIGHT BE BROKEN
        ###############################################
        # Add STP
        if self.IF_STP:
            self.stp = Plasticity(self.USE, self.TAU_FAC, self.TAU_REC, self.DT, (self.N_BATCH, self.Na[0]), FLOAT=self.FLOAT, device=self.device)
            self.W_stp = self.Wab[:self.Na[0], :self.Na[0]]
            self.Wab[:self.Na[0], :self.Na[0]] = 0
        
        # Moving average of the rates
        mv_rates = 0
        
        # Temporal loop
        for step in range(self.N_STEPS):
            # update dynamics
            rates, rec_input = self.update_dynamics(rates, self.ff_input[:, step], rec_input)

            # update moving average
            mv_rates += rates
            
            # Reset moving average to start at 0
            if step == self.N_STEADY-self.N_WINDOW-1:
                mv_rates *= 0.0
            
            # update output every N_WINDOW steps
            if step >= self.N_STEADY:
                if step % self.N_WINDOW == 0:
                    
                    if self.VERBOSE:
                        self.print_activity(step, rates)
                    
                    # output.append(mv_rates / self.N_WINDOW)
                    if not REC_LAST_ONLY:
                        output.append(mv_rates[..., :self.Na[0]] / self.N_WINDOW)
                    
                    # Reset moving average
                    mv_rates = 0
                    
        del self.Wab
        
        if not REC_LAST_ONLY:
            # Stack output list to 1st dim so that output is (N_BATCH, N_STEPS, N_NEURON)
            output = torch.stack(output, dim=1)
        
        # Add Linear readout (N_BATCH, N_EVAL_WIN, 1) on last few steps
        if self.LR_TRAIN:
            y_pred = self.linear(output[:, -self.lr_eval_win:, ...])
            del output
            return y_pred.squeeze(-1)
        
        if REC_LAST_ONLY:
            output = rates[...,:self.Na[0]]
        
        del rates, rec_input
        
        clear_cache()
        
        if self.VERBOSE:
            end = perf_counter()
            print("Elapsed (with compilation) = {}s".format((end - start)))
        
        return output
    
    def print_activity(self, step, rates):
        
        times = np.round((step - self.N_STEADY) / self.N_STEPS * self.DURATION, 2)
        
        activity = []
        for i in range(self.N_POP):
            activity.append(np.round(torch.mean(rates[:, self.csumNa[i]:self.csumNa[i+1]]).item(), 2))
        
        print("times (s)", times, "rates (Hz)", activity)
    
    def loadConfig(self, conf_file, sim_name, repo_root, **kwargs):
        # Loading configuration file
        conf_path = repo_root + '/conf/'+ conf_file
        # if self.VERBOSE:
        #     print('Loading config from', conf_path)
        param = safe_load(open(conf_path, "r"))
        
        param["FILE_NAME"] = sim_name
        param.update(kwargs)
        
        for k, v in param.items():
            setattr(self, k, v)
        
        self.DATA_PATH = repo_root + "/data/simul/"
        self.MAT_PATH = repo_root + "/data/matrix/"

        if not os.path.exists(self.DATA_PATH):
            os.makedirs(self.DATA_PATH)

        if not os.path.exists(self.MAT_PATH):
            os.makedirs(self.MAT_PATH)
        
        if self.FLOAT_PRECISION == 32:
            self.FLOAT = torch.float
        else:
            self.FLOAT = torch.float64
        
        self.device = torch.device(self.DEVICE)
        
    def initialization(self, ff_input=None):
        if ff_input is None:
            if self.VERBOSE:
                print('generating ff input')
            ff_input = self.init_ff_input()
        else:
            ff_input = ff_input.to(self.device)
            # print('ff_input', ff_input.shape)
            self.N_BATCH = ff_input.shape[0]
        
        rec_input = torch.randn((self.N_BATCH, self.N_NEURON), dtype=self.FLOAT, device=self.device)
        rates = Activation()(ff_input[:, 0] + rec_input, func_name=self.TF_TYPE, thresh=self.THRESH[0])
        
        return rates, rec_input, ff_input

    def initConst(self):
        self.N_STEADY = int(self.T_STEADY / self.DT)
        self.N_WINDOW = int(self.T_WINDOW / self.DT)
        self.N_STEPS = int(self.DURATION / self.DT) + self.N_STEADY + self.N_WINDOW
        
        self.N_STIM_ON = [int(i / self.DT) + self.N_STEADY for i in self.T_STIM_ON]
        self.N_STIM_OFF = [int(i / self.DT) + self.N_STEADY for i in self.T_STIM_OFF]
        
        self.Na = []
        self.Ka = []
        
        if 'all2all' in self.CON_TYPE:
            self.K = 1.0
        
        for i_pop in range(self.N_POP):
            self.Na.append(int(self.N_NEURON * self.frac[i_pop]))
            # self.Ka.append(self.K * const.frac[i_pop])
            self.Ka.append(self.K)
        
        self.Na = torch.tensor(self.Na, dtype=torch.int, device=self.device)
        self.Ka = torch.tensor(self.Ka, dtype=self.FLOAT, device=self.device)
        self.csumNa = torch.cat((torch.tensor([0], device=self.device), torch.cumsum(self.Na, dim=0)))
        
        if self.VERBOSE:
            print("Na", self.Na, "Ka", self.Ka, "csumNa", self.csumNa)

        self.TAU = torch.tensor(self.TAU, dtype=self.FLOAT, device=self.device)
        self.EXP_DT_TAU = torch.ones(self.N_NEURON, dtype=self.FLOAT, device=self.device)
        self.DT_TAU = torch.ones(self.N_NEURON, dtype=self.FLOAT, device=self.device)
        
        for i_pop in range(self.N_POP):
            self.EXP_DT_TAU[self.csumNa[i_pop]:self.csumNa[i_pop + 1]] = torch.exp(-self.DT / self.TAU[i_pop])
            # self.EXP_DT_TAU[self.csumNa[i_pop] : self.csumNa[i_pop + 1]] = (1.0 - self.DT / self.TAU[i_pop])
            self.DT_TAU[self.csumNa[i_pop]:self.csumNa[i_pop + 1]] = self.DT / self.TAU[i_pop]

        # if self.VERBOSE:
        #     print("DT", self.DT, "TAU", self.TAU)
        
        self.TAU_SYN = torch.tensor(self.TAU_SYN, dtype=self.FLOAT, device=self.device)
        self.EXP_DT_TAU_SYN = torch.ones(self.N_NEURON, dtype=self.FLOAT, device=self.device)
        self.DT_TAU_SYN = torch.ones(self.N_NEURON, dtype=self.FLOAT, device=self.device)
        
        for i_pop in range(self.N_POP):
            self.EXP_DT_TAU_SYN[self.csumNa[i_pop]:self.csumNa[i_pop + 1]] = torch.exp(-self.DT / self.TAU_SYN[i_pop])
            # self.EXP_DT_TAU_SYN[self.csumNa[i_pop] : self.csumNa[i_pop + 1]] = (1.0-self.DT / self.TAU_SYN[i_pop])
            self.DT_TAU_SYN[self.csumNa[i_pop]:self.csumNa[i_pop + 1]] = self.DT / self.TAU_SYN[i_pop]
        
        # if self.VERBOSE:
        #     print("EXP_DT_TAU", self.EXP_DT_TAU, "DT_TAU", self.DT_TAU)
        
        self.PROBA_TYPE = np.array(self.PROBA_TYPE).reshape(self.N_POP, self.N_POP)
        self.SIGMA = torch.tensor(self.SIGMA, dtype=self.FLOAT, device=self.device).view(self.N_POP, self.N_POP)
        self.KAPPA = torch.tensor(self.KAPPA, dtype=self.FLOAT, device=self.device).view(self.N_POP, self.N_POP)
        self.PHASE = torch.tensor(self.PHASE * torch.pi / 180.0, dtype=self.FLOAT, device=self.device)
        
        self.VAR_FF = torch.sqrt(torch.tensor(self.VAR_FF, dtype=self.FLOAT, device=self.device))
        
        if self.TASK == 'dual':
            mean_ = torch.tensor(self.LR_MEAN, dtype=self.FLOAT, device=self.device)
            # Define the covariance matrix
            cov_ = torch.tensor(self.LR_COV, dtype=self.FLOAT, device=self.device)
            # print(self.LR_COV)
            multivariate_normal = MultivariateNormal(mean_, cov_)
            
            self.PHI0 = multivariate_normal.sample((self.Na[0],)).T
            
            # if cov_[1][0] == 0 :
            #     self.PHI0[1] = self.PHI0[1] - self.PHI0[1] @ self.PHI0[0] / (self.PHI0[0] @ self.PHI0[0]) * self.PHI0[0]
            
            del mean_, cov_
            del multivariate_normal
            
    def scaleParam(self):

        # scaling recurrent weights Jab
        if self.VERBOSE:
            print("Jab", self.Jab)
        
        self.Jab = torch.tensor(self.Jab, dtype=self.FLOAT, device=self.device).reshape(self.N_POP, self.N_POP) * self.GAIN
        
        for i_pop in range(self.N_POP):
            self.Jab[:, i_pop] = self.Jab[:, i_pop] / torch.sqrt(self.Ka[i_pop])
        
        # scaling FF weights
        if self.VERBOSE:
            print("Ja0", self.Ja0)
        
        self.Ja0 = torch.tensor(self.Ja0, dtype=self.FLOAT, device=self.device)
        # self.Ja0 = self.Ja0 * self.M0 * torch.sqrt(self.Ka[0])
        # now inputs are scaled in init_ff_input
        
    def init_ff_input(self):
        """
        Creates ff input for all timestep
        Inputs can be noisy or not and depend on the task.
        """
        
        # ff_input = torch.zeros(self.N_BATCH, self.N_STEPS, self.N_NEURON, dtype=self.FLOAT, device=self.device)
        ff_input = torch.randn((self.N_BATCH, self.N_STEPS, self.N_NEURON), dtype=self.FLOAT, device=self.device)
        
        for i_pop in range(self.N_POP):
            ff_input[self.csumNa[i_pop] : self.csumNa[i_pop + 1]] = ff_input[self.csumNa[i_pop] : self.csumNa[i_pop + 1]] * self.VAR_FF[i_pop] / torch.sqrt(self.Ka[0])
        
        for i in range(self.N_POP):
            ff_input[:, :self.N_STIM_ON[0], self.csumNa[i]:self.csumNa[i+1]] += self.Ja0[i] / torch.sqrt(self.Ka[0]) # * (1-self.BUMP_SWITCH[i])
        
        for i in range(self.N_POP):
            ff_input[:, self.N_STIM_ON[0]:, self.csumNa[i]:self.csumNa[i+1]] += self.Ja0[i]
        
        for i in range(len(self.N_STIM_ON)):
            
            size = (self.N_BATCH, self.N_STIM_OFF[i]-self.N_STIM_ON[i], self.Na[0])
            
            stimulus = Stimuli(self.TASK, size)(self.I0[i], self.SIGMA0[i], self.PHI0[2*i+1])
            
            ff_input[:, self.N_STIM_ON[i]:self.N_STIM_OFF[i],
                     self.csumNa[0]:self.csumNa[1]] += self.Ja0[0] + stimulus
        
        del stimulus
        
        return ff_input * torch.sqrt(self.Ka[0]) * self.M0 
