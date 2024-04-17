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

class LIFNetwork(nn.Module):
    
    def __init__(self, conf_file, sim_name, repo_root, **kwargs):
        '''
        Class: Network
        Creates a recurrent network of lif units with customizable connectivity and dynamics.
        The network can be trained with standard torch optimization technics.
        Parameters:
               conf_file: yml,
               name of the configuration file with network's parameters.
               sim_name: str,
               name of the output file for saving purposes.
               repo_root: str,
               root path for the NeuroFlame repository.
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
            self.U = nn.Parameter(torch.randn((self.N_NEURON, int(self.RANK)), device=self.device))
            # self.V = nn.Parameter(torch.randn((self.N_NEURON, int(self.RANK)), device=self.device))

            # Mask to train excitatory neurons only
            self.mask = torch.zeros((self.N_NEURON, self.N_NEURON), device=self.device)
            self.mask[self.slices[0], self.slices[0]] = 1.0
            
            # Linear readout for supervised learning
            # self.linear = nn.Linear(self.N_NEURON, 1, device=self.device,  bias=False)
            self.linear = nn.Linear(self.Na[0], 1, device=self.device,  bias=False)
            
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
        self.Wab_T = torch.zeros((self.N_NEURON, self.N_NEURON),  device=self.device)
        
        for i_pop in range(self.N_POP):
            for j_pop in range(self.N_POP):
                
                weights = Connectivity(self.Na[i_pop],
                                       self.Na[j_pop],
                                       self.Ka[j_pop],
                                       device=self.device)(self.CON_TYPE,
                                                           self.PROBA_TYPE[i_pop][j_pop],
                                                           kappa=self.KAPPA[i_pop][j_pop],
                                                           phase=self.PHASE,
                                                           sigma=self.SIGMA[i_pop][j_pop],
                                                           lr_mean= self.LR_MEAN,
                                                           lr_cov=self.LR_COV,
                                                           ksi=self.PHI0)
                
                self.Wab_T[self.slices[i_pop], self.slices[j_pop]] = self.Jab[i_pop][j_pop] * weights
                
        del weights

        self.Wab_T = self.Wab_T.T
        # if self.CON_TYPE=='sparse':
        #     self.Wab_T = self.Wab_T.to_sparse()
        
    def update_dynamics(self, volt, ff_input, rec_input, spikes):
        '''LIF Dynamics'''
        
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
        volt = volt * self.EXP_DT_TAU + self.DT_TAU * net_input
        
        # Update spikes
        spikes = volt>=self.V_THRESH
        volt[spikes] = self.V_REST
        spikes = spikes * 1.0
        
        return volt, rec_input, spikes
    
        
    def forward(self, ff_input=None, REC_LAST_ONLY=0):
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
        
        # Initialization (if  ff_input is None, ff_input is generated)
        volt, ff_input, rec_input, spikes = self.initialization(ff_input)
        
        # Moving average of the rates
        mv_rates = 0
        
        output = []
        # Temporal loop
        for step in range(self.N_STEPS):
            # update dynamics
            volt, rec_input, spikes = self.update_dynamics(volt, ff_input[:, step], rec_input, spikes)
            
            # update moving average
            mv_rates += spikes
            
            # Reset moving average to start at 0
            if step == self.N_STEADY-self.N_WINDOW-1:
                mv_rates *= 0.0
            
            # update output every N_WINDOW steps
            if step >= self.N_STEADY:
                if step % self.N_WINDOW == 0:
                    
                    if self.VERBOSE:
                        self.print_activity(step, mv_rates)
                    
                    if not REC_LAST_ONLY:
                        output.append(mv_rates)
                    
                    # Reset moving average
                    mv_rates = 0
        
        if not REC_LAST_ONLY:
            # Stack output list to 1st dim so that output is (N_BATCH, N_STEPS, N_NEURON)
            output = torch.stack(output, dim=1)
        
        # Add Linear readout (N_BATCH, N_EVAL_WIN, 1) on last few steps
        if self.LR_TRAIN:
            y_pred = self.linear(output[:, -self.lr_eval_win:, ...])
            del output
            return y_pred.squeeze(-1)
        
        if REC_LAST_ONLY:
            output = mv_rates
        
        self.ff_input = ff_input
        del volt, ff_input, rec_input, spikes
        
        clear_cache()
        
        if self.VERBOSE:
            end = perf_counter()
            print("Elapsed (with compilation) = {}s".format((end - start)))
        
        return output
    
    def print_activity(self, step, rates):
        
        times = np.round((step - self.N_STEADY) / self.N_STEPS * self.DURATION, 2)
        
        activity = []
        for i in range(self.N_POP):
            activity.append(np.round(torch.mean(rates[:, self.slices[i]]).item(), 2))
            
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
        
        rec_input = torch.zeros((self.N_BATCH, self.N_NEURON),  device=self.device)
        volt = torch.zeros((self.N_BATCH, self.N_NEURON),  device=self.device)
        
        # Update spikes
        spikes = volt>=self.V_THRESH
        volt[spikes] = self.V_REST
        spikes = spikes * 1.0
        
        return volt, ff_input, rec_input, spikes

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
            self.Ka.append(self.K * self.frac[i_pop])
            # self.Ka.append(self.K)
        
        self.Na = torch.tensor(self.Na, dtype=torch.int, device=self.device)
        self.Ka = torch.tensor(self.Ka,  device=self.device)
        self.csumNa = torch.cat((torch.tensor([0], device=self.device), torch.cumsum(self.Na, dim=0)))

        self.slices = []
        for i_pop in range(self.N_POP):
            self.slices.append(slice(self.csumNa[i_pop], self.csumNa[i_pop + 1]))
        
        if self.VERBOSE:
            print("Na", self.Na, "Ka", self.Ka, "csumNa", self.csumNa)

        self.TAU = torch.tensor(self.TAU,  device=self.device)
        self.EXP_DT_TAU = torch.ones(self.N_NEURON,  device=self.device)
        self.DT_TAU = torch.ones(self.N_NEURON,  device=self.device)
        
        for i_pop in range(self.N_POP):
            self.EXP_DT_TAU[self.slices[i_pop]] = torch.exp(-self.DT / self.TAU[i_pop])
            # self.EXP_DT_TAU[self.csumNa[i_pop] : self.csumNa[i_pop + 1]] = (1.0 - self.DT / self.TAU[i_pop])
            self.DT_TAU[self.slices[i_pop]] = self.DT / self.TAU[i_pop]

        # if self.VERBOSE:
        #     print("DT", self.DT, "TAU", self.TAU)
        
        self.TAU_SYN = torch.tensor(self.TAU_SYN,  device=self.device)
        self.EXP_DT_TAU_SYN = torch.ones(self.N_NEURON,  device=self.device)
        self.DT_TAU_SYN = torch.ones(self.N_NEURON,  device=self.device)
        
        for i_pop in range(self.N_POP):
            self.EXP_DT_TAU_SYN[self.slices[i_pop]] = torch.exp(-self.DT / self.TAU_SYN[i_pop])
            # self.EXP_DT_TAU_SYN[self.csumNa[i_pop] : self.csumNa[i_pop + 1]] = (1.0-self.DT / self.TAU_SYN[i_pop])
            self.DT_TAU_SYN[self.slices[i_pop]] = self.DT / self.TAU_SYN[i_pop]
        
        # if self.VERBOSE:
        #     print("EXP_DT_TAU", self.EXP_DT_TAU, "DT_TAU", self.DT_TAU)
        
        self.PROBA_TYPE = np.array(self.PROBA_TYPE).reshape(self.N_POP, self.N_POP)
        self.SIGMA = torch.tensor(self.SIGMA,  device=self.device).view(self.N_POP, self.N_POP)
        self.KAPPA = torch.tensor(self.KAPPA,  device=self.device).view(self.N_POP, self.N_POP)
        self.PHASE = torch.tensor(self.PHASE * torch.pi / 180.0,  device=self.device)
        
        self.VAR_FF = torch.sqrt(torch.tensor(self.VAR_FF,  device=self.device))
        
        if self.PROBA_TYPE[0][0] == 'lr':
            mean_ = torch.tensor(self.LR_MEAN,  device=self.device)
            # Define the covariance matrix
            cov_ = torch.tensor(self.LR_COV,  device=self.device)
            # print(self.LR_COV)
            multivariate_normal = MultivariateNormal(mean_, cov_)
            
            self.PHI0 = multivariate_normal.sample((self.Na[0],)).T
                        
            del mean_, cov_
            del multivariate_normal
            
    def scaleParam(self):

        # scaling recurrent weights Jab
        if self.VERBOSE:
            print("Jab", self.Jab)
        
        self.Jab = torch.tensor(self.Jab,  device=self.device).reshape(self.N_POP, self.N_POP) * self.GAIN * (self.V_THRESH - self.V_REST)
        
        for i_pop in range(self.N_POP):
            self.Jab[:, i_pop] = self.Jab[:, i_pop] * torch.sqrt(self.Ka[0]) / self.Ka[i_pop] / self.TAU_SYN[i_pop]
        
        # scaling FF weights
        if self.VERBOSE:
            print("Ja0", self.Ja0)
        
        self.Ja0 = torch.tensor(self.Ja0,  device=self.device) * (self.V_THRESH - self.V_REST)
        
    def init_ff_input(self):
        """
        Creates ff input for all timestep
        Inputs can be noisy or not and depend on the task.
        """
        
        # ff_input = torch.zeros(self.N_BATCH, self.N_STEPS, self.N_NEURON,  device=self.device)
        ff_input = torch.randn((self.N_BATCH, self.N_STEPS, self.N_NEURON),  device=self.device)
        
        for i_pop in range(self.N_POP):
            ff_input[..., self.slices[i_pop]] = ff_input[..., self.slices[i_pop]] * self.VAR_FF[i_pop] / torch.sqrt(self.Ka[0])
        
        for i_pop in range(self.N_POP):
            if self.BUMP_SWITCH[i_pop]:
                ff_input[:, :self.N_STIM_ON[0], self.slices[i_pop]] += self.Ja0[i_pop] / torch.sqrt(self.Ka[0])
            else:
                ff_input[:, :self.N_STIM_ON[0], self.slices[i_pop]] += self.Ja0[i_pop]
        
        for i_pop in range(self.N_POP):
            ff_input[:, self.N_STIM_ON[0]:, self.slices[i_pop]] += self.Ja0[i_pop]
        
        if self.TASK != 'None':
            for i ,_ in enumerate(self.N_STIM_ON):            
                size = (self.N_BATCH, self.N_STIM_OFF[i]-self.N_STIM_ON[i], self.Na[0])
                # PHI0 should be PHI0[i] not PHI0[2*i+1]
                if self.TASK == 'dual':
                    random = torch.randn((self.Na[0],),  device=self.device)
                    stimulus = Stimuli(self.TASK, size)(self.I0[i], self.SIGMA0[i], random)
                    # stimulus = Stimuli(self.TASK, size)(self.I0[i], self.SIGMA0[i], self.PHI0[2*i+1])
                else:
                    stimulus = Stimuli(self.TASK, size)(self.I0[i], self.SIGMA0[i], self.PHI0[i])
                
                ff_input[:, self.N_STIM_ON[i]:self.N_STIM_OFF[i], self.slices[0]] += stimulus
        
            del stimulus
        
        return ff_input * torch.sqrt(self.Ka[0]) * self.M0
