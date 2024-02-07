import os
import numpy as np

import torch
from torch import nn
from torch.distributions import Normal, MultivariateNormal, Exponential

from yaml import safe_load
from time import perf_counter

from src.connectivity import Connectivity
from src.activation import Activation
from src.stimuli import Stimuli
from src.plasticity import STP_Model
from src.utils import set_seed, clear_cache

import warnings
warnings.filterwarnings("ignore")

class Network(nn.Module):
    def __init__(self, conf_file, sim_name, repo_root, **kwargs):
        super().__init__()

        # load parameters
        self.loadConfig(conf_file, sim_name, repo_root, **kwargs)

        set_seed(self.SEED)
        
        # compute constants (time steps, time constants, ...)
        self.initConst()
        
        # rescale weights
        self.scaleParam()
        
        # adds stp
        if self.IF_STP:
            self.stp = STP_Model(self.N_NEURON, self.csumNa, self.DT, self.FLOAT, self.device)
        
        # initialize network connectivity
        self.initWeights()
        
        # self.U = nn.Parameter(torch.randn((self.N_NEURON, int(self.RANK)), device=self.device, dtype=self.FLOAT))
        # self.mask = torch.ones((self.N_NEURON, self.N_NEURON), device=self.device, dtype=self.FLOAT)
        
        # for i_pop in range(self.N_POP):
        #     for j_pop in range(self.N_POP):
        #         if i_pop!=0 and j_pop!=0:
        #             self.mask[self.csumNa[i_pop] : self.csumNa[i_pop + 1],
        #                       self.csumNa[j_pop] : self.csumNa[j_pop + 1]] = 0
        
    def initWeights(self):
        # set seed for connectivity
        
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
                
                self.Wab[self.csumNa[i_pop] : self.csumNa[i_pop + 1]
                         , self.csumNa[j_pop] : self.csumNa[j_pop + 1]] = self.Jab[i_pop][j_pop] * weights
                
        del weights
        # resets the seed
        set_seed(0)
    
    def update_dynamics(self, rates, ff_input, rec_input):
        '''Updates the dynamics of the model at each timestep'''
        
        # update stp variables
        A_u_x = 1.0
        if self.IF_STP:
            A_u_x = self.stp.markram_stp(rates)
        
        # lr = self.mask * (1.0 + self.U @ self.U.T / torch.sqrt(self.Ka[0]))
        # hidden = (A_u_x * rates) @ (self.Wab.T * lr)
        
        hidden = (A_u_x * rates) @ self.Wab.T
        
        # update reccurent input
        if self.SYN_DYN:
            rec_input = self.EXP_DT_TAU_SYN * rec_input + self.DT_TAU_SYN * hidden
        else:
            rec_input = hidden
            
        # compute net input
        net_input = ff_input + rec_input
        
        # update rates
        if self.RATE_DYN:
            rates = self.EXP_DT_TAU * rates + self.DT_TAU * Activation()(net_input, func_name=self.TF_TYPE, thresh=self.THRESH[0])
        else:
            rates = Activation()(net_input, func_name=self.TF_TYPE, thresh=self.THRESH[0])
        
        return rates, rec_input
    
    def forward(self, ff_input=None, REC_LAST_ONLY=1):
        '''
        main method of the class
        :param ff_input: float (N_BATCH, N_TIME, N_NEURONS)
        :param REC_LAST_ONLY: bool
        '''
        
        if self.VERBOSE:
            start = perf_counter()
        
        result = []
        
        # initialization
        rates, rec_input, self.ff_input = self.initialization(ff_input)
        
        # moving average
        mv_rates = 0
        
        for step in range(self.N_STEPS):
            rates, rec_input = self.update_dynamics(rates, self.ff_input[:, step], rec_input)
            mv_rates += rates
            
            if step == self.N_STEADY-self.N_WINDOW-1:
                mv_rates *= 0.0
            
            if step >= self.N_STEADY:
                if step % self.N_WINDOW == 0:
                    if self.VERBOSE:
                        self.print_activity(step, rates)
                    
                    if REC_LAST_ONLY==0:
                        result.append(mv_rates[..., :self.Na[0]].cpu().detach().numpy() / self.N_WINDOW)
                    
                    # reset mv avg
                    mv_rates = 0
        
        if REC_LAST_ONLY:
            result = rates[..., :self.Na[0]]
        else:
            result = np.array(result)
        
        # if self.VERBOSE:
        # print('Saving rates to:', self.DATA_PATH + self.FILE_NAME + '.npy')
        # np.save(self.DATA_PATH + self.FILE_NAME + '.npy', result)
        
        # clear_cache()
        
        if self.VERBOSE:
            end = perf_counter()
            print("Elapsed (with compilation) = {}s".format((end - start)))
        
        return result
    
    def print_activity(self, step, rates):
        
        times = np.round((step - self.N_STEADY) / self.N_STEPS * self.DURATION, 2)
        
        activity = []
        for i in range(self.N_POP):
            activity.append(np.round(torch.mean(rates[:, self.csumNa[i]:self.csumNa[i+1]]).cpu().detach().numpy(), 2))
        
        print("times (s)", np.round(times, 2), "rates (Hz)", activity)
    
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
            self.EXP_DT_TAU[self.csumNa[i_pop] : self.csumNa[i_pop + 1]] = torch.exp(-self.DT / self.TAU[i_pop])
            self.DT_TAU[self.csumNa[i_pop] : self.csumNa[i_pop + 1]] = self.DT / self.TAU[i_pop]

        # if self.VERBOSE:
        #     print("DT", self.DT, "TAU", self.TAU)
        
        self.TAU_SYN = torch.tensor(self.TAU_SYN, dtype=self.FLOAT, device=self.device)
        self.EXP_DT_TAU_SYN = torch.ones(self.N_NEURON, dtype=self.FLOAT, device=self.device)
        self.DT_TAU_SYN = torch.ones(self.N_NEURON, dtype=self.FLOAT, device=self.device)
        
        for i_pop in range(self.N_POP):
            self.EXP_DT_TAU_SYN[self.csumNa[i_pop] : self.csumNa[i_pop + 1]] = torch.exp(-self.DT / self.TAU_SYN[i_pop])
            self.DT_TAU_SYN[self.csumNa[i_pop] : self.csumNa[i_pop + 1]] = self.DT / self.TAU_SYN[i_pop]
        
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
            
            multivariate_normal = MultivariateNormal(mean_, cov_)
            del mean_, cov_
            
            self.PHI0 = multivariate_normal.sample((self.Na[0],)).T
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
        
        ff_input = torch.zeros(self.N_BATCH, self.N_STEPS, self.N_NEURON, dtype=self.FLOAT, device=self.device)
        noise = torch.randn((self.N_BATCH, self.N_STEPS, self.N_NEURON), dtype=self.FLOAT, device=self.device)

        for i_pop in range(self.N_POP):
            noise[self.csumNa[i_pop] : self.csumNa[i_pop + 1]] = noise[self.csumNa[i_pop] : self.csumNa[i_pop + 1]] * self.VAR_FF[i_pop]
        
        for i in range(self.N_POP):
            ff_input[:, :self.N_STIM_ON[0], self.csumNa[i]:self.csumNa[i+1]] = self.Ja0[i] * (1-self.BUMP_SWITCH[i])
        
        for i in range(self.N_POP):
            ff_input[:, self.N_STIM_ON[0]:, self.csumNa[i]:self.csumNa[i+1]] = self.Ja0[i]
        
        for i in range(len(self.N_STIM_ON)):
            ff_input[:, self.N_STIM_ON[i]:self.N_STIM_OFF[i],
                     self.csumNa[0]:self.csumNa[1]] = self.Ja0[0] + Stimuli(self.TASK, (self.N_BATCH,
                                                                                        self.N_STIM_OFF[i]-self.N_STIM_ON[i],
                                                                                        self.Na[0]),
                                                                            )(self.I0[i], self.SIGMA0[i], self.PHI0[i])
        
        return (ff_input + noise) * torch.sqrt(self.Ka[0]) * self.M0

    
