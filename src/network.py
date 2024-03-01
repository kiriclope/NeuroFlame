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
from src.lr_utils import get_theta

import warnings
warnings.filterwarnings("ignore")

class Network(nn.Module):

    def __init__(self, conf_file, sim_name, repo_root, **kwargs):
        '''
        Class: Network
        Creates a recurrent network of rate units with customizable connectivity and dynamics.
        The network can be trained with standard torch optimization technics.
        Parameters:
               conf_file: str, name of a .yml file contaning the model's parameters.
               sim_name: str, name of a .txt file to save the model's outputs.
               repo_root: str, root path of the NeuroTorch repository.
               **kwargs: **dict, any parameter in conf_file can be passed here
                                 and will then be overwritten.
        Returns:
               rates: tensorfloat of size (N_BATCH, N_SEQ_LEN or 1, N_NEURON).
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

        # Initialize low rank connectivity
        if self.LR_TRAIN:
            self.initLR()

        # Reset the seed
        set_seed(0)
    
    def initLR(self):
        # Low rank vector
        self.U = nn.Parameter(torch.randn((self.N_NEURON, int(self.RANK)),
                                          device=self.device, dtype=self.FLOAT))
        # self.V = nn.Parameter(torch.randn((self.N_NEURON, int(self.RANK)),
        # device=self.device, dtype=self.FLOAT))

        # Mask to train excitatory neurons only
        self.mask = torch.zeros((self.N_NEURON, self.N_NEURON),
                                device=self.device, dtype=self.FLOAT)

        self.mask[self.slices[0], self.slices[0]] = 1.0

        # Linear readout for supervised learning
        self.linear = nn.Linear(self.Na[0], 1, device=self.device, dtype=self.FLOAT, bias=False)

        self.lr_kappa = nn.Parameter(5 * torch.rand(1))

        # Window where to evaluate loss
        self.lr_eval_win = int(self.LR_EVAL_WIN / self.DT / self.N_WINDOW)

    def initWeights(self):
        '''
        Initializes the connectivity matrix self.Wab.
        Loops over (pre, post) blocks to create the full matrix.
        Relies on class Connectivity from connetivity.py
        '''

        # in pytorch, Wij is i to j.
        self.Wab_T = torch.zeros((self.N_NEURON, self.N_NEURON), dtype=self.FLOAT, device=self.device)

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

                # self.Wab_T[self.slices[i_pop], self.slices[j_pop]] = weights
                self.Wab_T[self.slices[i_pop], self.slices[j_pop]] = self.Jab[i_pop][j_pop] * weights

        del weights

        # U, S, V = torch.svd(self.Wab_T)

        # Use only the first singular value and vectors for rank 1 approximation
        # self.Wab_T = S[0] * U[:, 0].unsqueeze(1) @ V[:, 0].unsqueeze(0)

        self.Wab_T = self.Wab_T.T

        # if self.CON_TYPE=='sparse':
        #     self.Wab_T = self.Wab_T.to_sparse()

    def update_dynamics(self, rates, ff_input, rec_input):
        '''Updates the dynamics of the model at each timestep'''

        # update stp variables
        A_u_x = 1.0
        if self.IF_STP:
            A_u_x = self.stp(rates[:, :self.Na[0]])
            stp_ee = (rates[:, :self.Na[0]] * A_u_x) @ self.W_stp.T

        # if self.LR_TRAIN:
        #     lr = (1.0 + self.mask * self.KAPPA[0][0] * (self.U @ self.U.T) / torch.sqrt(self.Ka[0]))
        #     hidden = rates @ (self.Wab_T * lr.T)
        # else:
        #     hidden = rates @ self.Wab_T

        # if self.CON_TYPE=='sparse':
        #     hidden = torch.sparse.mm(rates, self.Wab_T)
        # else:

        # update hidden state
        hidden = rates @ self.Wab_T

        if self.IF_STP:
            hidden[:, :self.Na[0]] = hidden[:, :self.Na[0]] + stp_ee

        # update reccurent input
        if self.SYN_DYN:
            rec_input = self.EXP_DT_TAU_SYN * rec_input + self.DT_TAU_SYN * hidden
        else:
            rec_input = hidden

        # compute net input
        net_input = ff_input + rec_input
        non_linear = Activation()(net_input,
                                  func_name=self.TF_TYPE,
                                  thresh=self.THRESH[0])

        # update rates
        if self.RATE_DYN:
            rates = self.EXP_DT_TAU * rates + self.DT_TAU * non_linear
        else:
            rates = non_linear

        del hidden, net_input, non_linear

        return rates, rec_input

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

        # Here, ff_input is (N_BATCH, N_NEURON) and is updated at each timestep.
        # Otherwise, ff_input is (N_BATCH, N_STEP, N_NEURON).
        # Live FF update is recommended when dealing with large batch size.

        if self.LIVE_FF_UPDATE:
            self.stim_mask = torch.ones((self.N_BATCH, self.Na[0]),
                                        dtype=self.FLOAT, device=self.device)

            self.stim_mask[self.N_BATCH//2:] = -1

            ff_input = torch.zeros((self.N_BATCH, self.N_NEURON),
                                   dtype=self.FLOAT, device=self.device)

            ff_input, noise = self.live_ff_input(0, ff_input)

        # Initialization (if  ff_input is None, ff_input is generated)
        rates, ff_input, rec_input = self.initialization(ff_input)

        # Add STP
        if self.IF_STP:
            self.stp = Plasticity(self.USE, self.TAU_FAC, self.TAU_REC, self.DT,
                                  (self.N_BATCH, self.Na[0]),
                                  FLOAT=self.FLOAT, device=self.device)

            self.W_stp = self.Wab_T[self.slices[0],self.slices[0]]
            self.Wab_T[self.slices[0], self.slices[0]] = 0

        # Moving average of the rates
        mv_rates = 0

        self.ff_input = []
        output = []
        # Temporal loop
        for step in range(self.N_STEPS):

            # update dynamics
            if self.LIVE_FF_UPDATE:
                ff_input, noise = self.live_ff_input(step, ff_input)
                rates, rec_input = self.update_dynamics(rates, ff_input + noise, rec_input)
            else:
                rates, rec_input = self.update_dynamics(rates, ff_input[:, step], rec_input)

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

                    if not REC_LAST_ONLY:
                        output.append(mv_rates[..., self.slices[0]] / self.N_WINDOW)
                        # output.append(mv_rates / self.N_WINDOW)

                    # Reset moving average
                    mv_rates = 0

        if not REC_LAST_ONLY:
            # Stack output list to 1st dim so that output is (N_BATCH, N_STEPS, N_NEURON)
            output = torch.stack(output, dim=1)
            
        # Add Linear readout (N_BATCH, N_EVAL_WIN, 1) on last few steps
        if self.LR_TRAIN:
            y_pred = self.linear(output[:, -self.lr_eval_win:, ...])
            del output
            clear_cache()
            return y_pred.squeeze(-1)

        if REC_LAST_ONLY:
            output = rates[..., self.slices[0]]

        # self.ff_input = ff_input
        del rates, ff_input, rec_input
        
        if self.VERBOSE:
            end = perf_counter()
            print("Elapsed (with compilation) = {}s".format((end - start)))

        clear_cache()

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

        rec_input = torch.randn((self.N_BATCH, self.N_NEURON), dtype=self.FLOAT, device=self.device)

        if self.LIVE_FF_UPDATE:
            rates = Activation()(ff_input + rec_input, func_name=self.TF_TYPE, thresh=self.THRESH[0])
        else:
            rates = Activation()(ff_input[:, 0] + rec_input, func_name=self.TF_TYPE, thresh=self.THRESH[0])

        return rates, ff_input, rec_input

    def initConst(self):
        self.N_STEADY = int(self.T_STEADY / self.DT)
        self.N_WINDOW = int(self.T_WINDOW / self.DT)
        self.N_STEPS = int(self.DURATION / self.DT) + self.N_STEADY + self.N_WINDOW

        self.N_STIM_ON = np.array([int(i / self.DT) + self.N_STEADY for i in self.T_STIM_ON])
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

        self.slices = []
        for i_pop in range(self.N_POP):
            self.slices.append(slice(self.csumNa[i_pop], self.csumNa[i_pop + 1]))

        if self.VERBOSE:
            print("Na", self.Na, "Ka", self.Ka, "csumNa", self.csumNa)

        self.TAU = torch.tensor(self.TAU, dtype=self.FLOAT, device=self.device)
        self.EXP_DT_TAU = torch.ones(self.N_NEURON, dtype=self.FLOAT, device=self.device)
        self.DT_TAU = torch.ones(self.N_NEURON, dtype=self.FLOAT, device=self.device)

        for i_pop in range(self.N_POP):
            self.EXP_DT_TAU[self.slices[i_pop]] = torch.exp(-self.DT / self.TAU[i_pop])
            # self.EXP_DT_TAU[self.csumNa[i_pop] : self.csumNa[i_pop + 1]] = (1.0 - self.DT / self.TAU[i_pop])
            self.DT_TAU[self.slices[i_pop]] = self.DT / self.TAU[i_pop]

        # if self.VERBOSE:
        #     print("DT", self.DT, "TAU", self.TAU)

        self.TAU_SYN = torch.tensor(self.TAU_SYN, dtype=self.FLOAT, device=self.device)
        self.EXP_DT_TAU_SYN = torch.ones(self.N_NEURON, dtype=self.FLOAT, device=self.device)
        self.DT_TAU_SYN = torch.ones(self.N_NEURON, dtype=self.FLOAT, device=self.device)

        for i_pop in range(self.N_POP):
            self.EXP_DT_TAU_SYN[self.slices[i_pop]] = torch.exp(-self.DT / self.TAU_SYN[i_pop])
            # self.EXP_DT_TAU_SYN[self.csumNa[i_pop] : self.csumNa[i_pop + 1]] = (1.0-self.DT / self.TAU_SYN[i_pop])
            self.DT_TAU_SYN[self.slices[i_pop]] = self.DT / self.TAU_SYN[i_pop]

        # if self.VERBOSE:
        #     print("EXP_DT_TAU", self.EXP_DT_TAU, "DT_TAU", self.DT_TAU)

        self.PROBA_TYPE = np.array(self.PROBA_TYPE).reshape(self.N_POP, self.N_POP)
        self.SIGMA = torch.tensor(self.SIGMA, dtype=self.FLOAT, device=self.device).view(self.N_POP, self.N_POP)
        self.KAPPA = torch.tensor(self.KAPPA, dtype=self.FLOAT, device=self.device).view(self.N_POP, self.N_POP)
        self.PHASE = torch.tensor(self.PHASE * torch.pi / 180.0, dtype=self.FLOAT, device=self.device)

        self.VAR_FF = torch.sqrt(torch.tensor(self.VAR_FF, dtype=self.FLOAT, device=self.device)) / torch.sqrt(self.Ka[0]) * self.M0

        if self.PROBA_TYPE[0][0] == 'lr':
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

        if self.LIVE_FF_UPDATE:
            self.Ja0.mul_(self.M0 * torch.sqrt(self.Ka[0]))
        # now inputs are scaled in init_ff_input

    def live_ff_input(self, step, ff_input):

        noise = 0
        if self.VAR_FF[0]>0:
            noise = torch.randn((self.N_BATCH, self.N_NEURON), dtype=self.FLOAT, device=self.device)
            for i_pop in range(self.N_POP):
                noise[:, self.slices[i_pop]].mul_(self.VAR_FF[i_pop])

        if step==0:
            for i_pop in range(self.N_POP):
                if self.BUMP_SWITCH[i_pop]:
                    ff_input[:, self.slices[i_pop]] = self.Ja0[i_pop] / torch.sqrt(self.Ka[0])
                else:
                    ff_input[:, self.slices[i_pop]] = self.Ja0[i_pop]
        
        if step==self.N_STIM_ON[0]:
            for i_pop in range(self.N_POP):
                ff_input[:, self.slices[i_pop]] = self.Ja0[i_pop]
        
        stimulus = torch.zeros((1, 1), device=self.device)
        
        if self.TASK != 'None':
            if step in self.N_STIM_ON:
                i = np.where(self.N_STIM_ON == step)[0][0]
                
                size = (self.N_BATCH, self.Na[0])
                
                if 'dual' in self.TASK:
                    if 'rand' in self.TASK:
                        theta = get_theta(self.PHI0[0], self.PHI0[2])
                        stimulus = Stimuli('odr', size, device=self.device)(self.I0[i], self.SIGMA0[i], self.PHI0[i],
                                                                            rnd_phase=1, theta_list=theta)
                        del theta
                        
                    else:
                        stimulus = Stimuli(self.TASK, size)(self.I0[i], self.SIGMA0[i], self.PHI0[2*i+1])
                        if i == 0:
                            stimulus = self.stim_mask * stimulus
                else:
                    stimulus = Stimuli(self.TASK, size)(self.I0[i], self.SIGMA0[i], self.PHI0[i])
                
                ff_input[:, self.slices[0]] = self.Ja0[0] + stimulus * torch.sqrt(self.Ka[0]) * self.M0
                del stimulus
                
            if step in self.N_STIM_OFF:
                ff_input[:, self.slices[0]] = self.Ja0[0]
        
        return ff_input, noise

    def init_ff_input(self):
        """
        Creates ff input for all timestep
        Inputs can be noisy or not and depend on the task.
        """

        ff_input = torch.randn((self.N_BATCH, self.N_STEPS, self.N_NEURON), dtype=self.FLOAT, device=self.device)

        for i_pop in range(self.N_POP):
            ff_input[..., self.slices[i_pop]].mul_(self.VAR_FF[i_pop] / torch.sqrt(self.Ka[0]))

        for i_pop in range(self.N_POP):
            if self.BUMP_SWITCH[i_pop]:
                ff_input[:, :self.N_STIM_ON[0], self.slices[i_pop]].add_(self.Ja0[i_pop] / torch.sqrt(self.Ka[0]))
            else:
                ff_input[:, :self.N_STIM_ON[0], self.slices[i_pop]].add_(self.Ja0[i_pop])

        for i_pop in range(self.N_POP):
            ff_input[:, self.N_STIM_ON[0]:, self.slices[i_pop]].add_(self.Ja0[i_pop])

        if self.TASK != 'None':
            for i ,_ in enumerate(self.N_STIM_ON):
                size = (self.N_BATCH, self.Na[0])
                if 'dual' in self.TASK:
                    if 'rand' in self.TASK:
                        theta = get_theta(self.PHI0[0], self.PHI0[2])
                        stimulus = Stimuli('odr', size, device=self.device)(self.I0[i], self.SIGMA0[i], self.PHI0[i], rnd_phase=1, theta_list=theta)
                        del theta

                        stimulus = stimulus.unsqueeze(1).expand((stimulus.shape[0],
                                                                 1,
                                                                 stimulus.shape[-1]))
                    else:
                        stimulus = Stimuli(self.TASK, size)(self.I0[i], self.SIGMA0[i], self.PHI0[2*i+1])
                        
                else:
                    stimulus = Stimuli(self.TASK, size, device=self.device)(self.I0[i], self.SIGMA0[i], self.PHI0[i])
                
                # if 'rand' in self.TASK:
                #     # theta = get_theta(self.PHI0[0], self.PHI0[2])
                #     # idx = theta.argsort()
                #     # print(idx)
                    
                #     ff_input[:, self.N_STIM_ON[i]:self.N_STIM_OFF[i], idx].add_(stimulus)
                # else:
                ff_input[:, self.N_STIM_ON[i]:self.N_STIM_OFF[i], self.slices[0]].add_(stimulus)
                
                del stimulus
        
        return ff_input * torch.sqrt(self.Ka[0]) * self.M0
