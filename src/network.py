import os
import numpy as np

import torch
from torch import nn
from torch.distributions import Normal, MultivariateNormal, Exponential

from yaml import safe_load
from time import perf_counter

from src.utils import set_seed, get_theta, clear_cache
from src.activation import Activation
from src.plasticity import STP_Model

import warnings
warnings.filterwarnings("ignore")


class Network(nn.Module):
    def __init__(self, conf_file, sim_name, repo_root, **kwargs):
        super().__init__()

        # load parameters
        self.loadConfig(conf_file, sim_name, repo_root, **kwargs)

        # set csts
        self.initConst()

        # scale parameters
        self.scaleParam()

        if self.IF_STP:
            self.stp = STP_Model(self.N_NEURON, self.csumNa, self.DT, self.FLOAT, self.device)

        # initialize network
        self.init_network()

        # self.exp_dist = Exponential(torch.tensor([self.I1[0]], dtype=self.FLOAT, device=self.device))
        
    def init_network(self):
        # set seed for Cij
        set_seed(self.SEED)

        # in pytorch, Wij is i to j.
        self.Wab = nn.Linear(self.N_NEURON, self.N_NEURON, bias=False, dtype=self.FLOAT, device=self.device)
        # if self.IF_NMDA:
        #     self.W_NMDA = nn.Linear(self.N_NEURON, self.Na[0], bias=False, dtype=self.FLOAT, device=self.device)
        # print(self.W_NMDA)

        for i_pop in range(self.N_POP):
            for j_pop in range(self.N_POP):

                self.Wab.weight.data[self.csumNa[i_pop] : self.csumNa[i_pop + 1],
                                    self.csumNa[j_pop] : self.csumNa[j_pop + 1]] = self.initWeights(i_pop, j_pop)

                # if self.W_NMDA and i_pop==0:
                #     self.W_NMDA.weight.data[self.csumNa[i_pop] : self.csumNa[i_pop + 1],
                #                             self.csumNa[j_pop] : self.csumNa[j_pop + 1]] = self.initWeights(i_pop, j_pop)
        
        # resets the seed
        set_seed(0)

    def update_rec_input(self, rates, rec_input, rec_NMDA):
        '''Dynamics of the recurrent inputs'''

        A_u_x = 1.0
        if self.IF_STP:
            self.stp.markram_stp(rates)
            A_u_x = self.stp.A_u_x_stp

        if self.SYN_DYN:
            rec_input = self.EXP_DT_TAU_SYN * rec_input + self.DT_TAU_SYN * (self.Wab(A_u_x * rates))
            # if self.IF_NMDA:
            #     rec_NMDA = self.EXP_DT_TAU_NMDA * rec_NMDA + self.DT_TAU_NMDA * (self.W_NMDA(A_u_x * rates))
        else:
            rec_input = self.Wab(A_u_x * rates)

        return rec_input, rec_NMDA

    def update_net_input(self, rec_input, rec_NMDA, ff_input):
        '''Updating the net input into the neurons'''

        noise = 0
        if self.VAR_FF[0]>0:
            noise = torch.randn((self.N_BATCH, self.N_NEURON,), dtype=self.FLOAT, device=self.DEVICE) * self.VAR_FF[0]

        net_input = ff_input + rec_input + noise

        # if self.IF_NMDA:
        #     net_input += rec_NMDA

        return net_input

    def update_rates(self, rates, net_input):
        '''Dynamics of the rates'''
        # using array slices is faster than indices
        if self.RATE_DYN:
            rates = self.EXP_DT_TAU * rates + self.DT_TAU * Activation()(net_input, func_name=self.TF_TYPE, thresh=self.THRESH[0])
        else:
            rates = Activation()(net_input, func_name=self.TF_TYPE, thresh=self.THRESH[0])

        return rates

    def forward(self, rates, rec_input, rec_NMDA, ff_input):
        '''This is the main function of the class'''
        rec_input, rec_NMDA = self.update_rec_input(rates, rec_input, rec_NMDA)
        net_input = self.update_net_input(rec_input, rec_NMDA, ff_input)
        rates = self.update_rates(rates, net_input)

        return rates, rec_input, rec_NMDA

    def batch_loader(self, ini_list, phi_list, Ja0_list, IF_DIST=0):
        self.N_BATCH = len(ini_list) * len(phi_list) * len(Ja0_list)
        
        if IF_DIST:
            if len(phi_list)!=1:
                self.PHI1 = torch.tensor(phi_list, dtype=self.FLOAT, device=self.DEVICE)
                self.PHI1 = self.PHI1.unsqueeze(0).unsqueeze(2).expand(len(ini_list), -1, len(Ja0_list))
                # print('PHI1', self.PHI1.shape)
            
                self.PHI1 = self.PHI1.reshape(-1, 1)
        else:
            if len(phi_list)!=1:
                self.PHI0 = torch.tensor(phi_list, dtype=self.FLOAT, device=self.DEVICE)
                self.PHI0 = self.PHI0.unsqueeze(0).unsqueeze(2).expand(len(ini_list), -1, len(Ja0_list))
                # print('PHI0', self.PHI0.shape)
                
                self.PHI0 = self.PHI0.reshape(-1, 1)
                
        if len(Ja0_list)!=1:
            self.Je0 = torch.tensor(Ja0_list, dtype=self.FLOAT, device=self.DEVICE) * torch.sqrt(self.Ka[0]) * self.M0
            self.Je0 = self.Je0.unsqueeze(0).unsqueeze(1).expand(len(ini_list), len(phi_list), -1)
            # print('Je0', self.Je0.shape)
            self.Je0 = self.Je0.reshape(-1, 1)
        else:
            self.Je0 = self.Ja0[0]
    
    def run(self, ini_list=[1], phi_list=[1], Ja0_list=[1], IF_DIST=0):
        if self.VERBOSE:
            start = perf_counter()

        result = []

        with torch.no_grad():
            self.batch_loader(ini_list, phi_list, Ja0_list, IF_DIST)

            rates, rec_input, rec_NMDA, ff_input = self.initialization()
            mv_rates = rates

            for step in range(self.N_STEPS):
                ff_input = self.update_ff_input(step, ff_input)
                rates, rec_input, rec_NMDA = self.forward(rates, rec_input, rec_NMDA, ff_input)
                
                mv_rates += rates

                if step >= self.N_STEADY:
                    if step % self.N_WINDOW == 0:
                        if self.VERBOSE:
                            self.print_activity(step, rates)

                        if self.REC_LAST_ONLY==0:
                            result.append(mv_rates.cpu().detach().numpy() / self.N_WINDOW)
                            mv_rates = 0

        if self.REC_LAST_ONLY:
            result.append(rates.cpu().detach().numpy())

        result = np.array(result)

        if self.VERBOSE:
            print('Saving rates to:', self.DATA_PATH + self.FILE_NAME + '.npy')
        # np.save(self.DATA_PATH + self.FILE_NAME + '.npy', result)

        clear_cache()

        if self.VERBOSE:
            end = perf_counter()
            print("Elapsed (with compilation) = {}s".format((end - start)))

        return result

    def print_activity(self, step, rates):

        times = np.round((step - self.N_STEADY) / self.N_STEPS * self.DURATION, 2)

        activity = []

        activity.append(np.round(torch.mean(rates[:self.csumNa[1]]).cpu().detach().numpy(), 2))

        if self.N_POP > 1:
            activity.append(np.round(torch.mean(rates[self.csumNa[1]:self.csumNa[2]]).cpu().detach().numpy(), 2))

        if self.N_POP > 2:
            activity.append(np.round(torch.mean(rates[self.csumNa[2]:]).cpu().detach().numpy(), 2))

        print(
            "times (s)",
            np.round(times, 2),
            "rates (Hz)",
            activity,
        )

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

    def initialization(self):
        rates = torch.zeros(self.N_BATCH, self.N_NEURON, dtype=self.FLOAT, device=self.device)
        rec_input = torch.zeros(self.N_BATCH, self.N_NEURON, dtype=self.FLOAT, device=self.device)
        # rec_NMDA = torch.zeros(self.N_BATCH, self.N_NEURON, dtype=self.FLOAT, device=self.device)
        rec_NMDA = 0
        ff_input = torch.zeros(self.N_BATCH, self.N_NEURON, dtype=self.FLOAT, device=self.device)

        return rates, rec_input, rec_NMDA, ff_input

    def initWeights(self, i_pop, j_pop):

        Na = self.Na[i_pop]
        Nb = self.Na[j_pop]
        Kb = self.Ka[j_pop]

        Pij = torch.tensor(1.0, dtype=self.FLOAT, device=self.device)

        if 'lr' in self.STRUCTURE[i_pop, j_pop]:

            mean = torch.tensor([0.0, 0.0], dtype=self.FLOAT, device=self.device)

            # Define the covariance matrix
            covariance = torch.tensor([[1.0, self.LR_COV],
                                       [self.LR_COV, 1.0],], dtype=self.FLOAT, device=self.device)


            multivariate_normal = MultivariateNormal(mean, covariance)
            self.ksi = multivariate_normal.sample((Nb,)).T

            # while torch.abs(self.ksi[0] @ self.ksi[1]) > .10:
            #     multivariate_normal = MultivariateNormal(mean, covariance)
            #     self.ksi = multivariate_normal.sample((Nb,)).T

            if self.VERBOSE:
                print('ksi', self.ksi.shape)
                print('ksi . ksi1', self.ksi[0] @ self.ksi[1])

            Pij = 1.0 + self.KAPPA[i_pop, j_pop] * (torch.outer(self.ksi[0], self.ksi[0])
                                                    + torch.outer(self.ksi[1], self.ksi[1])) / torch.sqrt(self.Ka[j_pop])
            # Pij[Pij>1] = 1
            # Pij[Pij<0] = 0

            if self.VERBOSE:
                print('Pij', Pij.shape)

        if 'cos' in self.STRUCTURE[i_pop, j_pop]:
            
            theta_i, theta_j = torch.meshgrid(self.theta_list[i_pop], self.theta_list[j_pop], indexing="ij")
            theta_diff = theta_i - theta_j
            
            if 'spec' in self.STRUCTURE[i_pop, j_pop]:
                self.KAPPA[i_pop, j_pop] = self.KAPPA[i_pop, j_pop] / torch.sqrt(Kb)
            
            Pij = 1.0 + 2.0 * self.KAPPA[i_pop, j_pop] * torch.cos(theta_diff - self.PHASE)
            
            del theta_i
            del theta_j
            del theta_diff

        if 'sparse' in self.CONNECTIVITY:
            if self.VERBOSE:
                print('Sparse random connectivity ')

            Cij = self.Jab[i_pop, j_pop] * (torch.rand(Na, Nb, device=self.device) < Kb / float(Nb) * Pij)
            del Pij

        if 'all2all' in self.CONNECTIVITY:
            if self.VERBOSE:
                print('All to all connectivity ')

            Cij = self.Jab[i_pop, j_pop] * Pij / float(Nb)
            del Pij

            if self.SIGMA[i_pop, j_pop] > 0.0:
                if self.VERBOSE:
                    print('with heterogeneity, SIGMA', self.SIGMA[i_pop, j_pop])

                Hij = self.SIGMA[i_pop, j_pop] * torch.randn((Na, Nb), dtype=self.FLOAT, device=self.device)
                Cij = Cij + Hij / torch.sqrt(torch.tensor(Nb, device=self.device, dtype=self.FLOAT))
                del Hij

        if self.VERBOSE:
            if "cos" in self.STRUCTURE[i_pop, j_pop]:
                if "spec" in self.STRUCTURE[i_pop, j_pop]:
                    print('with weak cosine structure, KAPPA %.2f' % self.KAPPA[i_pop, j_pop].cpu().detach().numpy())
                else:
                    print('with strong cosine structure, KAPPA', self.KAPPA[i_pop, j_pop])
            elif "lr" in self.STRUCTURE[i_pop, j_pop]:
                print('with weak low rank structure, KAPPA %.2f' % self.KAPPA[i_pop, j_pop].cpu().detach().numpy())

        return Cij

    def initConst(self):
        self.N_STEADY = int(self.T_STEADY / self.DT)
        self.N_STEPS = int(self.DURATION / self.DT) + self.N_STEADY
        self.N_WINDOW = int(self.T_WINDOW / self.DT)

        self.N_STIM_ON = int(self.T_STIM[0] / self.DT) + self.N_STEADY
        self.N_STIM_OFF = int(self.T_STIM[1] / self.DT) + self.N_STEADY

        self.N_DIST_ON = int(self.T_DIST[0] / self.DT) + self.N_STEADY
        self.N_DIST_OFF = int(self.T_DIST[1] / self.DT) + self.N_STEADY

        self.Na = []
        self.Ka = []

        if 'all2all' in self.CONNECTIVITY:
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

        self.TAU_NMDA = torch.tensor(self.TAU_NMDA, dtype=self.FLOAT, device=self.device)
        self.EXP_DT_TAU_NMDA = torch.ones(self.N_NEURON, dtype=self.FLOAT, device=self.device)
        self.DT_TAU_NMDA = torch.ones(self.N_NEURON, dtype=self.FLOAT, device=self.device)

        for i_pop in range(self.N_POP):
            self.EXP_DT_TAU_NMDA[self.csumNa[i_pop] : self.csumNa[i_pop + 1]] = torch.exp(-self.DT / self.TAU_NMDA[i_pop])
            self.DT_TAU_NMDA[self.csumNa[i_pop] : self.csumNa[i_pop + 1]] = self.DT / self.TAU_NMDA[i_pop]

        # if self.VERBOSE:
        #     print("EXP_DT_TAU", self.EXP_DT_TAU, "DT_TAU", self.DT_TAU)

        self.STRUCTURE = np.array(self.STRUCTURE).reshape(self.N_POP, self.N_POP)
        self.SIGMA = torch.tensor(self.SIGMA, dtype=self.FLOAT, device=self.device).view(self.N_POP, self.N_POP)
        self.KAPPA = torch.tensor(self.KAPPA, dtype=self.FLOAT, device=self.device).view(self.N_POP, self.N_POP)
        # self.PHASE = torch.tensor(self.PHASE * torch.pi / 180.0, dtype=self.FLOAT, device=self.device)
        self.PHASE = torch.pi
        
        self.theta_list = []
        for _ in range(self.N_POP):
            self.theta_list.append(torch.linspace(0, 2.0 * torch.pi, self.Na[i_pop] + 1, dtype=self.FLOAT, device=self.device)[:-1])
        
        # if self.VERBOSE:
        #     print(self.STRUCTURE)
        #     print(self.SIGMA)
        #     print(self.KAPPA)
        #     print(self.PHASE)

    def scaleParam(self):

        # scaling recurrent weights Jab
        if self.VERBOSE:
            print("Jab", self.Jab)
        
        self.Jab = torch.tensor(self.Jab, dtype=self.FLOAT, device=self.device).reshape(self.N_POP, self.N_POP) * self.GAIN

        for i_pop in range(self.N_POP):
            self.Jab[:, i_pop] = self.Jab[:, i_pop] / torch.sqrt(self.Ka[i_pop])

        # if self.VERBOSE:
        #     print("scaled Jab", self.Jab)

        # scaling FF weights
        if self.VERBOSE:
            print("Ja0", self.Ja0)

        self.Ja0 = torch.tensor(self.Ja0, dtype=self.FLOAT, device=self.device)
        self.Ja0 = self.Ja0 * torch.sqrt(self.Ka[0]) * self.M0

        # if self.VERBOSE:
        #     print("scaled Ja0", self.Ja0)

        self.VAR_FF = torch.sqrt(torch.tensor(self.VAR_FF, dtype=self.FLOAT, device=self.device) * self.DT)

    def update_ff_input(self, step, ff_input):
        """Perturb the inputs based on the simulus parameters."""

        if step == 0:
            for i in range(self.N_POP):
                if self.BUMP_SWITCH[i]:
                    # self.Wab[i, i].bias.data.fill_(0.0)
                    ff_input[:, self.csumNa[i]:self.csumNa[i+1]] = 0.0
                else:
                    ff_input[:, self.csumNa[i]:self.csumNa[i+1]] = self.Ja0[i]
        
        if step in (self.N_STIM_ON, self.N_DIST_ON):
            if np.any(self.I0!=0):
                if self.VERBOSE:
                    print("STIM ON")

                if step == self.N_STIM_ON:
                    ff_input[:, self.csumNa[0]:self.csumNa[1]] = self.Je0 + self.stimFunc(0, 0)
                else:
                    ff_input[:, self.csumNa[0]:self.csumNa[1]] = self.Je0 + self.stimFunc(0, 1)
            
            # if self.PHI0 == 0:
            #     ff_input[self.csumNa[0]:self.csumNa[1]] = self.Ja0[0] * (1.0 + self.ksi[0] * self.I0[0] * self.M0)
            # if self.PHI0 == 180:
            #     ff_input[self.csumNa[0]:self.csumNa[1]] = self.Ja0[0] * (1.0 - self.ksi[0] * self.I0[0] * self.M0)

            # if self.PHI0 == 90:
            #     ff_input[self.csumNa[0]:self.csumNa[1]] = self.Ja0[0] * (1.0 + self.ksi[1] * self.I0[0] * self.M0)
            # if self.PHI0 == 270:
            #     ff_input[self.csumNa[0]:self.csumNa[1]] = self.Ja0[0] * (1.0 - self.ksi[1] * self.I0[0] * self.M0)

        
        if step in (self.N_STIM_OFF, self.N_DIST_OFF):
            if np.any(self.I0!=0):
                if self.VERBOSE:
                    print("STIM OFF")
                ff_input[:, self.csumNa[0]:self.csumNa[1]] = self.Je0

        return ff_input

    def stimFunc(self, i_pop, STIM=0):
        """Stimulus shape"""
        
        if STIM==1:
            # Amp = 1.0 - self.exp_dist.sample((self.N_BATCH,))
            Amp = self.I1[0]
            if self.I1[1]>0:
                 Amp = Amp + self.I1[1] * torch.randn((self.N_BATCH, 1), dtype=self.FLOAT, device=self.DEVICE)
            
            return nn.ReLU()(Amp * (1.0 + self.SIGMA1 * torch.cos(self.theta_list[i_pop] - self.PHI1 * torch.pi / 180.0))) * torch.sqrt(self.Ka[0]) * self.M0
        
        return self.I0[i_pop] * nn.ReLU()(1.0 + self.SIGMA0 * torch.cos(self.theta_list[i_pop] - self.PHI0 * torch.pi / 180.0)) * torch.sqrt(self.Ka[0]) * self.M0
