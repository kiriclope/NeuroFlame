import os
import gc
import numpy as np
import torch
from torch import nn
from yaml import safe_load
from time import perf_counter
import warnings

warnings.filterwarnings("ignore")
# torch.backends.cudnn.deterministic = False
# torch.backends.cudnn.benchmark = False
# torch.manual_seed(1)

# class Activation(torch.nn.Module):
#     def __init__(self, func_name='erf'):
#         super().__init__()
#         self.func_name = func_name
        
#     def forward(self, x, THRESH=15):
#         if self.func_name == 'erf':
#             return THRESH * 0.5 / (1.0 + torch.erf(x / np.sqrt(2.0)))
#         elif self.func_name == 'relu':
#             return nn.ReLU()(x - THRESH)
#         else:
#             raise ValueError('Invalid function name')

class Activation(torch.nn.Module):
    def forward(self, x, THRESH=15):
        # return nn.ReLU()(x)
        return THRESH * 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))

class Network(nn.Module):
    def __init__(self, conf_file, sim_name, repo_root, **kwargs):
        super().__init__()

        # load parameters
        self.loadConfig(conf_file, sim_name, repo_root, **kwargs)

        # set csts
        self.initConst()

        # scale parameters
        self.scaleParam()
        
        # Create recurrent layer with nn.Linear
        self.Wab = [[None]*self.N_POP for _ in range(self.N_POP)]

        for i_pop in range(self.N_POP):
            for j_pop in range(self.N_POP):
                Wij = nn.Linear(self.Na[i_pop], self.Na[j_pop], bias=(i_pop==j_pop), dtype=self.FLOAT, device=self.device)
                Wij.weight.data = self.initWeights(i_pop, j_pop)
                self.Wab[i_pop][j_pop] = Wij
                self.Wab[i_pop][j_pop] = self.Wab[i_pop][j_pop]

        # Here we store the constant external input in the bias
        for i_pop in range(self.N_POP):
            self.Wab[i_pop][i_pop].bias.data.fill_(self.Ja0[i_pop])


    def update_rec_input(self, rates):
        '''Dynamics of the recurrent inputs'''
        rec_input = torch.zeros((self.N_POP, self.N_NEURON), device=self.device)

        for j_pop in range(self.N_POP):                
            for i_pop in range(self.N_POP):
                input = rec_input[j_pop][self.csumNa[i_pop] : self.csumNa[i_pop + 1]]
                W = self.Wab[i_pop][j_pop](rates[self.csumNa[j_pop] : self.csumNa[j_pop + 1]])
                rec_input[j_pop][self.csumNa[i_pop] : self.csumNa[i_pop + 1]] = input + W 

                
        return rec_input

    def update_net_input(self, rec_input):
        '''Updating the net input into the neurons'''

        net_input = torch.zeros(self.N_NEURON, dtype=self.FLOAT, device=self.device)
        
        for  i_pop in range(self.N_POP):
            if self.VAR_FF[i_pop] > 0:
                net_input[self.csumNa[i_pop] : self.csumNa[i_pop + 1]] = torch.normal(mean=0,
                                                                                      std=self.VAR_FF[i_pop],
                                                                                      size=(self.Na[i_pop],),
                                                                                      dtype=self.FLOAT,
                                                                                      device=self.device)
        
        for i_pop in range(self.N_POP):
            net_input = net_input + rec_input[i_pop]
        
        return net_input

    def update_rates(self, rates, net_input):
        '''Dynamics of the rates'''
        # using array slices is faster than indices
        if self.RATE_DYN:
            for i_pop in range(self.N_POP):
                rates[self.csumNa[i_pop] : self.csumNa[i_pop + 1]] = rates[self.csumNa[i_pop] : self.csumNa[i_pop + 1]] * self.EXP_DT_TAU[i_pop]
                rates[self.csumNa[i_pop] : self.csumNa[i_pop + 1]] = rates[self.csumNa[i_pop] : self.csumNa[i_pop + 1]] + self.DT_TAU[i_pop] * Activation()(net_input[self.csumNa[i_pop] : self.csumNa[i_pop + 1]])
        else:
            rates = Activation()(net_input)
            
        return rates

    def forward(self, rates):
        '''This is the main function of the class'''
        rec_input = self.update_rec_input(rates)
        net_input = self.update_net_input(rec_input)
        rates = self.update_rates(rates, net_input)

        return rates

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
    
    def run(self):
        result = []
        # init rates
        hidden = self.initRates()

        start = perf_counter()
        
        self.N_STEPS = int(self.DURATION / self.DT)
        self.N_STEADY = int(self.T_STEADY / self.DT)
        self.N_WINDOW = int(self.T_WINDOW / self.DT)
        self.N_STIM_ON = int(self.T_STIM_ON / self.DT)
        self.N_STIM_OFF = int(self.T_STIM_OFF / self.DT)
        
        for step in range(self.N_STEPS):
            self.update_stim(step)
            hidden = self.forward(hidden)
            
            if step > self.N_STEADY:
                if step % self.N_WINDOW == 0:
                    if self.VERBOSE:
                        self.print_activity(step, hidden)
                
                result.append(hidden.cpu().detach().numpy())
        
        result = np.array(result)

        print('Saving rates to:', self.DATA_PATH + self.FILE_NAME + '.npy')
        np.save(self.DATA_PATH + self.FILE_NAME + '.npy', result)
        
        for obj in gc.get_objects():
            if torch.is_tensor(obj):
                del obj
        
        # Manually triggering the garbage collector afterwards
        gc.collect()
        torch.cuda.empty_cache()
        
        end = perf_counter()

        print("Elapsed (with compilation) = {}s".format((end - start)))

        return result

    def loadConfig(self, conf_file, sim_name, repo_root, **kwargs):
        # Loading configuration file
        conf_path = repo_root + '/conf/'+ conf_file
        print('Loading config from', conf_path)
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

    def initRates(self):
        return torch.zeros(self.N_NEURON, dtype=self.FLOAT, device=self.device)
    
    def initWeights(self, i_pop, j_pop):

        Na = self.Na[i_pop]
        Nb = self.Na[j_pop]
        Kb = self.Ka[j_pop]
        
        Pij = 1.0

        if 'cos' in self.STRUCTURE[i_pop][j_pop]:
            theta = torch.arange(0, 2.0 * torch.pi, 2.0 * torch.pi / float(Na), dtype=self.FLOAT, device=self.device)
            phi = torch.arange(0, 2.0 * torch.pi, 2.0 * torch.pi / float(Nb), dtype=self.FLOAT, device=self.device)

            i, j = torch.meshgrid(torch.arange(Na, device=self.device), torch.arange(Nb, device=self.device), indexing="ij")
            
            theta_diff = theta[i] - phi[j]

            if 'spec' in self.STRUCTURE[i_pop][j_pop]:
                self.KAPPA[i_pop][j_pop] = self.KAPPA[i_pop][j_pop] / torch.sqrt(self.Ka[j_pop])

            Pij = 1.0 + 2.0 * self.KAPPA[i_pop][j_pop] * torch.cos(theta_diff - self.PHASE)

        if 'sparse' in self.CONNECTIVITY:
            if self.VERBOSE:
                print('Sparse random connectivity ')
            Cij = self.Jab[i_pop][j_pop] * (torch.rand(Na, Nb, device=self.device) < Kb / float(Nb) * Pij)

        if 'all2all' in self.CONNECTIVITY:
            if self.VERBOSE:
                print('All to all connectivity ')
                
            Cij = self.Jab[i_pop][j_pop] * Pij / float(Nb)

            if self.SIGMA[i_pop][j_pop] > 0.0:
                if self.VERBOSE:
                    print('with heterogeneity, SIGMA', self.SIGMA[i_pop][j_pop])
                
                Hij = torch.normal(0, self.SIGMA[i_pop][j_pop], size=(Na, Nb), dtype=self.FLOAT, device=self.device)
                Cij = Cij + Hij / float(Nb)
                
        if self.VERBOSE:
            if "cos" in self.STRUCTURE[i_pop][j_pop]:
                if "spec" in self.STRUCTURE[i_pop][j_pop]:
                    print('with weak cosine structure, KAPPA %.2f' % self.KAPPA[i_pop][j_pop].cpu().detach().numpy())
                else:
                    print('with strong cosine structure, KAPPA', self.KAPPA[i_pop][j_pop])

        return Cij

    def initConst(self):
        self.Na = []
        self.Ka = []

        for i_pop in range(self.N_POP):
            self.Na.append(int(self.N_NEURON * self.frac[i_pop]))
            # self.Ka.append(self.K * const.frac[i_pop])
            self.Ka.append(self.K)

        self.Na = torch.tensor(self.Na, dtype=torch.int)
        self.Ka = torch.tensor(self.Ka, dtype=self.FLOAT)
        self.csumNa = torch.cat((torch.tensor([0]), torch.cumsum(self.Na, dim=0)))

        if self.VERBOSE:
            print("Na", self.Na, "Ka", self.Ka, "csumNa", self.csumNa)

        self.EXP_DT_TAU = []
        self.DT_TAU = []

        for i_pop in range(self.N_POP):
            self.EXP_DT_TAU.append(np.exp(-self.DT / self.TAU[i_pop]))
            self.DT_TAU.append(self.DT / self.TAU[i_pop])

        if self.VERBOSE:
            print("DT", self.DT, "TAU", self.TAU)

        # if self.VERBOSE:
        #     print("EXP_DT_TAU", self.EXP_DT_TAU, "DT_TAU", self.DT_TAU)

        self.STRUCTURE = np.array(self.STRUCTURE).reshape(self.N_POP, self.N_POP)
        self.SIGMA = torch.tensor(self.SIGMA, dtype=self.FLOAT).view(self.N_POP, self.N_POP)
        self.KAPPA = torch.tensor(self.KAPPA, dtype=self.FLOAT).view(self.N_POP, self.N_POP)
        self.PHASE = self.PHASE * torch.pi / 180.0

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

        for i_pop in range(self.N_POP):
            self.VAR_FF[i_pop] = np.sqrt(self.VAR_FF[i_pop])
            
    def update_stim(self, step):
        """Perturb the inputs based on the simulus parameters."""

        if step == 0:
            for i in range(self.N_POP):
                if self.BUMP_SWITCH[i]:
                    self.Wab[i][i].bias.data.fill_(0.0)
                if self.K !=1 and self.BUMP_SWITCH[i]:
                    self.Wab[i][i].bias.data.fill_(self.Ja0[i] / torch.sqrt(self.Ka[0]))

        if step == self.N_STIM_ON:
            for i in range(self.N_POP):
                if self.BUMP_SWITCH[i]:
                    self.Wab[i][i].bias.data.fill_(self.Ja0[i])
        
        if step == self.N_STIM_ON and np.any(self.I0!=0):
            if self.VERBOSE:
                print("STIM ON")
            
            for i in range(self.N_POP):
                self.Wab[i][i].bias.data = self.Ja0[i] + self.stimFunc(i)
            
        if step == self.N_STIM_OFF and np.any(self.I0!=0):
            if self.VERBOSE:
                print("STIM OFF")

            for i in range(self.N_POP):
                self.Wab[i][i].bias.data.fill_(self.Ja0[i])

    def stimFunc(self, i_pop):
        """Stimulus shape"""
        theta = torch.arange(0, 2.0 * torch.pi, 2.0 * torch.pi / float(self.Na[i_pop]), dtype=self.FLOAT, device=self.device)
        return self.I0[i_pop] * (1.0 + self.SIGMA0 * torch.cos(theta - self.PHI0 * torch.pi / 180.0)) * self.Ka[0] * self.M0
