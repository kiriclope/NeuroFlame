import torch
from torch.distributions import MultivariateNormal

class Connectivity():
    def __init__(self, Na, Nb, Kb, device='cuda', dtype=torch.float, verbose=0):
        '''
        Class: Connectivity
        Creates a connectivity matrix. The connectivity can be sparse are all to all.
        Connections can be tuned with different profiles see (forward method).
        Parameters:
        Na : int, number of postsynaptic neurons
        Nb : int, number of presynaptic neurons
        Kb : float, in degree
        '''
        
        self.Na = torch.tensor(Na)
        self.Nb = torch.tensor(Nb)
        self.Kb = torch.tensor(Kb)
        
        self.verbose = verbose
        self.device = device
        self.dtype = dtype
        
    def low_rank_proba(self, kappa, lr_mean, lr_cov, ksi=None):
        '''returns Low rank probability of connection'''
        
        if ksi is None:
            if self.verbose:
                print('Generating low rank vectors')
            mean_ = torch.tensor(lr_mean, dtype=self.dtype, device=self.device)
            cov_ = torch.tensor(lr_cov, dtype=self.dtype, device=self.device)
            
            if mean_.shape[0]>1:
                mv_normal = MultivariateNormal(mean_, cov_)
                self.ksi = mv_normal.sample((self.Nb,)).T
            else:
                self.ksi = torch.randn((1, self.Nb), device=self.device, dtype=self.dtype)
            
            del mean_
            del cov_
            del mv_normal
        else:
            self.ksi = ksi
        
        if self.verbose:
            print('ksi', self.ksi.shape)


        if self.ksi.shape[0]==4:
            Lij = torch.outer(self.ksi[0], self.ksi[1])
            Lij = Lij + torch.outer(self.ksi[2], self.ksi[3])
            
            Pij = 1.0 + kappa * Lij / torch.sqrt(self.Kb)
            del Lij
        else:
            Pij = 1.0 + kappa * (ksi.T @ ksi) / torch.sqrt(self.Kb)
        
        # print(Pij.shape)
        
        # Pij[Pij>1] = 1
        # Pij[Pij<0] = 0
        
        return Pij
    
    def cosine_proba(self, kappa, phase=0):
        '''returns cosine probability of connection'''
        
        theta_list = torch.linspace(0, 2.0 * torch.pi, self.Na + 1, dtype=self.dtype, device=self.device)[:-1]
        phi_list = torch.linspace(0, 2.0 * torch.pi, self.Nb + 1, dtype=self.dtype, device=self.device)[:-1]
        
        theta_i, theta_j = torch.meshgrid(theta_list, phi_list, indexing="ij")
        theta_diff = theta_i - theta_j
        
        Pij = 1.0 + kappa * torch.cos(theta_diff - phase)
        
        del theta_list
        del phi_list
        
        del theta_i
        del theta_j
        del theta_diff
        
        return Pij

    def get_con_proba(self, proba_type, **kwargs):
        '''returns probability of connection of type proba_type with footprint/strength kappa'''
        
        if 'cos' in proba_type:
            if 'spec' in proba_type:
                Pij = self.cosine_proba(kwargs['kappa'] / torch.sqrt(self.Kb))
                if self.verbose:
                    print('weak cosine probability')
            else:
                Pij = self.cosine_proba(kwargs['kappa'], kwargs['phase'])
                if self.verbose:
                    print('strong cosine probability')
        elif 'lr' in proba_type:
            Pij = self.low_rank_proba(kwargs['kappa'], kwargs['lr_mean'], kwargs['lr_cov'], kwargs['ksi'])
            if self.verbose:
                print('low rank probability')
        else:
            Pij = torch.tensor(1.0, dtype=self.dtype, device=self.device)
            if self.verbose:
                print('uniform probability')
                
        return Pij
    
    def forward(self, con_type, proba_type, **kwargs):
        '''
        returns connectivity Cij
        :param con_type: string either 'all2all' or 'sparse'
        :param proba_type: string either 'cosine' or 'lr'
        :param kappa: float
        :param phase: float
        :param sigma: float
        :param lr_mean: array
        :param lr_cov: array
        '''
        
        Pij = self.get_con_proba(proba_type=proba_type, **kwargs)
        
        # sparse network with probability of connection Kb/Nb * Pij
        if 'sparse' in con_type:
            if self.verbose:
                print('Sparse random connectivity')
            
            Cij = torch.rand(self.Na, self.Nb, device=self.device) <= (self.Kb / float(self.Nb) * Pij)
            
        # fully connected network that scales as 1/Nb
        if 'all2all' in con_type:
            if self.verbose:
                print('All to all connectivity')
            
            Cij = Pij / float(self.Nb)
            
            # adds heterogeneity that scales as 1/sqrt(Nb)            
            if 'sigma' in kwargs:
                if self.verbose:
                    print('with heterogeneity, SIGMA', kwargs['sigma'])
                
                Hij = kwargs['sigma'] * torch.randn((self.Na, self.Nb), dtype=self.dtype, device=self.device)
                Cij = Cij + Hij / torch.sqrt(self.Nb)
            
            del Hij
            
        del Pij
        
        if self.verbose:
            if "cos" in proba_type:
                if "spec" in proba_type:
                    print('with weak cosine structure, KAPPA %.2f' % self.kappa.cpu().detach().numpy())
                else:
                    print('with strong cosine structure, KAPPA', self.kappa.cpu().detach().numpy())
            elif "lr" in proba_type:
                print('with weak low rank structure, KAPPA %.2f' % self.kappa.cpu().detach().numpy())
        
        return Cij
    
    def __call__(self, con_type='sparse', proba_type='unif', **kwargs):
        # This method will be called when you do Conn()()
        return self.forward(con_type, proba_type, **kwargs)
