import torch
from torch.distributions import MultivariateNormal


class Connectivity:
    def __init__(self, Na, Nb, Kb, device, verbose=0):
        self.Na = torch.tensor(Na).detach().clone()
        self.Nb = torch.tensor(Nb).detach().clone()
        self.Kb = torch.tensor(Kb).detach().clone()
        self.verbose = verbose
        self.device = device

    def low_rank_proba(self, kappa, lr_mean, lr_cov, ksi=None, **kwargs):
        if ksi is None:
            if self.verbose:
                print("Generating low rank vectors")
            mean_ = torch.tensor(lr_mean, device=self.device)
            cov_ = torch.tensor(lr_cov, device=self.device)
            if mean_.shape[0] > 1:
                mv_normal = MultivariateNormal(mean_, cov_)
                self.ksi = mv_normal.sample((self.Nb,)).T
            else:
                self.ksi = torch.randn((1, self.Nb), device=self.device)
        else:
            self.ksi = ksi

        if self.verbose:
            print("ksi", self.ksi.shape)

        if self.ksi.shape[0] == 4:
            Lij = torch.outer(self.ksi[0], self.ksi[1])
            Lij = Lij + torch.outer(self.ksi[2], self.ksi[3])
            Pij = 1.0 + kappa * Lij / torch.sqrt(self.Kb)
        else:
            Pij = 1.0 + kappa * (self.ksi.T @ self.ksi) / torch.sqrt(self.Kb)
        return Pij

    def cosine_proba(self, kappa, phase=0):
        theta_list = torch.linspace(0, 2.0 * torch.pi, self.Na + 1, device=self.device)[:-1]
        phi_list = torch.linspace(0, 2.0 * torch.pi, self.Nb + 1, device=self.device)[:-1]
        theta_i, theta_j = torch.meshgrid(theta_list, phi_list, indexing="ij")
        theta_diff = theta_i - theta_j
        return 1.0 + kappa * torch.cos(theta_diff - phase)

    def von_mises_proba(self, kappa):
        theta_list = torch.linspace(0, 2.0 * torch.pi, self.Na + 1, device=self.device)[:-1]
        phi_list = torch.linspace(0, 2.0 * torch.pi, self.Nb + 1, device=self.device)[:-1]
        theta_i, theta_j = torch.meshgrid(theta_list, phi_list, indexing="ij")
        theta_diff = theta_i - theta_j
        return (
            torch.exp(kappa * torch.cos(theta_diff))
            / torch.special.i0(torch.tensor(kappa, device=self.device))
            / 2.0
            / torch.pi
        )

    def get_con_proba(self, proba_type, **kwargs):
        if "cos" in proba_type:
            if "spec" in proba_type:
                Pij = self.cosine_proba(kwargs["kappa"] / torch.sqrt(self.Kb))
            else:
                Pij = self.cosine_proba(kwargs["kappa"], kwargs["phase"])
        elif "lr" == proba_type:
            Pij = self.low_rank_proba(**kwargs)
        elif "von_mises" in proba_type:
            Pij = self.von_mises_proba(kwargs["kappa"])
        elif "gaussian" in proba_type:
            Pij = torch.randn((self.Na, self.Nb), device=self.device)
        else:
            Pij = torch.tensor(1.0, device=self.device)
        return Pij

    def forward(self, con_type, proba_type, **kwargs):
        Pij = self.get_con_proba(proba_type=proba_type, **kwargs)
        if "sparse" in con_type:
            Cij = torch.rand(self.Na, self.Nb, device=self.device) <= (
                self.Kb / float(self.Nb) * Pij
            ).clamp_(min=0, max=1)
        else:
            if "dense" in proba_type:
                Cij = Pij / torch.sqrt(1.0 * self.Nb)
            else:
                Cij = Pij / (1.0 * self.Nb)
        return Cij

    def __call__(self, con_type="sparse", proba_type="unif", **kwargs):
        return self.forward(con_type, proba_type, **kwargs)
