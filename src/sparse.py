import torch

Class SparseMatrix():
    def __init__(self, in_deg, n_cols, n_rows):
        self.n_cols = n_cols
        self.n_rows = n_rows
        self.in_deg = in_deg

        self.proba = in_deg / n_cols

    def forward(self):

        indices
        
        for i in range(n_rows):
            for j in range(n_cols):
                if torch.rand(1) <= self.proba:
                    
