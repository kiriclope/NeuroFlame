import torch
from torch import optim, nn
import lightning as L

# Lightning Module
class LRNet(L.LightningModule):
    def __init__(self, model, lr=.01, penalty=None, lbd=0.1):
        super().__init__()
    
        self.model = model
        self.linear = nn.Linear(model.Na[0], 1, device=model.device, dtype=model.FLOAT)
        
        self.lr = lr

        self.criterion = nn.BCEWithLogitsLoss()
        
        self.penalty = penalty
        self.lbd = lbd
        
    def forward(self, ff_input=None):
        rates = self.model.forward(ff_input)
        y_pred = self.linear(rates[:, -self.lr_eval_win:])
        return y_pred.squeeze(-1)


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters, lr=self.lr)
        return optimizer
            
    def training_step(self, train_batch, batch_idx):
        X, y = train_batch

        y_pred = self.model(X)
        loss = self.criterion(y_pred, y)
        
        if self.penalty is not None:
            reg_loss = 0
            for param in self.parameters():
                if self.penalty=='l1':
                    reg_loss += torch.sum(torch.abs(param))
                else:
                    reg_loss += torch.sum(torch.square(param))
                
                loss = loss + self.lbd * reg_loss
                
        self.log('train_loss', loss, on_epoch=True)
        
        return loss

    def validation_step(self, val_batch, batch_idx):
        X, y = val_batch
        
        y_pred = self.model(X)
        loss = self.criterion(y_pred, y)

        self.log('val_loss', loss)
        
