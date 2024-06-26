import torch
from torch import optim, nn
import lightning as L

# Lightning Module
class LRNet(L.LightningModule):
    def __init__(self, model, lr=.01, penalty=None, lbd=0.1):
        super().__init__()

        self.model = model
        self.linear = nn.Linear(model.Na[0], 1, device=model.device)

        self.lr = lr

        self.criterion = nn.BCEWithLogitsLoss()

        self.penalty = penalty
        self.lbd = lbd

    def forward(self, ff_input=None):
        rates = self.model.forward(ff_input)
        y_pred = self.linear(rates)
        return y_pred[:, -1, :]

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        X, y = train_batch

        y_pred = self.forward(X)
        loss = self.criterion(y_pred[..., -1].unsqueeze(-1), y)

        if self.penalty is not None:
            reg_loss = 0
            for param in self.parameters():
                if self.penalty=='l1':
                    reg_loss += torch.sum(torch.abs(param))
                else:
                    reg_loss += torch.sum(torch.square(param))

                loss = loss + self.lbd * reg_loss

        self.log('train_loss', loss, prog_bar=True, logger=True)

        return loss

    def validation_step(self, val_batch, batch_idx):
        X, y = val_batch

        y_pred = self.forward(X)
        loss = self.criterion(y_pred, y)

        self.log('val_loss', loss, prog_bar=True, logger=True)

    # def on_train_epoch_end(self):
    # avg_train_loss = self.trainer.callback_metrics["train_loss"]
    # print(f"Epoch {self.current_epoch} - Training loss: {avg_train_loss.item()}")

    def on_validation_epoch_end(self):
        avg_train_loss = self.trainer.callback_metrics["train_loss"]
        avg_val_loss = self.trainer.callback_metrics["val_loss"]

        print(f"Epoch {self.current_epoch} - Training loss: {avg_train_loss.item()} - Validation loss: {avg_val_loss.item()}")
