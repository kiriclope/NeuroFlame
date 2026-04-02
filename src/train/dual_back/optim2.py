import torch
import torch.nn as nn

class Optim(nn.module):
    def __init__(self, model, loss, optimizer, scheduler):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler


    def train_step(self, dataloader):
        device = self.model.device

        self.model.train()
        total_loss = 0.0
        total_batches = len(dataloader)

        for (X, y) in dataloader:
            X, y = X.to(device), y.to(device)

            self.optimizer.zero_grad()

            y_pred = self.model(X)
            loss = self.loss(y_pred, y)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / total_batches
        return avg_loss


    def val_step(self, dataloader):
        device = self.model.device
        size = len(dataloader.dataset)

        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)

                y_pred = self.model(X)
                batch_loss = self.loss(y_pred, y)
                val_loss += batch_loss.item() * X.size(0)

        val_loss /= size
        return val_loss


    def optimization(self, train_loader, val_loader, num_epochs=100, thresh=0.005):

        loss_list = []
        val_loss_list = []

        for epoch in range(num_epochs):
            loss = self.train_step(train_loader)
            val_loss = self.val_step(val_loader)

            self.scheduler.step()

            loss_list.append(loss)
            val_loss_list.append(val_loss)

            print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {loss:.4f}, Validation Loss: {val_loss:.4f}')

            if val_loss < thresh and loss < thresh:
                print(f'Stopping training as loss has fallen below the threshold: {loss}, {val_loss}')
                break

            if val_loss > 300:
                print(f'Stopping training as loss is too high: {val_loss}')
                break

            if torch.isnan(torch.tensor(loss)):
                print(f'Stopping training as loss is NaN.')
                break

        return loss_list, val_loss_list
