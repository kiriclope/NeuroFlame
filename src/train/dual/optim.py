import torch
import torch.optim as optim
from tqdm import tqdm

def training_step(dataloader, model, loss_fn, optimizer, zero_grad=None):

    model.train()
    total_loss = 0.0
    total_batches = len(dataloader)

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(model.device), y.to(model.device)

        optimizer.zero_grad()

        model(X)
        loss = loss_fn(model.readout, y)

        loss.backward()

        if zero_grad is not None:
            try:
                if zero_grad == 'all':
                    model.low_rank.U.grad[:, :] = 0
                    model.low_rank.V.grad[:, :] = 0
                else:
                    model.low_rank.U.grad[:, zero_grad] = 0
                    model.low_rank.V.grad[:, zero_grad] = 0
            except:
                pass

        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / total_batches

    return avg_loss

def validation_step(dataloader, model, loss_fn):
    size = len(dataloader.dataset)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(model.device), y.to(model.device)

            model(X)
            y_pred = model.readout

            batch_loss = loss_fn(y_pred, y)
            val_loss += batch_loss.item() * X.size(0)

    val_loss /= size
    return val_loss

def optimization(model, train_loader, val_loader, loss_fn, optimizer, num_epochs=10, thresh=.15, zero_grad=None):

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.1, verbose=True)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    loss_list = []
    val_loss_list = []

    for epoch in tqdm(range(num_epochs)):
        loss = training_step(train_loader, model, loss_fn, optimizer, zero_grad=zero_grad)
        val_loss = validation_step(val_loader, model, loss_fn)

        scheduler.step(val_loss)
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

        # Early stopping: check if training loss didn't improve sufficiently
        if epoch >= 5:
            recent_losses = loss_list[-5:]
            if max(recent_losses) - min(recent_losses) < 1e-2:
                print('Stopping as training loss did not improve by more than 1e-2 over 5 epochs.')
                break

    return loss_list, val_loss_list
