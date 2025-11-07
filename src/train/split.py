from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import Dataset, DataLoader, TensorDataset


def split_data(X, Y, train_perc=0.8, batch_size=32):

    if Y.ndim==3:
      X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                          train_size=train_perc,
                                                          stratify=Y[:, 0, 0].cpu().numpy(),
                                                          shuffle=True)
    else:
      X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                          train_size=train_perc,
                                                          stratify=Y[:, 0].cpu().numpy(),
                                                          shuffle=True)

    # print(X_train.shape, X_test.shape)
    # print(Y_train.shape, Y_test.shape)

    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_test, Y_test)

    # Create data loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def cross_val_data(X, Y, n_splits=5, batch_size=32):
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True)

    for fold, (train_index, val_index) in enumerate(kf.split(X, Y[:, 0].cpu().numpy() if Y.ndim == 2 else Y[:, 0, 0].cpu().numpy())):
        X_train, X_val = X[train_index], X[val_index]
        Y_train, Y_val = Y[train_index], Y[val_index]

        print(f'Fold {fold}')
        print("Train:", X_train.shape, Y_train.shape)
        print("Val:", X_val.shape, Y_val.shape)

        train_dataset = TensorDataset(X_train, Y_train)
        val_dataset = TensorDataset(X_val, Y_val)

        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

        yield train_loader, val_loader
