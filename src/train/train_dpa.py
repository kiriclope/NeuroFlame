model.J_STP.requires_grad = False

if model.LR_READOUT:
    for param in model.low_rank.linear.parameters():
        param.requires_grad = False
    model.low_rank.linear.bias.requires_grad = False

if model.LR_KAPPA:
    model.low_rank.lr_kappa.requires_grad = True

steps = np.arange(0, model.N_STEPS - model.N_STEADY, model.N_WINDOW)
mask = (steps >= (model.N_STIM_ON[4].cpu().numpy() - model.N_STEADY)) & (steps <= (model.N_STEPS - model.N_STEADY))
rwd_idx = np.where(mask)[0]
print('rwd', rwd_idx)

# mask for A/B memory from sample to test
stim_mask = (steps >= (model.N_STIM_ON[0].cpu().numpy() - model.N_STEADY)) & (steps < (model.N_STIM_ON[-1].cpu().numpy() - model.N_STEADY))
stim_idx = np.where(stim_mask)[0]
print('stim', stim_idx)

model.lr_eval_win = np.max((rwd_idx.shape[0], stim_idx.shape[0]))

mask_zero = ~mask  # & ~stim_mask
zero_idx = np.where(mask_zero)[0]
print('zero', zero_idx)

N_BATCH = 256
model.N_BATCH = N_BATCH

model.lr_eval_win = np.max( (rwd_idx.shape[0], stim_idx.shape[0]))

ff_input = []
labels = np.zeros((2, 4, model.N_BATCH, model.lr_eval_win))

l=0
for i in [-1, 1]:
        for k in [-1, 1]:

            model.I0[0] = i # sample
            model.I0[4] = k # test

            if i == 1:
                    labels[1, l] = np.ones((model.N_BATCH, model.lr_eval_win))

            if i==k: # Pair Trials
                labels[0, l] = np.ones((model.N_BATCH, model.lr_eval_win))

            l+=1

            ff_input.append(model.init_ff_input())

labels = torch.tensor(labels, dtype=torch.float, device=DEVICE).reshape(2, -1, model.lr_eval_win).transpose(0, 1)

ff_input = torch.vstack(ff_input)
print('ff_input', ff_input.shape, 'labels', labels.shape)

splits = [split_data(ff_input, labels, train_perc=0.8, batch_size=batch_size)]
# splits = cross_val_data(ff_input, labels, n_splits=5, batch_size=batch_size)
criterion = DualLoss(alpha=1.0, thresh=4.0, rwd_idx=rwd_idx, stim_idx=stim_idx, zero_idx=zero_idx, imbalance=[1.0, 1.0], read_idx=[1, 0])
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print('training DPA')
num_epochs = 15
start = perf_counter()
for train_loader, val_loader in splits:
    loss, val_loss = optimization(model, train_loader, val_loader, criterion, optimizer, num_epochs, zero_grad=None)
end = perf_counter()
print("Elapsed (with compilation) = %dh %dm %ds" % convert_seconds(end - start))
torch.save(model.state_dict(), '../models/dual/dpa_%d.pth' % seed)

torch.save(model.state_dict(), '../models/dual/dpa_%d.pth' % seed)
