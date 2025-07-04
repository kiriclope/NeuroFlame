model_state_dict = torch.load('../models/dual/dpa_%d.pth' % seed)
model.load_state_dict(model_state_dict)

model.J_STP.requires_grad = False
model.low_rank.lr_kappa.requires_grad = False

for name, param in model.named_parameters():
      if param.requires_grad:
            print(name, param.shape)

steps = np.arange(0, model.N_STEPS - model.N_STEADY, model.N_WINDOW)

# mask for lick/nolick  from cue to test
rwd_mask = (steps >= (model.N_STIM_ON[2].cpu().numpy() - model.N_STEADY)) & (steps < (model.N_STIM_ON[4].cpu().numpy() - model.N_STEADY))
rwd_idx = np.where(rwd_mask)[0]
print('rwd', rwd_idx)

# mask for Go/NoGo memory from dist to cue
stim_mask = (steps >= (model.N_STIM_ON[1].cpu().numpy() - model.N_STEADY)) & (steps < (model.N_STIM_ON[2].cpu().numpy() - model.N_STEADY))
stim_idx = np.where(stim_mask)[0]
# stim_idx = []
print('stim', stim_idx)

mask_zero = (steps < (model.N_STIM_ON[1].cpu().numpy() - model.N_STEADY))
zero_idx = np.where(mask_zero)[0]
print('zero', zero_idx)

model.lr_eval_win = np.max( (rwd_idx.shape[0], stim_idx.shape[0]))

model.N_BATCH = 512

model.I0[0] = 0
model.I0[1] = A0
model.I0[2] = float(B0)
model.I0[3] = 0
model.I0[4] = 0

Go = model.init_ff_input()

model.I0[0] = 0
model.I0[1] = -A0
model.I0[2] = float(B0)
model.I0[3] = 0
model.I0[4] = 0

NoGo = model.init_ff_input()

ff_input = torch.cat((Go, NoGo))
print(ff_input.shape)

labels_Go = torch.ones((model.N_BATCH, model.lr_eval_win))
labels_NoGo = torch.zeros((model.N_BATCH, model.lr_eval_win))
labels = torch.cat((labels_Go, labels_NoGo))
print(labels.shape)
labels =  labels.repeat((2, 1, 1))
labels = torch.transpose(labels, 0, 1)
print('labels', labels.shape)

train_loader, val_loader = split_data(ff_input, labels, train_perc=0.8, batch_size=batch_size)
criterion = DualLoss(alpha=1.0, thresh=4.0, rwd_idx=rwd_idx, zero_idx=zero_idx, stim_idx=stim_idx, imbalance=[0.0, 1.0], read_idx=[1, 1])
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print('training DRT')
num_epochs = 15
start = perf_counter()
loss, val_loss = optimization(model, train_loader, val_loader, criterion, optimizer, num_epochs, zero_grad=0)
end = perf_counter()
print("Elapsed (with compilation) = %dh %dm %ds" % convert_seconds(end - start))

torch.save(model.state_dict(), '../models/dual/dual_naive_%d.pth' % seed)
