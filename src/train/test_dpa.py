model_state_dict = torch.load('../models/dual/dpa_%d.pth' % seed)
model.load_state_dict(model_state_dict)
model.eval();

N_BATCH = 1
model.N_BATCH = N_BATCH

ff_input = []

# 2 readouts (sample/choice), 4 conditions AC, AD, BC, BD
labels = np.zeros((2, 4, model.N_BATCH))

l=0
for i in [-1, 1]:
        for k in [-1, 1]:

            model.I0[0] = i # sample
            model.I0[4] = k # test

            if i == 1:
                    labels[1, l] = np.ones(model.N_BATCH)

            if i==k: # Pair Trials
                labels[0, l] = np.ones(model.N_BATCH)

            l+=1
            ff_input.append(model.init_ff_input())

labels = torch.tensor(labels, dtype=torch.float, device=DEVICE).reshape(2, -1)

ff_input = torch.vstack(ff_input)
print('ff_input', ff_input.shape, 'labels', labels.shape)

rates = model.forward(ff_input=ff_input).detach().cpu().numpy()
print(rates.shape)
