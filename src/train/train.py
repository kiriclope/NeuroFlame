REPO_ROOT = "/home/leon/models/NeuroFlame"
conf_name = "train_dual.yml"
DEVICE = 'cuda:0'

# seed = np.random.randint(0, 1e6)
seed = 2
print(seed)

A0 = 1.0 # sample/dist
B0 = 1.0 # cue
C0 = 0.0 # DRT rwd

model = Network(conf_name, REPO_ROOT, VERBOSE=0, DEVICE=DEVICE, SEED=seed, N_BATCH=1)
device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
model.to(device)

batch_size = 16
learning_rate = 0.1
