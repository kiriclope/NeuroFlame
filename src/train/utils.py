import pickle as pkl
import os


def pkl_save(obj, name, path="."):
    os.makedirs(path, exist_ok=True)
    destination = path + "/" + name + ".pkl"
    print("saving to", destination)
    pkl.dump(obj, open(destination, "wb"))


def pkl_load(name, path="."):
    source = path + "/" + name + '.pkl'
    # print('loading from', source)
    return pkl.load(open( source, "rb"))


def convert_seconds(seconds):
      h = seconds // 3600
      m = (seconds % 3600) // 60
      s = seconds % 60
      return h, m, s
