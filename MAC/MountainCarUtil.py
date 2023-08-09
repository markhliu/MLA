import os
import gym
import matplotlib.pyplot as plt
os.environ["SDL_VIDEODRIVER"] = "dummy"
from IPython.display import clear_output
import pickle
import tarfile


with open("mountain_car.p","rb") as f:
    f0=pickle.load(f)
      

def get_frame(obs):
    pos=int(obs[0]*100)
    arr=f0.get(pos)
    if arr is not None:
        return arr 
    for i in range(1,49):
        if f0.get(pos-i) is not None:
            return f0.get(pos-i)
        if f0.get(pos+i) is not None:
            return f0.get(pos+i)
    return f0.get(0)    



