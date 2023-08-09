import os
import gym
import matplotlib.pyplot as plt
os.environ["SDL_VIDEODRIVER"] = "dummy"
from IPython.display import clear_output
import pickle
import tarfile

try:
    with open("cartpolearr.p","rb") as f:
        f0=pickle.load(f)
except:
    try:
        file=tarfile.open('CartPole.tar.gz')
        file.extractall('')
        file.close()
        with open("cartpolearr.p","rb") as f:
            f0=pickle.load(f)  
    except:
        print('''please download CartPole.tar.gz from
              the book's GitHub repo and place in the 
              folder /Desktop/mla/MAC/''')                  

def get_frame(obs):
    pos0=int(100*obs[0])
    pos2=int(100*obs[2])
    arr=f0.get((pos0,pos2))
    if arr is not None:
        return arr 
    for i in range(1,49):
        if f0.get((pos0-i,pos2)) is not None:
            return f0.get((pos0-i,pos2))
        if f0.get((pos0,pos2-i)) is not None:
            return f0.get((pos0,pos2-i))
        if f0.get((pos0+i,pos2)) is not None:
            return f0.get((pos0+i,pos2))
        if f0.get((pos0,pos2+i)) is not None:
            return f0.get((pos0,pos2+i))
    return f0.get((0,0))    



