from matplotlib import pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# Use the Q-table you just trained
Q = np.loadtxt('files/ch14/trained_Q.csv',delimiter=",")

# states and actions in each step
states = [0,4,8,9,13,14]
actions = [1,1,2,1,2,2]
xs = [0,0,2,0,2,2]
ys = [6,2,-2,-3,-7,-8]

def Q_steps():
    for stepi in range(6):
        fig, ax=plt.subplots(figsize=(10,9), dpi=200)
        # table grid
        for x in range(-2,4,1):
            ax.plot([2*x,2*x],[-4.5,4.5],color='gray',linewidth=3)
        for y in range(-9,10,1):
            if y != 8:
                ax.plot([-6,6],[y/2,y/2],color='gray',linewidth=3)
        
        # four actions and 16 states
        plt.text(-3.5, 8.1/2, "action=",fontsize=18)
        for i in range(16):
            plt.text(-3.5, (6.1-i)/2, f"state {i}",fontsize=18)
        actions = ["left", "down", "right", "up"]
        for i in range(4):
            plt.text(-1.3+2*i, 8.1/2, f"{i}",fontsize=18)
            plt.text(-1.5+2*i, 7.2/2, f"{actions[i]}",fontsize=18)
        # write the 64 Q-values onto the graph
        for i in range(16):
            for j in range(4):
                plt.text(-1.8+2*j, (6.2-i)/2, f"{Q[i,j]:.3f}",fontsize=18)
        
        ax.set_xlim(-4,6)
        ax.set_ylim(-4.5,4.5)
        plt.savefig("files/ch14/qtableplt.png")
        plt.axis("off")
        plt.grid()
        plt.savefig(f"files/ch14/plt_Qs_stepa{stepi}.png")
      
        # highlight state row
        ax.add_patch(Rectangle((-4,ys[stepi]/2), 12,0.5,
                     facecolor = 'b',alpha=0.2))
        plt.savefig(f"files/ch14/plt_Qs_stepb{stepi}.png")
        # highlight action cell
        ax.add_patch(Rectangle((xs[stepi], ys[stepi]/2),
                               2,0.5,facecolor='r',alpha=0.8))
        plt.savefig(f"files/ch14/plt_Qs_stepc{stepi}.png")
        plt.close(fig)
        
        
from utils.frozenlake_env import Frozen
import turtle as t

def record_boards():
    env=Frozen()
    state=env.reset()
    env.render()
    step=0
    try:
        ts=t.getscreen() 
    except t.Terminator:
        ts=t.getscreen()
    env.render()
    
    ts.getcanvas().postscript(file=f"files/ch14/board{step}.ps")
    
    while True:
        action=np.argmax(Q[state, :])
        state, reward, done, _ = env.step(action)
        env.render()
        step += 1    
        ts.getcanvas().postscript(file=f"files/ch14/board{step}.ps")
        if done==True:
            break    
    env.close()






from PIL import Image
import numpy as np
import imageio

def board_Q_table():
    for i in range(7):
        im = Image.open(f"files/ch14/board{i}.ps")
        fig, ax=plt.subplots(figsize=(9,9), dpi=200)
        newax = fig.add_axes([0,0,1,1], anchor='NE', zorder=1)
        newax.imshow(im)
        newax.axis('off')
        ax.set_xlim(-4.5,4.5)
        ax.set_ylim(-4.5,4.5)
        plt.axis("off")
        #plt.grid()
        plt.savefig(f"files/ch14/board{i}plt.png")
        plt.close(fig)
        
    frames=[]
    
    for i in range(6):
        for letter in ["a", "b", "c"]:
            im=Image.open(f"files/ch14/board{i}plt.png")
            f0=np.asarray(im)
            im=Image.open(f"files/ch14/plt_Qs_step{letter}{i}.png")
            f1=np.asarray(im)
            fs=np.concatenate([f0,f1],axis=1)
            frames.append(fs)
    im=Image.open("files/ch14/board6plt.png")
    f0=np.asarray(im)
    im=Image.open("files/ch14/plt_Qs_stepa5.png")
    f1=np.asarray(im)
    fs=np.concatenate([f0,f1],axis=1)
    frames.append(fs)
    frames.append(fs)
    frames.append(fs)
    
    imageio.mimsave('files/ch14/frozen_q_steps.gif',frames,duration=500)
    return frames
    
    
    
    



















       
        