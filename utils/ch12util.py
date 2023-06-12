from utils.ttt_env import ttt
import time
import random
from copy import deepcopy
import numpy as np
import tensorflow as tf
import turtle as t

reload = tf.keras.models.load_model('files/ch12/trained_ttt100K.h5')



def best_move(board, valids):
    # if there is only one valid move, take it
    if len(valids)==1:
        return valids[0]
    # Set the initial value of bestoutcome        
    bestoutcome=-1;
    bestmove=None  
    # record winning probabilities for all hypothetical moves
    p_wins={}
    #go through all possible moves hypothetically to predict outcome
    for move in valids:
        tooccupy=deepcopy(board).reshape(9,)
        tooccupy[int(move)-1]=1
        prediction=reload.predict(np.array(tooccupy).reshape(-1,3,3,1),verbose=0)
        p_win=prediction[0][1]
        p_wins[move]=p_win
        if p_win>bestoutcome:
            # Update the bestoutcome
            bestoutcome = p_win
            # Update the best move
            bestmove = move
    return bestmove, p_wins

def record():
    # Initiate the game environment
    env=ttt()
    state=env.reset()  
    step=0
    try:
        ts=t.getscreen() 
    except t.Terminator:
        ts=t.getscreen()
    t.hideturtle()
    env.render()
    ts.getcanvas().postscript(file=f"files/ch12/ttt_step{step}.ps")

    # Create a list to record game history
    history=[]
    # Play a full game 
    while True:
        print(f"the current state is \n{state}") 
        bestmove,p_wins=best_move(env.state, env.validinputs)
        action=bestmove
        print(f"Player X has chosen action={action}")  
        old_state=deepcopy(state)
        new_state, reward, done, info = env.step(action)
        history.append([old_state,p_wins,action,deepcopy(new_state),done])
        env.render()
        step += 1      
        ts.getcanvas().postscript(file=f"files/ch12/ttt_step{step}.ps")
        if done:
            if reward==1:
                print(f"Player X has won!") 
            else:
                print(f"It's a tie!") 
            break
        print(f"the current state is state={new_state}")    
        action = random.choice(env.validinputs)
        print(f"Player O has chosen action={action}")    
        new_new_state, reward, done, info = env.step(action)
        env.render()
        step += 1      
        ts.getcanvas().postscript(file=f"files/ch12/ttt_step{step}.ps")
        if done:
            print(f"Player O has won!") 
            break
        else: 
            # play next round
            state=new_new_state
    env.close()    
    return history

def record_ttt():
    while True:
        history=record()
        # We want Player X to win in 3 steps
        if len(history)==3:
            break
    return history


import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import pickle
import os
from PIL import Image
import numpy as np
import imageio

def gen_images():
    # load up the history data
    history = pickle.load(open('files/ch12/ttt_game_history.p', 'rb'))
    # get the best moves 
    bests=[]
    for h in history:
        p=h[1]
        best=max(p,key=p.get)
        bests.append(best)
    

    
    # Generate pictures
    for stage in range(3):
        fig = plt.figure(figsize=(10,10), dpi=200)
        ax = fig.add_subplot(111) 
        ax.set_xlim(0,8)
        ax.set_ylim(-4.5, 3.5)
        #plt.grid()
        plt.axis("off")
        plt.savefig(f"files/ch12/ttt_stage{stage*2}step1.png") 
        xys = [[(4,-4.1),(2,0)],
           [(4,-3.2),(2,0)],           
           [(4,-2.3),(2,0)],           
           [(4,-1.4),(2,0)],
           [(4,-0.5),(2,0)],
           [(4,0.4),(2,0)],
           [(4,1.3),(2,0)],
           [(4,2.2),(2,0)],
           [(4,3.1),(2,0)]]
        for xy in xys:
            ax.annotate("",xy=xy[0],xytext=xy[1],
            arrowprops=dict(arrowstyle = '->', color = 'g', linewidth = 2))  
        # add rectangle to plot
        ax.add_patch(Rectangle((0,-0.6), 2, 1.3,
                         facecolor = 'b',alpha=0.1))
        plt.text(0.2,-0.5,"Deep\nNeural\nNetwork",fontsize=20)        
        for m in range(9):
            plt.text(4.1, 3.1-0.9*m, f"Cell {m+1}, Pr(win)={history[stage][1].get(int(m+1),0):.4f}", fontsize=20, color="r")  
       
    
        plt.savefig(f"files/ch12/ttt_stage{stage*2}step2.png") 
        
        # highlight the best action
        ax.add_patch(Rectangle((4,3.85-bests[stage]*0.9),
                               3.5, 0.5,facecolor = 'b',alpha=0.5))     
        plt.savefig(f"files/ch12/ttt_stage{stage*2}step3.png")     
        plt.close(fig)
        
    
    


def combine_animation():
    for i in range(6):
        im = Image.open(f"files/ch12/ttt_step{i}.ps")
        fig, ax=plt.subplots(figsize=(10,10), dpi=200)
        newax = fig.add_axes([0,0,1,1])
        newax.imshow(im)
        newax.axis('off')
        ax.set_xlim(-5,5)
        ax.set_ylim(-5,5)
        plt.axis("off")
        #plt.grid()
        plt.savefig(f"files/ch12/ttt_step{i}plt.png")
        plt.close(fig)
    
    frames=[]
    
    for stage in [0, 2, 4]:
        for step in [1,2,3]:
            im = Image.open(f"files/ch12/ttt_step{stage}plt.png")
            f0=np.asarray(im)
            im = Image.open(f"files/ch12/ttt_stage{stage}step{step}.png")
            f1=np.asarray(im)
            fs = np.concatenate([f0,f1],axis=1)
            frames.append(fs)
            if step==0:
                frames.append(fs)            
    im = Image.open("files/ch12/ttt_step5plt.png")
    f0=np.asarray(im)
    im = Image.open("files/ch12/ttt_stage4step1.png")
    f1=np.asarray(im)
    fs = np.concatenate([f0,f1],axis=1)
    frames.append(fs)
    frames.append(fs)


    
    imageio.mimsave('files/ch12/ttt_DL_steps.gif', frames, fps=2)
    return frames












