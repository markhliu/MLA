from utils.conn_env import conn
import random
from copy import deepcopy
import numpy as np
import tensorflow as tf
import turtle as t

model_conn = tf.keras.models.load_model('files/ch13/trained_conn.h5')

def best_move(board, valids, occupied):
    # if there is only one valid move, take it
    if len(valids)==1:
        return valids[0]
    # Set the initial value of bestoutcome        
    bestoutcome = -2;
    bestmove=None 
    # record winning probabilities for all hypothetical moves
    p_wins={}    
    #go through all possible moves hypothetically to predict outcome
    for col in valids:
        tooccupy=deepcopy(board)
        row = 1+len(occupied[col-1])
        tooccupy[col-1][row-1]=1
        prediction=model_conn.predict(np.array(tooccupy).reshape((-1, 7, 6, 1)),verbose=0)
        p_win=prediction[0][1]
        p_wins[col]=p_win
        if p_win>bestoutcome:
            # Update the bestoutcome
            bestoutcome = p_win
            # Update the best move
            bestmove = col
    return bestmove, p_wins

def gen_ps():
    # Initiate the game environment
    env=conn()
    state=env.reset()  
    step=0
    try:
        ts=t.getscreen() 
    except t.Terminator:
        ts=t.getscreen()
    t.hideturtle()
    env.render()
    ts.getcanvas().postscript(file=f"files/ch13/conn_step{step}.ps")
    
    # Create a list to record game history
    history=[]
    
    
    
    # Play a full game 
    while True: 
        bestmove,p_wins=best_move(state, env.validinputs, env.occupied)
        action=bestmove 
        old_state=deepcopy(state)
        new_state, reward, done, info = env.step(action)
        history.append([old_state,p_wins,action,deepcopy(new_state),done])
        step += 1  
        env.render()
        ts.getcanvas().postscript(file=f"files/ch13/conn_step{step}.ps")
        if done:
            break  
        # yellow makes random move
        if len(history)<=1:
            # choose 4 in the first move to make game more interesting
            action=4
        else:
            action = random.choice(env.validinputs)  
        new_new_state, reward, done, info = env.step(action)
        step += 1   
        env.render()
        ts.getcanvas().postscript(file=f"files/ch13/conn_step{step}.ps")
        if done:
            break
        else: 
            # play next round
            state=new_new_state
    env.close() 
    return history
    
def record_conn():
    while True:
        history=gen_ps()
        # We want Player red to win in 4 steps
        if len(history)==4:
            break
    return history    
    
    
    

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import pickle
import os
from PIL import Image
import numpy as np


def stage_pics():
    # load up the history data
    history= pickle.load(open('files/ch13/conn_game_history.p','rb'))
    # remember the best moves 
    bests = []
    for item in history:
        bests.append(item[2])
    # Generate pictures
    for stage in range(len(history)):
        fig = plt.figure(figsize=(10,6), dpi=200)
        ax = fig.add_subplot(111) 
        ax.set_xlim(0,10)
        ax.set_ylim(-2.5, 3.5)
        #plt.grid()
        plt.axis("off")
        plt.savefig(f"files/ch13/conn_stage{stage*2}step1.png") 
        xys = [[(4,-2.3),(2,0)],           
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
        for m in range(7):
            plt.text(4.1, 3.1-0.9*m, f"Column {m+1}, \
        Pr(win)={history[stage][1].get(m+1,0):.4f}", fontsize=20, color="r")  
       
    
        plt.savefig(f"files/ch13/conn_stage{stage*2}step2.png") 
        
        # highlight the best action
        ax.add_patch(Rectangle((4,3.85-bests[stage]*0.9),
                               6, 0.5,facecolor = 'b',alpha=0.5))     
        plt.savefig(f"files/ch13/conn_stage{stage*2}step3.png")     
        plt.close(fig)   
    
    

import imageio
   
def DL_steps():
    for i in range(8):
        im = Image.open(f"files/ch13/conn_step{i}.ps")
        fig, ax=plt.subplots(figsize=(6,6), dpi=200)
        newax = fig.add_axes([0,0,1,1])
        newax.imshow(im)
        newax.axis('off')
        ax.set_xlim(-3,3)
        ax.set_ylim(-3,3)
        plt.axis("off")
        #plt.grid()
        plt.savefig(f"files/ch13/conn_step{i}plt.png")
        plt.close(fig)
    
    frames=[]
    
    for stage in range(4):
        for step in [1,2,3]:
            im = Image.open(f"files/ch13/conn_step{stage*2}plt.png")
            f0=np.asarray(im)
            im = Image.open(f"files/ch13/conn_stage{stage*2}step{step}.png")
            f1=np.asarray(im)
            fs = np.concatenate([f0,f1],axis=1)
            frames.append(fs)
            if step==0:
                frames.append(fs)            
    im = Image.open(f"files/ch13/conn_step{4}plt.png")
    f0=np.asarray(im)
    im = Image.open("files/ch13/conn_stage0step1.png")
    f1=np.asarray(im)
    fs = np.concatenate([f0,f1],axis=1)
    frames.append(fs)
    frames.append(fs)
    
    imageio.mimsave('files/ch13/conn_DL_steps.gif', frames, fps=2)
    return frames










