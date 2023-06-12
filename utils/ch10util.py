import tensorflow as tf
import numpy as np
import gym

env=gym.make('FrozenLake-v0',is_slippery=False)
reload = tf.keras.models.load_model("files/ch10/trained_frozen.h5")

# Define a onehot_encoder() function
def onehot_encoder(value, length):
    onehot=np.zeros((1,length))
    onehot[0,value]=1
    return onehot

action0=onehot_encoder(0, 4)
action1=onehot_encoder(1, 4)
action2=onehot_encoder(2, 4)
action3=onehot_encoder(3, 4)

def test_one_game():
    state=env.reset()
    winlose=0
    while True:
        # Convert state and action into onehots 
        state_arr = onehot_encoder(state, 16)
        # Use the trained model to predict the prob of winning 
        sa0 = np.concatenate([state_arr, action0], axis=1)    
        sa1 = np.concatenate([state_arr, action1], axis=1)  
        sa2 = np.concatenate([state_arr, action2], axis=1)  
        sa3 = np.concatenate([state_arr, action3], axis=1)
        sa = np.concatenate([sa0, sa1, sa2, sa3], axis=0)
        action = np.argmax(reload.predict(sa, verbose=0))
        new_state, reward, done, info = env.step(action)
        state = new_state
        if done == True:
            # change winlose to 1 if the last state is 15
            if state==15:
                winlose=1
            break
    return winlose



from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import pickle
import os

def frozen_lake_steps():
    # Load up date during the training process
    preds = pickle.load(open('files/ch10/frozen_predictions.p', 'rb'))
    
    states = [(-6,2),(-6,0),(-6,-2),(-5,-2),(-5,-4),(-4,-4),(-3,-4)]
    actions = [1,1,2,1,2,2]
    
    grid = [["S", "F", "F", "F"],
            ["F", "H", "F", "H"],
            ["F", "F", "F", "H"],
            ["H", "F", "F", "G"]]
    
    hs = [3.1,1.3,-1.4,-3.2]
    
    # Generate six pictures
    for stage in range(7):
    
        fig = plt.figure(figsize=(14,10), dpi=100)
        ax = fig.add_subplot(111) 
        ax.set_xlim(-7, 7)
        ax.set_ylim(-5, 5)
        #plt.grid()
        plt.axis("off")
    
        # table grid
        for x in range(-6,-1,1):
            ax.plot([x,x],[-4,4],color='gray',linewidth=3)
        for y in range(-4,5,2):
            ax.plot([-6,-2],[y,y],color='gray',linewidth=3)
        for row in range(4):
            for col in range(4):
                plt.text(col-5.8,2.6-2*row,grid[row][col],fontsize=60) 
    
        # highlight current state
        ax.add_patch(Rectangle(states[stage], 1,2,facecolor='r',alpha=0.8))             
        plt.savefig(f"files/ch10/frozen_stage{stage}step1.png")        
                
        if stage<=5:
            # reload trained model
            ps = preds[stage].reshape(4,)
            # Draw connections between neurons
            xys = [[(0,-3.2),(-2,0)],                   
                   [(0,-1.4),(-2,0)],
                   [(0,1.3),(-2,0)],
                   [(0,3.1),(-2,0)]]
            for xy in xys:
                ax.annotate("",xy=xy[0],xytext=xy[1],
                arrowprops=dict(arrowstyle = '->',
                                color = 'g', linewidth = 2))  
            # Put explanation texts on the graph
            plt.text(-1.5,1.25,"left",fontsize=20,color='g',rotation=55) 
            plt.text(-1.25,0.25,"down",fontsize=20,color='g',rotation=35) 
            plt.text(-1.25,-0.85,"right",fontsize=20,color='g',rotation=-35)
            plt.text(-1.5,-1.8,"up",fontsize=20,color='g',rotation=-55)     
            # add rectangle to plot
            for i in range(4):
                ax.add_patch(Rectangle((0,-0.6+hs[i]), 2, 1.3,
                             facecolor = 'b',alpha=0.1)) 
                plt.text(0.2,hs[i]-0.5,"Deep\nNeural\nNetwork",
                         fontsize=20) 
                plt.text(2.6, hs[i]-0.15, f"Prob(win)={ps[i]:.4f}",
                         fontsize=25, color="r")  
                ax.annotate("",xy=(2.5,hs[i]),xytext=(2,hs[i]),
                arrowprops=dict(arrowstyle = '->', 
                                color = 'g', linewidth = 2))   
    
            plt.savefig(f"files/ch10/frozen_stage{stage}step2.png") 
            
            # highlight the best action
            ax.add_patch(Rectangle((2.5,hs[actions[stage]]-0.4), 4.25, 1,
                         facecolor = 'b',alpha=0.5))     
            plt.savefig(f"files/ch10/frozen_stage{stage}step3.png")     
        plt.close(fig)
    











