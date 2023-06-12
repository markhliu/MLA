
import numpy as np

from matplotlib import pyplot as plt

def combine_frames(frames,random_frames):
    # Create a list to store the combined frames
    combined = []
    for step in range(0,200):    
        fig = plt.figure(figsize=(24,10), dpi=100)
        ax = fig.add_subplot(111) 
        ax.set_xlim(-10, 10)
        ax.set_ylim(-5,5)
        #plt.grid()
        plt.axis("off")
        ax.text(-9,4,f"The Cart Pole Game with Random Moves, Step {step+1}",fontsize=20)
        ax.text(1,4,f"The Cart Pole Game with Deep Learning, Step {step+1}",fontsize=20)
        # Add the frame from the random-move game to the left
        newax = fig.add_axes([0.05, 0.2, 0.55, 0.55])
        newax.imshow(random_frames[step])
        newax.axis('off')    
    
        # Add the frame from the deep-learning game to the right   
        newax2 = fig.add_axes([0.425, 0.2, 0.55, 0.55])
        newax2.imshow(frames[step])
        newax2.axis('off')
    
        # plot the picture 
        fig.canvas.draw()
        # Now we can save it to a numpy array.
        twoframes = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        twoframes = twoframes.reshape(fig.canvas.get_width_height()[::-1]+ (3,))
        combined.append(twoframes)
        plt.close(fig)
    return combined



def gen_subplots(random_frames):
    # Create a list to store the combined frames
    combined = []
    for step in [0,29,59,89,119,149,175,199]:    
        fig = plt.figure(figsize=(12,10), dpi=200)
        ax = fig.add_subplot(111) 
        ax.set_xlim(-5,5)
        ax.set_ylim(-5,5)
        #plt.grid()
        plt.axis("off")
        ax.text(-4.5,4.5,f"The Cart Pole Game with Random Moves, Step {step+1}",fontsize=20)
        # Add the frame from the random-move game to the left
        newax = fig.add_axes([0.1,0.1,0.8,0.8])
        newax.imshow(random_frames[step])
        newax.axis('off')    
    
        # plot the picture 
        fig.canvas.draw()
        # Now we can save it to a numpy array.
        twoframes = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        twoframes = twoframes.reshape(fig.canvas.get_width_height()[::-1]+ (3,))
        combined.append(twoframes)
        plt.close(fig)
    return combined

def subplots(frames):
    # Create a list to store the combined frames
    combined = []
    for step in [0,29,59,89,119,149,175,199]:    
        fig = plt.figure(figsize=(12,10), dpi=200)
        ax = fig.add_subplot(111) 
        ax.set_xlim(-5,5)
        ax.set_ylim(-5,5)
        #plt.grid()
        plt.axis("off")
        ax.text(-4.5,4.5,f"The Cart Pole Game with Deep Learning, Step {step+1}",fontsize=20)
        # Add the frame from the random-move game to the left
        newax = fig.add_axes([0.1,0.1,0.8,0.8])
        newax.imshow(frames[step])
        newax.axis('off')    
    
        # plot the picture 
        fig.canvas.draw()
        # Now we can save it to a numpy array.
        twoframes = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        twoframes = twoframes.reshape(fig.canvas.get_width_height()[::-1]+ (3,))
        combined.append(twoframes)
        plt.close(fig)
    return combined






