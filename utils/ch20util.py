import numpy as np
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
import tensorflow as tf
import matplotlib.pyplot as plt


reload=tf.keras.models.load_model("files/ch20/DoubleQ_Breakout.h5")

# Use the Baseline Atari environment
env = make_atari("BreakoutNoFrameskip-v4")
# Process and stack the frames
env = wrap_deepmind(env, frame_stack=True, scale=True)

titles=["time step t","time step t+1","time step t+2","time step t+3"]*3

def four_frames():
    frames=[]
    obs = env.reset()
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        if i>1 and i%25==0:
            for j in range(4):
                frames.append(obs[:,:,j])
        if done:
            obs = env.reset()
    # Create a subplot
    plt.figure(figsize=(10,9),dpi=100)
    for i in range(12):
        plt.subplot(3,4,i+1)
        plt.imshow(frames[i])
        plt.axis('off')
        plt.title(titles[i], fontsize=16)
    plt.subplots_adjust(bottom=0.001,right=0.999,top=0.999,
    left=0.001, hspace=-0.1,wspace=0.1)
    plt.savefig("files/ch20/four_frames.jpg")














def test_breakout():
    scores = []
    for i in range(100):
        state = env.reset()
        score = 0
        for j in range(10000):
            if np.random.rand(1)[0]<0.01:
                action = np.random.choice(4)
            else:
                state_tensor = tf.convert_to_tensor(state)
                state_tensor = tf.expand_dims(state_tensor, 0)
                action_probs = reload(state_tensor, training=False)
                # Take the best action
                action = tf.argmax(action_probs[0]).numpy()
            state, reward, done, info = env.step(action)
            score += reward
            if done:
                print(f"the score in episode {i+1} is {score}")
                scores.append(score)
                break
    env.close()
    
    print(f"the average score is {np.array(scores).mean()}")
    
    

import pickle   
 
def collect_episode():
    reward_sum = 0
    while True:
        state = env.reset()
        episode_frames = []
        for j in range(10000):
            if np.random.rand(1)[0]<0.01:
                action = np.random.choice(4)
            else:
                state_tensor = tf.convert_to_tensor(state)
                state_tensor = tf.expand_dims(state_tensor, 0)
                action_probs = reload(state_tensor, training=False)
                # Take the best action
                action = tf.argmax(action_probs[0]).numpy()    
            state, reward, done, info = env.step(action)
            reward_sum += reward
            episode_frames.append(env.render(mode='rgb_array'))
            if done:
                if reward_sum>125:
                    frames=episode_frames
                break
            env.close()
        if reward_sum>200:
            break
       
    pickle.dump(frames, open('files/ch20/breakout_frames.p', 'wb'))    
    
    

    
    
