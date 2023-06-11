import numpy as np
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
import tensorflow as tf
import matplotlib.pyplot as plt



# Use the Baseline Atari environment
env = make_atari("SpaceInvadersNoFrameskip-v4")
# Process and stack the frames
env = wrap_deepmind(env, frame_stack=True, scale=True)

titles=["time step t","time step t+1","time step t+2","time step t+3"]*3

def invaders_windows():
    frames=[]
    obs = env.reset()
    for i in range(500):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        if i>1 and i%100==0:
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
    plt.savefig("files/ch21/invaders_windows.jpg")















    
    
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models 

# Input and output shape
input_shape = (84, 84, 4,)
num_actions = env.action_space.n
def create_model():
    model=keras.models.Sequential()
    model.add(keras.layers.Conv2D(filters=32,kernel_size=8,
     strides=(4,4),activation="relu",input_shape=input_shape))
    model.add(keras.layers.Conv2D(filters=64,kernel_size=4,
     strides=(2,2),activation="relu"))
    model.add(keras.layers.Conv2D(filters=64,kernel_size=3,
     strides=(1,1),activation="relu"))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512,activation="relu"))
    model.add(keras.layers.Dense(num_actions))
    return model    
    
  
# Network for training
dnn=create_model()
# Network for predicting (target network)
target_dnn=create_model()    
  
lr=0.00025
optimizer = keras.optimizers.Adam(learning_rate=lr,clipnorm=1)
loss_function = keras.losses.Huber()    
  
    
import random
from collections import deque

# Discount factor for past rewards
gamma = 0.99 
# batch size
batch_size = 32  
# Create a replay buffer 
memory=deque(maxlen=50000)
# Create a running rewards list 
running_rewards=deque(maxlen=100)  
    
# Generate a batch
def gen_batch():
    # select a batch from the buffer memory
    samples = random.sample(memory,batch_size)
    dones = []
    frames = []
    new_frames = []
    rewards = []
    actions = []
    for sample in samples:
        frame, new_frame, action, reward, done = sample
        frames.append(frame)
        new_frames.append(new_frame)
        actions.append(action)  
        dones.append(done)
        rewards.append(reward)
    frames=np.array(frames)
    new_frames=np.array(new_frames)
    dones=tf.convert_to_tensor(dones)
    return dones,frames,new_frames,rewards,actions  
    
  
# Replay and update model parameters
def update_Q():
    dones,frames,new_frames,rewards,actions=gen_batch()
    # update the Q table
    preds = target_dnn.predict(new_frames, verbose=0)
    Qs = rewards + gamma * tf.reduce_max(preds, axis=1)
    # if done=1  reset Q to  -1; important
    new_Qs = Qs * (1 - dones) - dones
    # update model parameters
    onehot = tf.one_hot(actions, num_actions)
    with tf.GradientTape() as t:
        Q_preds=dnn(frames)
        # Calculate old Qs for the action taken
        old_Qs=tf.reduce_sum(tf.multiply(Q_preds,onehot),axis=1)
        # Calculate loss between new Qs and old Qs
        loss=loss_function(new_Qs, old_Qs)
    # Update using backpropagation
    gs=t.gradient(loss,dnn.trainable_variables)
    optimizer.apply_gradients(zip(gs,dnn.trainable_variables))  
    
    
    
# Let the game begin
running_reward = 0
frame_count = 0
# Number of frames to take random actions
epsilon_random_frames = 50000
# Train the model after 4 actions
update_after_actions = 4
# How often to update the target network
update_target_network = 10000    
    
    
def play_episode():
    global frame_count
    # reset state and episode reward before each episode
    state = np.array(env.reset())
    episode_reward = 0    
    # Allow 10,000 steps per episode
    for timestep in range(1, 10001):
        frame_count += 1
        # Calculate current epsilon based on frame count
        epsilon = max(0.1, 1 - frame_count * (1-0.1) /1000000)
        # Use epsilon-greedy for exploration
        if frame_count < epsilon_random_frames or \
            epsilon > np.random.rand(1)[0]:
            # Take random action
            action = np.random.choice(num_actions)
        # Use exploitation
        else:
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = dnn(state_tensor, training=False)
            action = tf.argmax(action_probs[0]).numpy()
        # Apply the sampled action in our environment
        state_next, reward, done, _ = env.step(action)
        state_next = np.array(state_next)
        episode_reward += reward
        # Change done to 1.0 or 0.0 to prevent error
        if done==True:
            done=1.0
        else:
            done=0.0
        # Save actions and states in replay buffer
        memory.append([state, state_next, action, reward, done])
        # current state becomes the next state in next round
        state = state_next
        # Update Q once batch size is over 32
        if len(memory) > batch_size and \
            frame_count % update_after_actions == 0:
            update_Q()
        if frame_count % update_target_network == 0:
            # update the the target network with new weights
            target_dnn.set_weights(dnn.get_weights())
            # Periodically save the model
            dnn.save("files/ch21/DoubleQ_Invaders.h5")         
        if done:
            running_rewards.append(episode_reward)
            break
  
def train_invaders():
    global frame_count    
    episode=0
    frame_count=0
    while True: 
        episode += 1
        play_episode()
        running_reward = np.mean(np.array(running_rewards))
        if episode%20==0:
            # Log details
            m="running reward: {:.2f} at episode {} and frame {}"
            print(m.format(running_reward,episode,frame_count))
        if running_reward>40:
            dnn.save("files/ch21/DoubleQ_Invaders.h5")
            print(f"solved at episode {episode}")
            break
  
    
  
    
  
    
  
    



def invaders_episode():

    reload=tf.keras.models.load_model("files/ch21/DoubleQ_Invaders.h5")
    state=env.reset()
    
    for i in range(3):
        for j in range(10000):
            if np.random.rand(1)[0]<0.01:
                action=np.random.choice(num_actions)
            else:
                state_tensor=tf.convert_to_tensor(state)
                state_tensor=tf.expand_dims(state_tensor,0)
                action_Qs=reload(state_tensor,training=False)
                action=tf.argmax(action_Qs[0]).numpy()    
            state, reward, done, info = env.step(action)
            env.render()
            if done:
                break
    env.close()









def test_invaders():
    
    reload=tf.keras.models.load_model("files/ch21/DoubleQ_Invaders.h5")
    scores = []
    state = env.reset()
    for i in range(100):
        
        score = 0
        for j in range(10000):
            if np.random.rand(1)[0]<0.01:
                action = np.random.choice(num_actions)
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
                if info['ale.lives']==0:
                    state = env.reset()
                break
    env.close()
    
    print(f"the average score is {np.array(scores).mean()}")
    
      
    
  
import pickle, imageio
 
def collect_invaders():
    
    reload=tf.keras.models.load_model("files/ch21/DoubleQ_Invaders.h5")
    frames = []
    state = env.reset()
    for i in range(3):
        episode_frames = []
        for j in range(10000):
            if np.random.rand(1)[0]<0.01:
                action = np.random.choice(4)
            else:
                state_tensor = tf.convert_to_tensor(state)
                state_tensor = tf.expand_dims(state_tensor,0)
                action_probs = reload(state_tensor,
                                      training=False)
                action = tf.argmax(action_probs[0]).numpy()    
            obs, reward, done, info = env.step(action)
            state=obs
            episode_frames.append(env.render(mode='rgb_array'))
            if done:
                frames.append(episode_frames)
                imageio.mimsave(f"files/ch21/episode{i+1}.gif", 
                                episode_frames[::5], fps=240)
                break
    env.close() 
    pickle.dump(frames, open(f'files/ch21/invaders.p', 'wb'))    
  
    
  
    
  
    
  