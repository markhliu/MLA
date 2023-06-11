import numpy as np
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
import tensorflow as tf
import matplotlib.pyplot as plt

#reload = tf.keras.models.load_model("files/ch20/DoubleQ_Breakout.h5")
#reload1=tf.keras.models.load_model("Seaquest.h5")

# Use the Baseline Atari environment
env1 = make_atari("SeaquestNoFrameskip-v4")
# Process and stack the frames
env1 = wrap_deepmind(env1, frame_stack=True, scale=True)

titles=["time step t","time step t+1","time step t+2","time step t+3"]*3

def seaquest_pixels():
    frames = []
    obs = env1.reset()
    history = []
    while True:
        action = env1.action_space.sample()
        obs, reward, done, info = env1.step(action)
        history.append(info)
        if len(history)>1:
            # Capture when the agent loses a life
            if info["ale.lives"]<history[-2]["ale.lives"]:
                for j in range(4):
                    frames.append(obs[:,:,j])
            # Stop if we have 3 example (12 frames)        
            if len(frames)>=12:
                break
        if done:
            obs = env1.reset()
    # Create a subplot
    plt.figure(figsize=(10,9),dpi=100)
    for i in range(12):
        plt.subplot(3,4,i+1)
        plt.imshow(frames[i])
        plt.axis('off')
        plt.title(titles[i], fontsize=16)
    plt.subplots_adjust(bottom=0.001,right=0.999,top=0.999,
    left=0.001, hspace=-0.1,wspace=0.1)
    plt.savefig("files/ch22/sequest_pixels.jpg")


# Use the Baseline Atari environment
env2 = make_atari("BeamRiderNoFrameskip-v4")
# Process and stack the frames
env2 = wrap_deepmind(env2, frame_stack=True, scale=True)
def beamrider_pixels():
    frames = []
    obs = env2.reset()
    history = []
    while True:
        action = env2.action_space.sample()
        obs, reward, done, info = env2.step(action)
        history.append(info)
        if len(history)>1:
            # Capture when the agent loses a life
            if info["ale.lives"]<history[-2]["ale.lives"]:
                for j in range(4):
                    frames.append(obs[:,:,j])
            # Stop if we have 3 example (12 frames)        
            if len(frames)>=12:
                break
        if done:
            obs = env2.reset()
    # Create a subplot
    plt.figure(figsize=(10,9),dpi=100)
    for i in range(12):
        plt.subplot(3,4,i+1)
        plt.imshow(frames[i])
        plt.axis('off')
        plt.title(titles[i], fontsize=16)
    plt.subplots_adjust(bottom=0.001,right=0.999,top=0.999,
    left=0.001, hspace=-0.1,wspace=0.1)
    plt.savefig("files/ch22/beamrider_pixels.jpg")














    


from tensorflow import keras
from tensorflow.keras import layers, models 

# Input and output shape
input_shape = (84, 84, 4,)
def create_model(num_actions):
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
def update_Q(num_actions):
    global dnn,target_dnn
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
    
    
def play_episode(num_actions,name):
    global frame_count,env,dnn,target_dnn
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
            update_Q(num_actions)
        if frame_count % update_target_network == 0:
            # update the the target network with new weights
            target_dnn.set_weights(dnn.get_weights())
            # Periodically save the model
            dnn.save(f"files/ch22/{name}.h5")         
        if done:
            running_rewards.append(episode_reward)
            break
  
def train_atari(name):
    global frame_count,env,num_actions,dnn,target_dnn
    # Use the Baseline Atari environment
    env = make_atari(f"{name}NoFrameskip-v4")
    # Process and stack the frames
    env = wrap_deepmind(env, frame_stack=True, scale=True)
    num_actions = env.action_space.n
    
    # Network for training
    dnn=create_model(num_actions)
    # Network for predicting (target network)
    target_dnn=create_model(num_actions)     
    episode=0
    frame_count=0
    while True: 
        episode += 1
        play_episode(num_actions,name)
        running_reward = np.mean(np.array(running_rewards))
        if episode%20==0:
            # Log details
            m="running reward: {:.2f} at episode {} and frame {}"
            print(m.format(running_reward,episode,frame_count))
        if running_reward>20:
            dnn.save(f"files/ch22/{name}.h5")
            print(f"solved at episode {episode}")
            break
  
    
  
    
def test_atari(name):
    reload = tf.keras.models.load_model(f"files/ch22/{name}.h5")
    env = make_atari(f"{name}NoFrameskip-v4")
    env = wrap_deepmind(env, frame_stack=True, scale=True)
    scores = []
    num_actions = env.action_space.n
    for i in range(100):
        state = env.reset()
        score = 0
        for j in range(10000):
            if np.random.rand(1)[0]<0.01:
                action = np.random.choice(num_actions)
            else:
                state_tensor = tf.convert_to_tensor(state)
                state_tensor = tf.expand_dims(state_tensor, 0)
                action_Qs = reload(state_tensor, training=False)
                action = tf.argmax(action_Qs[0]).numpy()    
            state, reward, done, info = env.step(action)
            score += reward
            if done:
                print(f"the score in episode {i+1} is {score}")
                scores.append(score)
                break
    env.close()
    print(f"the average score is {np.array(scores).mean()}")  
        
  
 
  
 
    
 
    