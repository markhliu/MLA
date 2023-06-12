import numpy as np
import pickle
import gym
import random
import matplotlib.pyplot as plt

def dif_breakout():
    env = gym.make("Breakout-v0")
    
    frames=[[],[],[]]
    
    for i in [20,25,30,35]:
        env.reset()
        for _ in range(i):
            action = np.random.choice([0,1,2,3])
            obs, reward, done, info = env.step(action)
            action = np.random.choice([0,1,2,3])
            next_obs, reward, done, info = env.step(action)
        
        obs = prepro(obs).reshape(80,80)
        next_obs = prepro(next_obs).reshape(80,80)
        dif = next_obs - obs
        frames[0].append(obs)
        frames[1].append(next_obs)        
        frames[2].append(dif)
        
    subplots=[]
    for frame in frames:
        for p in frame:
            subplots.append(p)
    titles=["time step t"]*4+["time step t+1"]*4+["difference"]*4
    
    plt.figure(figsize=(10,9),dpi=100)
    for i in range(12):
        plt.subplot(3,4,i+1)
        plt.imshow(subplots[i])
        plt.axis('off')
        plt.title(titles[i], fontsize=16)
    plt.subplots_adjust(bottom=0.001,right=0.999,top=0.999,
    left=0.001, hspace=-0.1,wspace=0.1)
    plt.savefig("files/ch19/dif_breakout.jpg")






H = 200 
learning_rate = 1e-4
gamma = 0.99 
decay_rate = 0.99 
D = 80 * 80 
num_actions = 4


# Define the one-hot encoder and softmax functions
def onehot_encoder(action):
    onehot=np.zeros((1,num_actions))
    onehot[0,action]=1
    return onehot

def softmax(x):
    xi=np.exp(x)
    return xi/xi.sum() 

def prepro(I):
    I = I[35:195] 
    I = I[::2,::2,0] 
    I[I == 144] = 0 
    I[I == 109] = 0 
    I[I != 0] = 1 
    return I.astype(float).ravel()


def policy_forward(model,x):
    h = np.dot(model['W1'], x)
    h[h<0] = 0 
    logp = np.dot(model['W2'], h)
    p = softmax(logp)
    return p, h 



def policy_backward(model, eph, epdlogp, epx):
    dW2 = np.dot(epdlogp.T, eph)
    dh = np.dot(epdlogp, model['W2'])
    dh[eph <= 0] = 0 
    dW1 = np.dot(dh.T, epx)
    return {'W1':dW1, 'W2':dW2}

def discount_rewards(r):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, len(r))):
        if r[t] != 0: 
            running_add = 0 
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def training(env,model,render):
    observation = env.reset()
    prev_x = None 
    xs,hs,dlogps,drs = [],[],[],[]
    reward_sum = 0
    lives=5
    while True:
        if render:
            env.render()
        cur_x = prepro(observation)
        x = cur_x - prev_x if prev_x is not None else np.zeros(D)
        prev_x = cur_x
        aprob, h = policy_forward(model,x)
        action=np.random.choice([0,1,2,3], p=aprob)
        xs.append(x) 
        hs.append(h) 
        y = onehot_encoder(action)
        # gradient for adjust weights, magic part
        dlogps.append(y - aprob)
        observation, reward, done, info = env.step(action)
        # restructuring rewards
        if (lives-info["ale.lives"])==1:
            lives -= 1
            reward=-1        
        reward_sum += reward
        drs.append(reward) 
        if done: 
            epx = np.vstack(xs)
            eph = np.vstack(hs)
            epdlogp = np.vstack(dlogps)
            epr = np.vstack(drs)
            discounted_epr = discount_rewards(epr)
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)
            epdlogp *= discounted_epr 
            grad = policy_backward(model,eph, epdlogp, epx)
            break
    return grad, reward_sum         

from collections import deque


def create_batch(env,render,model,batch_size):
    batchrewards=[]
    grad_buffer={k:np.zeros_like(v) for k,v in model.items()} 
    rmsprop_cache = {k:np.zeros_like(v) for k,v in model.items()}      
    for i in range(batch_size):
        grad,reward_sum = training(env,model,render)
        for k in model: 
            grad_buffer[k] += grad[k]
        batchrewards.append(reward_sum)
    for k,v in model.items():
        g = grad_buffer[k] 
        rmsprop_cache[k] = decay_rate * rmsprop_cache[k]\
            + (1 - decay_rate) * g**2
        model[k] += learning_rate * g
    return batchrewards


def policy_breakout(test=False,resume=False,render=False,\
                    cutoff=25):
    rewards=deque(maxlen=100)
    env = gym.make("Breakout-v0")
    if test:
        batch_size=1
    else:
        batch_size=10
    if resume==True or test==True:
        model=pickle.load(open('files/ch19/pg_breakout.p','rb'))
    else:
        model={}
        model['W1']=np.random.randn(H,D) / np.sqrt(D) 
        model['W2']=np.random.randn(num_actions,H) / np.sqrt(H) 
    episode_count=0
    while True:
        batchrewards=create_batch(env,render,model,batch_size)
        rewards += batchrewards
        episode_count += batch_size
        running_reward=np.mean(np.array(rewards)) 
        if episode_count % 100 == 0 or test==True:
            template = "running reward: {:.6f} at episode {}"
            print(template.format(running_reward,episode_count))
            pickle.dump(model,\
                        open('files/ch19/pg_breakout.p','wb'))
        if test==False and running_reward>=cutoff:  
            break
        if test==True and episode_count>=3:  
            break         
    env.close()    











