#!/usr/bin/env python
# coding: utf-8

# # Using Reinforcement Learning for Generator Optimization
# 
# 
# ### 1. Import the Necessary Packages

# In[50]:


import sys
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


# Set plotting options

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')
np.set_printoptions(precision=3, linewidth=120)


# ### 2. Specify the Environment, and Explore the State and Action Spaces
# 
# We'll use [OpenAI Gym](https://gym.openai.com/) environments to test and develop our algorithms. These simulate a variety of classic as well as contemporary reinforcement learning tasks.  Let's use an environment that has a continuous state space, but a discrete action space.

# Create the PowerPlant environment

# In[2]:


env = gym.make('gym_powerplant:powerplant_complex-v0')


# Run the next code cell to watch a random agent.

# In[3]:


state = env.reset()
score = 0

for t in range(3000):
    action = env.action_space.sample()
    env.render()
    state, reward, done, _ = env.step(action)
    print(state)
    score += reward
    if done:
        break 

print('Final score:', score)

env.close()


# Explore state (observation) space

# In[4]:


print("State space:", env.observation_space)
print("- low:", env.observation_space.low)
print("- high:", env.observation_space.high)


# Generate some samples from the state space 

# In[5]:


print("State space samples:")
sss = np.array([env.observation_space.sample() for i in range(10)])
print(sss)


# Explore the action space

# In[6]:


print("Action space:", env.action_space)


# Generate some samples from the action space

# In[7]:


print("Action space samples:")
ass = np.array([env.action_space.sample() for i in range(10)])
print(ass)


# ### 3. Discretize the State Space with a Uniform Grid

# In[19]:


def create_uniform_grid(low, high, bins=(10, 10, 10, 10, 10, 3, 3)):
    """Define a uniformly-spaced grid that can be used to discretize a space.
    
    Parameters
    ----------
    low : array_like
        Lower bounds for each dimension of the continuous space.
    high : array_like
        Upper bounds for each dimension of the continuous space.
    bins : tuple
        Number of bins along each corresponding dimension.
    
    Returns
    -------
    grid : list of array_like
        A list of arrays containing split points for each dimension.
    """

    dim = len(bins)
    grid = [np.linspace(low[d], high[d], bins[d] + 1)[1:-1] for d in range(dim)]
        
    print("Uniform grid: [<low>, <high>] / <bins> => <splits>")
    
    for l, h, b, splits in zip(low, high, bins, grid):
        print("    [{}, {}] / {} => {}".format(l, h, b, splits))
    
    return grid

low  = [-1.0, -5.0, -5.0, -5.0, -5.0, 0, 0]
high = [ 1.0,  5.0,  5.0,  5.0,  5.0, 3, 3]
create_uniform_grid(low, high)  # test


# In[11]:


def discretize(sample, grid):
    """Discretize a sample as per given grid.
    
    Parameters
    ----------
    sample : array_like
        A single sample from the (original) continuous space.
    grid : list of array_like
        A list of arrays containing split points for each dimension.
    
    Returns
    -------
    discretized_sample : array_like
        A sequence of integers with the same number of dimensions as sample.
    """
    
    # apply along each dimension
    discretized_sample = list(int(np.digitize(s, g)) for s, g in zip(sample, grid))
    
    return discretized_sample  


# Test with a simple grid and some samples

# In[12]:


low  = [-1.0, -5.0, -10, -10, -10, 0, 0]
high = [ 1.0,  5.0,  10,  10,  10, 3, 3]
grid = create_uniform_grid(low, high)

samples = np.array(
    [[-1.00, 5.0,  10.0,  9.4,  9.0, 0.0, 1.0],
     [-0.81, 4.1,   4.0,  5.6,  7.5, 1.0, 2.0],
     [-0.80, 4.0, -10.0, -2.0,  7.8, 1.0, 3.0],
     [-0.50, 0.0,  -9.0, -4.0,  4.4, 1.0, 1.0],
     [ 0.20, 1.9,   9.0,  2.3,  7.0, 1.0, 1.0],
     [ 0.80, 4.0,  -2.0,  6.6,  6.9, 1.0, 0.0],
     [ 0.81, 4.1,   2.0, -1.2,  8.8, 1.0, 2.0],
     [ 1.00, 5.0,  10.0, -9.0, -3.0, 0.0, 3.0]])

discretized_samples = np.array([discretize(sample, grid) for sample in samples])

print("\nSamples:", repr(samples), sep="\n")
print("\nDiscretized samples:", repr(discretized_samples), sep="\n")


# Create a grid to discretize the state space

# In[13]:


low  = env.observation_space.low
high = env.observation_space.high
state_grid = create_uniform_grid(low, high, bins=(10, 3, 10, 10, 10, 3, 3))
state_grid


# ### 5. Q-Learning
# 
# Provided below is a simple Q-Learning agent.

# In[14]:


class QLearningAgent:
    """Q-Learning agent that can act on a continuous state space by discretizing it."""

    def __init__(self, env, state_grid, alpha=0.02, gamma=0.97,
                 epsilon=1.0, epsilon_decay_rate=0.9995, min_epsilon=0.01, seed=505):
        """Initialize variables, create grid for discretization."""
        
        # Environment info
        self.env = env
        self.state_grid = state_grid
        self.state_size = tuple(len(splits) + 1 for splits in self.state_grid)  # n-dim state space
        self.action_size = self.env.action_space.n  # 1-dim discrete action space
        self.seed = np.random.seed(seed)
        print("Environment:", self.env)
        print("State space size:", self.state_size)
        print("Action space size:", self.action_size)
        
        # Learning parameters
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = self.initial_epsilon = epsilon  # initial exploration rate
        self.epsilon_decay_rate = epsilon_decay_rate # how quickly should we decrease epsilon
        self.min_epsilon = min_epsilon
        
        # Create Q-table
        self.q_table = np.zeros(shape=(self.state_size + (self.action_size,)))
        print("Q table size:", self.q_table.shape)

    def preprocess_state(self, state):
        """Map a continuous state to its discretized representation."""
        return tuple(discretize(state, self.state_grid))

    def reset_episode(self, state):
        """Reset variables for a new episode."""
        # Gradually decrease exploration rate
        self.epsilon *= self.epsilon_decay_rate
        self.epsilon = max(self.epsilon, self.min_epsilon)

        # Decide initial action
        self.last_state = self.preprocess_state(state)
        self.last_action = np.argmax(self.q_table[self.last_state])
        return self.last_action
    
    def reset_exploration(self, epsilon=None):
        """Reset exploration rate used when training."""
        self.epsilon = epsilon if epsilon is not None else self.initial_epsilon

    def act(self, state, reward=None, done=None, mode='train'):
        """Pick next action and update internal Q table (when mode != 'test')."""
        state = self.preprocess_state(state)
        
        if mode == 'test':
            # Test mode: Simply produce an action
            action = np.argmax(self.q_table[state])
        else:
            # Train mode (default): Update Q table, pick next action
            # Note: We update the Q table entry for the *last* (state, action) pair 
            # with current state, reward
            self.q_table[self.last_state + (self.last_action,)] += self.alpha * (reward + self.gamma * max(self.q_table[state]) - self.q_table[self.last_state + (self.last_action,)])

            # Exploration vs. exploitation
            do_exploration = np.random.uniform(0, 1) < self.epsilon
            if do_exploration:
                # Pick a random action
                action = np.random.randint(0, self.action_size)
            else:
                # Pick the best action from Q table
                action = np.argmax(self.q_table[state])

        # Roll over current state, action for next step
        self.last_state = state
        self.last_action = action
        return action

q_agent = QLearningAgent(env, state_grid)


# In[15]:


def run(agent, env, num_episodes=10000, mode='train'):
    """Run agent in given reinforcement learning environment and return scores."""
    
    scores = []
    max_avg_score = -np.inf
    
    for i_episode in range(1, num_episodes + 1):
        
        # Initialize episode
        state = env.reset()
        action = agent.reset_episode(state)
        total_reward = 0
        done = False

        # Roll out steps until done
        while not done:
            state, reward, done, info = env.step(action)
            total_reward += reward
            action = agent.act(state, reward, done, mode)

        # Save final score
        scores.append(total_reward)
        
        # Print episode stats
        if mode == 'train':
            if len(scores) > 100:
                avg_score = np.mean(scores[-100:])
                if avg_score > max_avg_score:
                    max_avg_score = avg_score
            if i_episode % 100 == 0:
                print("\rEpisode {}/{} | Max Average Score: {}".format(i_episode, num_episodes, max_avg_score), end="")
                sys.stdout.flush()

    return scores

scores = run(q_agent, env)


# Plot scores obtained per episode

# In[16]:


plt.plot(scores)
plt.title("Generator Optimization")
plt.xlabel("Episode")
plt.ylabel("Rewards")


# In[17]:


def plot_scores(scores, rolling_window=100):
    """Plot scores and optional rolling mean using specified window."""

    rolling_mean = pd.Series(scores).rolling(rolling_window).mean()
    plt.plot(rolling_mean)
    plt.title("Generator Optimization")
    plt.xlabel("Episode")
    plt.ylabel("Rolling Mean Rewards")
    return rolling_mean

rolling_mean = plot_scores(scores)


# ### 7. Watch a Q-Learning Agent

# Define mission time (days)

# In[206]:


days = 365*3
days


# Define plotting functions

# In[207]:


def plot_current(df, xmax):
    ax = plt.gca()

    df.plot(kind='line', y='Target Current Load', ax=ax, xlim=(0, xmax), figsize=(10, 5))
    df.plot(kind='line', y='Available Current Capacity', ax=ax, xlim = (0, xmax), figsize=(10, 5))

    plt.xlabel('Days Since Mission Start')
    plt.ylabel('Current (A)')
    
def plot_gen_run(df, xmax):
    ax2 = plt.gca()
    ax2.ylabel = "Runtime"

    df.plot(kind='line', y='Generator 1 Runtime', ax=ax2, xlim=(0, xmax), figsize=(10, 5))
    df.plot(kind='line', y='Generator 2 Runtime', ax=ax2, xlim=(0, xmax), figsize=(10, 5))
    df.plot(kind='line', y='Generator 3 Runtime', ax=ax2, xlim=(0, xmax), figsize=(10, 5))

    plt.xlabel('Days Since Mission Start')
    plt.ylabel('Generator Runtime (Days)')


# In[208]:


cols = ['Target Current Load', 
        'Available Current Capacity', 
        'Generator 1 Runtime', 
        'Generator 2 Runtime', 
        'Generator 3 Runtime',
        'Running Generator Count',
        'Failed Generator Count']

df = pd.DataFrame(columns=cols)

state = env.reset()
score = 0
action_list = []

for t in range(days): # was 365*3
    action = q_agent.act(state, mode='test')
    action_list.append(action)
    state, reward, done, _ = env.step(action)
    score += reward
    print(state)
    df = df.append(pd.Series(state, index=cols), ignore_index=True)
    if done:
        break 

print('Final Reward:', score)
env.close()


# In[209]:


df.shape


# In[210]:


df.head()


# In[211]:


df.describe()


# In[212]:


#for i in range(1, 50):
#    df = df.append(pd.Series(state, index=cols), ignore_index=True)


# In[213]:


df.shape


# In[214]:


df.tail()


# In[215]:


plot_current(df, days)


# In[216]:


plot_gen_run(df, days)


# In[217]:


len(action_list)


# In[218]:


print(action_list)


# # Random Generators On

# In[219]:


state = env.reset()
score = 0

df_random = pd.DataFrame(columns=cols)

for t in range(days): # was 375*3 - error?
    action = env.action_space.sample()
    env.render()
    state, reward, done, _ = env.step(action)
    print(state)
    df_random = df_random.append(pd.Series(state, index=cols), ignore_index=True)
    score += reward
    if done:
        break 

print('Final score:', score)
env.close()


# In[220]:


df_random.shape


# In[221]:


df_random.head()


# In[222]:


df_random.describe()


# In[223]:


#for i in range(1, 50):
#    df_random = df_random.append(pd.Series(state, index=cols), ignore_index=True)


# In[224]:


df_random.shape


# In[225]:


df_random.tail()


# In[226]:


plot_current(df_random, days)


# In[227]:


plot_gen_run(df_random, days)


# # All Generators On

# In[228]:


df_baseline = pd.DataFrame(columns=cols)

state = env.reset()
score = 0
choices = [0, 3, 6]

for t in range(days): # was 365*3
    action = random.choice(choices)
    env.render()
    state, reward, done, _ = env.step(action)
    print(state) 
    df_baseline = df_baseline.append(pd.Series(state, index=cols), ignore_index=True)
    score += reward
    if done:
        break

print('Final score:', score)
env.close()


# In[229]:


df_baseline.shape


# In[230]:


df_baseline.head()


# In[231]:


df_baseline.describe()


# In[232]:


#for i in range(1, 100):
#    df_baseline = df_baseline.append(pd.Series(state, index=cols), ignore_index=True)


# In[233]:


df_baseline.shape


# In[234]:


df_baseline.tail()


# In[235]:


plot_current(df_baseline, days)


# In[236]:


plot_gen_run(df_baseline, days)


# # Thresholding Agent

# In[237]:


df_thresh = pd.DataFrame(columns=cols)

state = env.reset()
score = 0

df_thresh = df_thresh.append(pd.Series(state, index=cols), ignore_index=True)

for t in range(days): # was 365*3
    
    # If Target Current Load is 90% of Max Current Capacity, 
    # turn on generator with least runtime
    if state[0] >= 0.90*state[1]:
        if state[2] <= state[3] and state[2] <= state[4]:
            action = 0
        elif state[3] <= state[2] and state[3] <= state[4]:
            action = 3
        elif state[4] <= state[2] and state[4] <= state[3]:
            action = 6
        else:
            action = 9
    
    # If no Generators are on, turn on a generator
    elif state[5] == 0:
        if state[2] <= state[3] and state[2] <= state[4]:
            action = 0
        elif state[3] <= state[2] and state[3] <= state[4]:
            action = 3
        elif state[4] <= state[2] and state[4] <= state[3]:
            action = 6
        else:
            action = random.choice([0, 3, 6])
    
    # If Target Current Load is less than 25% of Max Current Capacity, 
    # turn off generator with most runtime
    elif state[0] < 0.25*state[1] and state[5] > 1:
        if state[2] >= state[3] and state[2] >= state[4]:
            action = 1
        elif state[3] >= state[2] and state[3] >= state[4]:
            action = 4
        elif state[4] >= state[2] and state[4] >= state[3]:
            action = 7
        else:
            action = 9
        
    else:
        action = 9

    env.render()
    state, reward, done, _ = env.step(action)
    print(state)
        
    df_thresh = df_thresh.append(pd.Series(state, index=cols), ignore_index=True)
    
    score += reward
    if done:
        break

print('Final score:', score)
env.close()


# In[238]:


df_thresh.shape


# In[239]:


df_thresh.head()


# In[240]:


df_thresh.describe()


# In[241]:


#for i in range(1, 500):
#    df_thresh = df_thresh.append(pd.Series(state, index=cols), ignore_index=True)


# In[242]:


df_thresh.shape


# In[243]:


df_thresh.tail()


# In[244]:


plot_current(df_thresh, days)


# In[245]:


plot_gen_run(df_thresh, days)


# # Thresholding Agent with Maintenance

# In[246]:


df_thresh = pd.DataFrame(columns=cols)

state = env.reset()
score = 0

df_thresh = df_thresh.append(pd.Series(state, index=cols), ignore_index=True)

for t in range(days): # was 365*3
    
    # If Target Current Load is 90% of Max Current Capacity, 
    # turn on generator with least runtime
    if state[0] >= 0.90*state[1]:
        if state[2] <= state[3] and state[2] <= state[4]:
            action = 0
        elif state[3] <= state[2] and state[3] <= state[4]:
            action = 3
        elif state[4] <= state[2] and state[4] <= state[3]:
            action = 6
        else:
            action = 9
    
    # If no Generators are on, turn on a generator
    elif state[5] == 0:
        if state[2] <= state[3] and state[2] <= state[4]:
            action = 0
        elif state[3] <= state[2] and state[3] <= state[4]:
            action = 3
        elif state[4] <= state[2] and state[4] <= state[3]:
            action = 6
        else:
            action = random.choice([0, 3, 6])
    
    # If Target Current Load is less than 25% of Max Current Capacity, 
    # turn off generator with most runtime
    elif state[0] < 0.25*state[1] and state[5] > 1:
        if state[2] >= state[3] and state[2] >= state[4]:
            action = 2
        elif state[3] >= state[2] and state[3] >= state[4]:
            action = 5
        elif state[4] >= state[2] and state[4] >= state[3]:
            action = 8
        else:
            action = 9
        
    else:
        action = 9

    env.render()
    state, reward, done, _ = env.step(action)
    print(state)
        
    df_thresh = df_thresh.append(pd.Series(state, index=cols), ignore_index=True)
    
    score += reward
    if done:
        break

print('Final score:', score)
env.close()


# In[247]:


df_thresh.shape


# In[248]:


df_thresh.head()


# In[249]:


df_thresh.describe()


# In[250]:


#for i in range(1, 500):
#    df_thresh = df_thresh.append(pd.Series(state, index=cols), ignore_index=True)


# In[251]:


df_thresh.shape


# In[252]:


df_thresh.tail()


# In[253]:


plot_current(df_thresh, days)


# In[254]:


plot_gen_run(df_thresh, days)

