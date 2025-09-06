#!/usr/bin/env python
# coding: utf-8

# # Using Reinforcement Learning for Generator Optimization
# 
# 
# ### 1. Import the Necessary Packages

# In[1]:

import sys
import gym
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

# Set plotting options
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')
np.set_printoptions(precision=3, linewidth=120)


# ### 2. Specify the Environment, and Explore the State and Action Spaces
# 
# We'll use [OpenAI Gym](https://gym.openai.com/) environments to test and develop our algorithms. These simulate a variety of classic as well as contemporary reinforcement learning tasks.  Let's use an environment that has a continuous state space, but a discrete action space.

# In[2]:

# Create the PowerPlant environment
env = gym.make('gym_powerplant:powerplant_complex-v0')
#env.seed(505);


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

# In[4]:

# Explore state (observation) space
print("State space:", env.observation_space)
print("- low:", env.observation_space.low)
print("- high:", env.observation_space.high)


# In[5]:

# Generate some samples from the state space 
print("State space samples:")
print(np.array([env.observation_space.sample() for i in range(10)]))

# In[6]:

# Explore the action space
print("Action space:", env.action_space)

# Generate some samples from the action space
print("Action space samples:")
print(np.array([env.action_space.sample() for i in range(10)]))


# ### 3. Discretize the State Space with a Uniform Grid

# In[7]:

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
    grid = [np.linspace(low[dim], high[dim], bins[dim] + 1)[1:-1] for dim in range(len(bins))]
    print("Uniform grid: [<low>, <high>] / <bins> => <splits>")
    for l, h, b, splits in zip(low, high, bins, grid):
        print("    [{}, {}] / {} => {}".format(l, h, b, splits))
    return grid


low = [-1.0, -5.0, -5.0, -5.0, -5.0, 0, 0]
high = [1.0, 5.0, 5.0, 5.0, 5.0, 3, 3]
create_uniform_grid(low, high)  # [test]


# In[8]:


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
    return list(int(np.digitize(s, g)) for s, g in zip(sample, grid))  # apply along each dimension


# Test with a simple grid and some samples
grid = create_uniform_grid([-1.0, -5.0, -10, -10, -10, 0, 0], [1.0, 5.0, 10, 10, 10, 3, 3])
samples = np.array(
    [[-1.0 , 5.0, 10, 9.4, 9.0, 0, 1],
     [-0.81, 4.1, 4.0, 5.6, 7.5, 1, 2],
     [-0.8 , 4.0, -10, -2.0, 7.8, 1, 3],
     [-0.5 ,  0.0, -9, -4.0, 4.4, 1, 1],
     [ 0.2 , 1.9, 9, 2.3, 7.0, 1, 1],
     [ 0.8 ,  4.0, -2, 6.6, 6.9, 1, 0],
     [ 0.81,  4.1, 2.0, -1.2, 8.8, 1, 2],
     [ 1.0 ,  5.0, 10, -9, -3, 0, 3]])
discretized_samples = np.array([discretize(sample, grid) for sample in samples])
print("\nSamples:", repr(samples), sep="\n")
print("\nDiscretized samples:", repr(discretized_samples), sep="\n")


# In[9]:


# Create a grid to discretize the state space
state_grid = create_uniform_grid(env.observation_space.low, env.observation_space.high, bins=(10, 3, 10, 10, 10, 3, 3))
state_grid


# ### 5. Q-Learning
# 
# Provided below is a simple Q-Learning agent.

# In[10]:


class QLearningAgent:
    """Q-Learning agent that can act on a continuous state space by discretizing it."""

    def __init__(self, env, state_grid, alpha=0.02, gamma=0.97,
                 epsilon=1.0, epsilon_decay_rate=0.9995, min_epsilon=.01, seed=505):
        """Initialize variables, create grid for discretization."""
        # Environment info
        self.env = env
        self.state_grid = state_grid
        self.state_size = tuple(len(splits) + 1 for splits in self.state_grid)  # n-dimensional state space
        self.action_size = self.env.action_space.n  # 1-dimensional discrete action space
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
            # Note: We update the Q table entry for the *last* (state, action) pair with current state, reward
            self.q_table[self.last_state + (self.last_action,)] += self.alpha *                 (reward + self.gamma * max(self.q_table[state]) - self.q_table[self.last_state + (self.last_action,)])

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


# In[12]:


def run(agent, env, num_episodes=10000, mode='train'):
    """Run agent in given reinforcement learning environment and return scores."""
    scores = []
    max_avg_score = -np.inf
    for i_episode in range(1, num_episodes+1):
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
                #print(state)
                sys.stdout.flush()

    return scores

scores = run(q_agent, env)


# In[13]:


# Plot scores obtained per episode
plt.plot(scores); plt.title("Generator Optimization"); plt.xlabel("Episode"); plt.ylabel("Rewards")


# In[14]:


def plot_scores(scores, rolling_window=100):
    """Plot scores and optional rolling mean using specified window."""
    #plt.plot(scores); plt.title("Scores");
    rolling_mean = pd.Series(scores).rolling(rolling_window).mean()
    plt.plot(rolling_mean); plt.title("Generator Optimization"); plt.xlabel("Episode"); plt.ylabel("Rolling Mean Rewards")
    return rolling_mean

rolling_mean = plot_scores(scores)


# ### 7. Watch a Q-Learning Agent

# In[15]:


import pandas as pd
df = pd.DataFrame(columns=['Target Current Load', 
                           'Available Current Capacity', 
                           'Generator 1 Runtime', 
                           'Generator 2 Runtime', 
                           'Generator 3 Runtime',
                           'Running Generator Count',
                           'Failed Generator Count'])

state = env.reset()
score = 0
action_list = []
for t in range(365*3):
    action = q_agent.act(state, mode='test')
    action_list.append(action)
    #env.render()
    state, reward, done, _ = env.step(action)
    score += reward
    print(state)
    df = df.append(pd.Series(state, index=['Target Current Load', 
                                           'Available Current Capacity', 
                                           'Generator 1 Runtime', 
                                           'Generator 2 Runtime', 
                                           'Generator 3 Runtime',
                                           'Running Generator Count',
                                           'Failed Generator Count']), ignore_index=True)
    if done:
        break 
print('Final Reward:', score)
env.close()


# In[16]:


df.head()


# In[17]:


df.describe()


# In[18]:


for i in range(1,50):
        df = df.append(pd.Series(state, index=['Target Current Load', 
                                           'Available Current Capacity', 
                                           'Generator 1 Runtime', 
                                           'Generator 2 Runtime', 
                                           'Generator 3 Runtime',
                                           'Running Generator Count',
                                           'Failed Generator Count']), ignore_index=True)


# In[19]:


ax = plt.gca()

#df.plot(kind='line', y='Target Current Load', ax=ax, xlim = (0,1200))
#df.plot(kind='line', y='Available Current Capacity', ax=ax, xlim = (0,1200))

df.plot(kind='line', y='Target Current Load', ax=ax, xlim = (0,800), figsize=(10,5))
df.plot(kind='line', y='Available Current Capacity', ax=ax, xlim = (0,800), figsize=(10,5))

plt.xlabel('Days Since Mission Start')
plt.ylabel('Current (A)')


# In[20]:


ax2 = plt.gca()
ax2.ylabel = "Runtime"
df.plot(kind='line', y='Generator 1 Runtime', ax=ax2, xlim = (0,800), figsize=(10,5))
df.plot(kind='line', y='Generator 2 Runtime', ax=ax2, xlim = (0,800), figsize=(10,5))
df.plot(kind='line', y='Generator 3 Runtime', ax=ax2, xlim = (0,800), figsize=(10,5))

plt.xlabel('Days Since Mission Start')
plt.ylabel('Generator Runtime (Days)')


# In[21]:


print(action_list)


# # Random Generators On

# In[22]:


state = env.reset()
score = 0

df_random = pd.DataFrame(columns=['Target Current Load', 
                           'Available Current Capacity', 
                           'Generator 1 Runtime', 
                           'Generator 2 Runtime', 
                           'Generator 3 Runtime',
                           'Running Generator Count',
                           'Failed Generator Count'])

for t in range(375*3):
    action = env.action_space.sample()
    env.render()
    state, reward, done, _ = env.step(action)
    print(state)
        
    df_random = df_random.append(pd.Series(state, index=['Target Current Load', 
                                           'Available Current Capacity', 
                                           'Generator 1 Runtime', 
                                           'Generator 2 Runtime', 
                                           'Generator 3 Runtime',
                                           'Running Generator Count',
                                           'Failed Generator Count']), ignore_index=True)
    
    score += reward
    if done:
        break 
print('Final score:', score)
env.close()


# In[23]:


df.head()


# In[24]:

df.describe()

# In[25]:

for i in range(1,50):
    df_random = df_random.append(pd.Series(state, index=['Target Current Load', 
                                           'Available Current Capacity', 
                                           'Generator 1 Runtime', 
                                           'Generator 2 Runtime', 
                                           'Generator 3 Runtime',
                                           'Running Generator Count',
                                           'Failed Generator Count']), ignore_index=True)


# In[26]:

ax_random = plt.gca()
ax_random.ylabel = "Current"
df_random.plot(kind='line', y='Target Current Load', ax=ax_random, xlim=(0, 1200))
df_random.plot(kind='line', y='Available Current Capacity', ax=ax_random, xlim =(0, 1200))

plt.xlabel('Days Since Mission Start')
plt.ylabel('Current (A)')

# In[27]:

ax2_random = plt.gca()
ax2_random.ylabel = "Runtime"
df_random.plot(kind='line', y='Generator 1 Runtime', ax=ax2_random, xlim = (0, 1200))
df_random.plot(kind='line', y='Generator 2 Runtime', ax=ax2_random, xlim = (0, 1200))
df_random.plot(kind='line', y='Generator 3 Runtime', ax=ax2_random, xlim = (0, 1200))

plt.xlabel('Days Since Mission Start')
plt.ylabel('Generator Runtime (Days)')

# # All Generators On

# In[28]:

import random

df_baseline = pd.DataFrame(
    columns=['Target Current Load', 
             'Available Current Capacity', 
             'Generator 1 Runtime', 
             'Generator 2 Runtime', 
             'Generator 3 Runtime',
             'Running Generator Count',
             'Failed Generator Count'])

state = env.reset()
score = 0
choices = [0, 3, 6]

for t in range(365*3):
    action = random.choice(choices)
    env.render()
    state, reward, done, _ = env.step(action)
    print(state)
        
    df_baseline = df_baseline.append(
        pd.Series(
            state, 
            index=['Target Current Load', 
                   'Available Current Capacity', 
                   'Generator 1 Runtime', 
                   'Generator 2 Runtime', 
                   'Generator 3 Runtime',
                   'Running Generator Count',
                   'Failed Generator Count']), 
    ignore_index=True)
    
    score += reward
    if done:
        break 

print('Final score:', score)
env.close()

# In[29]:

df.head()

# In[30]:

df.describe()

# In[31]:

for i in range(1, 100):
    df_baseline = df_baseline.append(
        pd.Series(
            state, 
            index=['Target Current Load', 
                   'Available Current Capacity', 
                   'Generator 1 Runtime', 
                   'Generator 2 Runtime', 
                   'Generator 3 Runtime',
                   'Running Generator Count',
                   'Failed Generator Count']), 
        ignore_index=True)

# In[32]:


ax_baseline = plt.gca()
ax_baseline.ylabel = "Current"
df_baseline.plot(kind='line', y='Target Current Load', ax=ax_baseline, xlim = (0, 800), figsize=(10, 5))
df_baseline.plot(kind='line', y='Available Current Capacity', ax=ax_baseline, xlim = (0, 800), figsize=(10, 5))

plt.xlabel('Days Since Mission Start')
plt.ylabel('Current (A)')


# In[33]:


ax2_baseline = plt.gca()
ax2_baseline.ylabel = "Runtime"
df_baseline.plot(kind='line', y='Generator 1 Runtime', ax=ax2_baseline, xlim = (0, 800), figsize=(10, 5))
df_baseline.plot(kind='line', y='Generator 2 Runtime', ax=ax2_baseline, xlim = (0, 800), figsize=(10, 5))
df_baseline.plot(kind='line', y='Generator 3 Runtime', ax=ax2_baseline, xlim = (0, 800), figsize=(10, 5))

plt.xlabel('Days Since Mission Start')
plt.ylabel('Generator Runtime (Days)')

# # Thresholding Agent

# In[34]:

import random

df_thresholding = pd.DataFrame(
    columns=['Target Current Load', 
             'Available Current Capacity', 
             'Generator 1 Runtime', 
             'Generator 2 Runtime', 
             'Generator 3 Runtime',
             'Running Generator Count',
             'Failed Generator Count'])

state = env.reset()
score = 0

df_thresholding = df_thresholding.append(pd.Series(state, index=['Target Current Load', 
                                           'Available Current Capacity', 
                                           'Generator 1 Runtime', 
                                           'Generator 2 Runtime', 
                                           'Generator 3 Runtime',
                                           'Running Generator Count',
                                           'Failed Generator Count']), ignore_index=True)


for t in range(365*3):
    
    # If Target Current Load is 90% of Max Current Capacity, turn on generator with least runtime
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
    
    # If Target Current Load is less than 25% of Max Current Capacity, turn off generator with most runtime
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
        
    df_thresholding = df_thresholding.append(
        pd.Series(
           state, 
           index=['Target Current Load', 
                  'Available Current Capacity', 
                  'Generator 1 Runtime', 
                  'Generator 2 Runtime', 
                  'Generator 3 Runtime',
                  'Running Generator Count',
                  'Failed Generator Count']), 
        ignore_index=True)
    
    score += reward
    if done:
        break 

print('Final score:', score)
env.close()

# In[35]:

df.head()

# In[36]:

df.describe()

# In[37]:

for i in range(1, 500):
    df_thresholding = df_thresholding.append(
        pd.Series(
            state, 
            index=['Target Current Load', 
                   'Available Current Capacity', 
                   'Generator 1 Runtime', 
                   'Generator 2 Runtime', 
                   'Generator 3 Runtime',
                   'Running Generator Count',
                   'Failed Generator Count']), 
        ignore_index=True)

# In[38]:

ax_thresholding = plt.gca()
ax_thresholding.ylabel = "Current"
df_thresholding.plot(kind='line', y='Target Current Load', ax=ax_thresholding, xlim = (0, 1200), figsize =(10, 5))
df_thresholding.plot(kind='line', y='Available Current Capacity', ax=ax_thresholding, xlim = (0, 1200), figsize =(10, 5))

plt.xlabel('Days Since Mission Start')
plt.ylabel('Current (A)')

# In[39]:

ax2_thresholding = plt.gca()
ax2_thresholding.ylabel = "Runtime"
df_thresholding.plot(kind='line', y='Generator 1 Runtime', ax=ax2_thresholding, xlim = (0, 1200))
df_thresholding.plot(kind='line', y='Generator 2 Runtime', ax=ax2_thresholding, xlim = (0, 1200))
df_thresholding.plot(kind='line', y='Generator 3 Runtime', ax=ax2_thresholding, xlim = (0, 1200))

plt.xlabel('Days Since Mission Start')
plt.ylabel('Generator Runtime (Days)')


# # Thresholding Agent with Maintenance

# In[40]:

import random

df_thresholding = pd.DataFrame(
    columns=['Target Current Load', 
             'Available Current Capacity', 
             'Generator 1 Runtime', 
             'Generator 2 Runtime', 
             'Generator 3 Runtime',
             'Running Generator Count',
             'Failed Generator Count'])

state = env.reset()
score = 0

df_thresholding = df_thresholding.append(
    pd.Series(
        state, 
        index=['Target Current Load', 
               'Available Current Capacity', 
               'Generator 1 Runtime', 
               'Generator 2 Runtime', 
               'Generator 3 Runtime',
               'Running Generator Count',
               'Failed Generator Count']), 
    ignore_index=True)


for t in range(365*3):
    
    # If Target Current Load is 90% of Max Current Capacity, turn on generator with least runtime
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
    
    # If Target Current Load is less than 25% of Max Current Capacity, turn off generator with most runtime
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
        
    df_thresholding = df_thresholding.append(
        pd.Series(
            state, 
            index=['Target Current Load', 
                   'Available Current Capacity', 
                   'Generator 1 Runtime', 
                   'Generator 2 Runtime', 
                   'Generator 3 Runtime',
                   'Running Generator Count',
                   'Failed Generator Count']), 
        ignore_index=True)
    
    score += reward
    if done:
        break 

print('Final score:', score)
env.close()


# In[41]:

df.head()

# In[42]:

df.describe()

# In[43]:

for i in range(1, 500):
    df_thresholding = df_thresholding.append(
        pd.Series(
            state, 
            index=['Target Current Load', 
                   'Available Current Capacity', 
                   'Generator 1 Runtime', 
                   'Generator 2 Runtime', 
                   'Generator 3 Runtime',
                   'Running Generator Count',
                   'Failed Generator Count']), 
        ignore_index=True)


# In[44]:

ax_thresholding = plt.gca()
ax_thresholding.ylabel = "Current"
df_thresholding.plot(kind='line', y='Target Current Load', ax=ax_thresholding, xlim = (0, 1200), figsize=(10, 5))
df_thresholding.plot(kind='line', y='Available Current Capacity', ax=ax_thresholding, xlim = (0, 1200), figsize=(10, 5))

plt.xlabel('Days Since Mission Start')
plt.ylabel('Current (A)')


# In[45]:

ax2_thresholding = plt.gca()
ax2_thresholding.ylabel = "Runtime"
df_thresholding.plot(kind='line', y='Generator 1 Runtime', ax=ax2_thresholding, xlim = (0, 1200))
df_thresholding.plot(kind='line', y='Generator 2 Runtime', ax=ax2_thresholding, xlim = (0, 1200))
df_thresholding.plot(kind='line', y='Generator 3 Runtime', ax=ax2_thresholding, xlim = (0, 1200))

plt.xlabel('Days Since Mission Start')
plt.ylabel('Generator Runtime (Days)')

