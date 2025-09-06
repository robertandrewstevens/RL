#!/usr/bin/env python
# coding: utf-8

# https://medium.com/@lynzt/no-module-named-gym-jupyter-notebook-4c75fb55e299
#
# In the terminal, run the following commands:
#
# - Install gym: `pip install gym`
#
# - Show location of installed gym package (note the Location line): `pip show gym`
#
# Back in the Jupyter notebook, add the following in the cell that imports the `gym` module:
#
# ```
# import sys
# sys.path.append('location found above')
# ```

# https://stackoverflow.com/questions/56641165/modulenotfounderror-no-module-named-keras-for-jupyter-notebook
#
# You have to install all the dependencies first before using it. Try using
#
# ```
# conda install tensorflow
# conda install keras
# ```
#
# by installing it with conda command it manage your versions compatibility with other libraries.

# Updated python from version 3.6 to version 3.8 to get keras to work

# In[1]:


import sys
# sys.path.append('/home/tdird/.local/lib/python3.6/site-packages')
# sys.path.append('/home/tdird/anaconda3/envs/py38/lib/python3.8/site-packages')


# In[2]:


import gym
# import gym_merlin


# In[3]:


# sys.path.append('/home/tdird/anaconda3/envs/py36/lib/python3.6/site-packages')
# sys.path.append('/home/tdird/anaconda3/envs/py38/lib/python3.8/site-packages')


# In[4]:


import numpy as np
import random
# https://stackoverflow.com/questions/53135439/issue-with-add-method-in-tensorflow-attributeerror-module-tensorflow-python
# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# from keras.optimizers import Adam
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from collections import deque


# In[5]:


class DQNN:
    def __init__(self, env):
        self.env     = env
        self.memory  = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.01
        self.tau = 0.05
        self.model = self.create_model()
        # "hack" implemented by DeepMind to improve convergence
        self.target_model = self.create_model()

    def create_model(self):
        model = Sequential()
        state_shape = self.env.observation_space.shape
        model.add(Dense(24, input_dim=state_shape[0], activation="relu"))
        model.add(Dense(48, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(self.env.action_space.n))
        model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        batch_size = 32
        if len(self.memory) < batch_size:
            return
        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            state = state.reshape(1, -1)
            target = self.target_model.predict(state)
            if done:
                target[0][action] = reward
            else:
                new_state = new_state.reshape(1, -1)
                Q_future = max(self.target_model.predict(new_state)[0])
                target[0][action] = reward + Q_future * self.gamma
            self.model.fit(state, target, epochs=1, verbose=0)

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i]
        self.target_model.set_weights(target_weights)

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            result = self.env.action_space.sample()
            return result
        state = state.reshape(1, -1)
        prediction = self.model.predict(state)
        maxarg = np.argmax(prediction[0])
        return maxarg


# In[6]:


def main():
    env = gym.make("MountainCar-v0")
    # env = gym.make("gym_merlin:merlin_chilled_water_simple-v0")
    gamma = 0.9
    epsilon = 0.95
    trials = 10  # was 100
    trial_len = 100  # was 500
    # updateTargetNetwork = 1000  # ???
    dqn_agent = DQNN(env=env)  # was 'DQN' but class is 'DQNN'
    # steps = []  # not used?
    for trial in range(trials):
        cur_state = env.reset()        
        for step in range(trial_len):
            action = dqn_agent.act(cur_state)
            env.render()
            new_state, reward, done, _ = env.step(action)
            reward = reward if not done else -20
            print('Trial {}, Step {}, Reward = {}'.format(trial, step, reward))
            cur_state = env.reset()
            dqn_agent.remember(cur_state, action, reward, new_state, done)
            dqn_agent.replay()
            dqn_agent.target_train()
            cur_state = new_state
            if done:
                break

        if step < trial_len - 1:
            print('Trial {} stopped early at step {}'.format(trial, step))
        else:
            print("Trial {} completed {} steps".format(trial, trial_len))
            # break

    print('Done')


# In[7]:


if __name__ == "__main__":
    main()


# In[ ]:




