
import numpy as np
from tqdm import tqdm
from dqn.agent import Agent
from utils.sample_buffer import Sampling_Pool
import os
from utils.im_processor import im_processor
import game.wrapped_flappy_bird as game
from keras.utils  import to_categorical

STATUS = "explore"

class DQN:
    """
    """
    def __init__(self, state_shape, n_action, net, model_path='model/dqn'):
        self.state_shape = state_shape
        self.n_action = n_action
        self.lr = 1e-4
        self.gamma = 0.9
        self.sampling_size = 20000
        self.agent = Agent(self.state_shape,self.n_action,self.lr,0.9,net)
        self.sampling_pool = Sampling_Pool(self.sampling_size)
        self.cum_r = []
        self.model_path = model_path

    def train_agent(self):
        state, reward, done, action, next_state = self.sampling_pool.get_sample(self.batch_size)
        q_target = self.agent.q_target(next_state)
        q = self.agent.q_eval(state)
        q_next = self.agent.q_eval(next_state)

        for i in range(self.batch_size):
            if done[i]:
                q[i, action[i]] = reward[i]
            else:
                max_action = np.argmax(q_next[i,:])
                q[i, action[i]] = reward[i] + self.gamma * q_target[i,max_action]
        self.agent.update(state, q)

    def train(self,episode, batch_size=64, freq=100):
        self.batch_size = batch_size
        tqdm_e = tqdm(range(episode))
        env = game.GameState()

        for i in tqdm_e:
            state = env.reset()
            cum_r = 0
            done = False
            while not done:
                # STATUS = "explore"
                state_newaxis = state[np.newaxis,:]
                action = self.agent.e_greedy_action(state_newaxis)
                action_array = np.array([0,0])
                action_array[action] = 1
                next_state,reward,done = env.step(action_array)
                action_onehot = to_categorical(action, self.n_action)
                ob = (state, reward, done, action_onehot, next_state)
                self.sampling_pool.add_to_buffer(ob)
                state = next_state
                cum_r += reward

                if (self.sampling_pool.get_size() > self.batch_size):
                    self.train_agent()
                    STATUS = "train"
                    if  i % freq == 0:
                        self.agent.transfer_weights()
                        STATUS = "transfer weights"
            self.cum_r.append(cum_r)
            if (i > 10000) &  (not(i % 10000)):
                self.save_model(f"{i}-eps-.h5")
            tqdm_e.set_description("Score: " + str(cum_r) + "\n Status" + STATUS)
            tqdm_e.refresh()
        self.save_model(f"final-{i}-eps-.h5")

    def save_model(self, save_name):
        path = self.model_path
        if not os.path.exists(path):
            os.makedirs(path)
        self.agent.q_eval_net.save(os.path.join(path, save_name))
