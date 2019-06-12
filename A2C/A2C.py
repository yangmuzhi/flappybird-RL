import numpy as np
from tqdm import tqdm
from keras.models import Model
from keras import regularizers
from keras.utils import to_categorical
from keras.layers import Input, Dense, Flatten
from utils.sample_buffer import Sampling_Pool
from .Actor import Actor
from .Critic import Critic
import tensorflow as tf
import keras.backend as K
import os
from utils.im_processor import im_processor
import game.wrapped_flappy_bird as game


sampling_pool = Sampling_Pool()
class A2C:
    """
    """
    def __init__(self, state_shape, n_action, net, model_path='model/a2c'):
        self.state_shape = state_shape
        self.n_action = n_action
        self.lr = 1e-4
        self.gamma = 0.9
        share_net = self.build_share_net(net)
        self.critic = Critic(self.state_shape,1,
                self.lr, share_net)
        self.actor = Actor(self.state_shape,self.n_action,
                self.lr, share_net)
        self.cum_r = []
        self.actor_update = self.actor.update()
        self.critic_update = self.critic.update()
        self.model_path = model_path
        global graph
        graph = tf.get_default_graph()


    def build_share_net(self, net):
        # 暂时使用一个简单的网络, 后面应该做一个net.py
        # actor 和 critic 共享一个net
        if  isinstance(self.state_shape, int):
            inputs = Input((self.state_shape,))
        else:
            inputs = Input(self.state_shape)
        x = net(inputs)

        return Model(inputs, x)

    def discount(self, reward):
        """ 没有使用discount,这是一个错误
        """
        discounted_reward, cumul_reward = np.zeros_like(reward), 0
        for t in reversed(range(0, len(reward))):
            cumul_reward = reward[t] + cumul_reward * self.gamma
            discounted_reward[t] = cumul_reward
        return discounted_reward

    def update(self, sampling_pool):
        state, reward, done, action, next_state = \
            sampling_pool.get_sample(shuffle=False)
        value = self.critic.value(state)
        # next_value = self.critic.value(next_state)
        # td_error = reward + self.gamma * next_value - value
        # new_value = reward + self.gamma * next_value
        discounted_reward = self.discount(reward)
        advantages = discounted_reward - value
        self.actor_update([state, action, advantages])
        self.critic_update([state, discounted_reward])
        sampling_pool.clear()

    def train(self,episode, sampling_pool=sampling_pool):

        with graph.as_default():
            tqdm_e = tqdm(range(episode))
            for i in tqdm_e:
                env = game.GameState()
                state = env.reset()
                cum_r = 0
                done = False
                while not done:

                    state = im_processor(state)
                    state_newaxis = state[np.newaxis,:]
                    action = self.actor.explore(state_newaxis)
                    action_array = np.array([0,0])
                    action_array[action] = 1
                    next_state,reward,done = env.step(action_array)
                    action_onehot = to_categorical(action, self.n_action)
                    ob = (state, reward, done, action_onehot, next_state)
                    sampling_pool.add_to_buffer(ob)
                    state = next_state
                    cum_r += reward

                self.update(sampling_pool)
                self.cum_r.append(cum_r)
                tqdm_e.set_description("Score: " + str(cum_r))
                tqdm_e.refresh()
                if (i > 10000) &  (not(i % 10000)):
                    self.save_model(f"{i}-eps-.h5")
                del env
            self.save_model(f"final-{i}-eps-.h5")


    def random_act(self,episode):

        tqdm_e = tqdm(range(episode))
        for i in tqdm_e:
            env = game.GameState()
            state = env.reset()
            cum_r = 0
            done = False
            while not done:

                action_array = np.array([0,0])
                action_array[np.random.randint(2)] = 1
                _, reward, done = env.step(action_array)
                cum_r += reward

            self.cum_r.append(cum_r)
            tqdm_e.set_description("Score: " + str(cum_r))
            tqdm_e.refresh()
            if (i > 10000) &  (not(i % 10000)):
                self.save_model(f"{i}-eps-.h5")
            del env





    def save_model(self, save_name):
        path = self.model_path
        if not os.path.exists(path):
            os.makedirs(path)
        self.actor.model.save(os.path.join(path, save_name))
