import numpy as np
from tqdm import tqdm
from keras.models import Model
from keras import regularizers
from keras.utils import to_categorical
from keras.layers import Input, Dense, Flatten
from collections import deque
from .Actor import Actor
from .Critic import Critic
import tensorflow as tf
import keras.backend as K
import os
from utils.im_processor import im_processor
import game.wrapped_flappy_bird as game

STACK_NUM = 1

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

    def update(self, state, reward, done, action, next_state):
        state = np.array(state)
        reward = np.array(reward)
        action = np.array(action)
        next_state = np.array(next_state)
        # print("state dims", state.shape)
        value = self.critic.value(state)
        discounted_reward = self.discount(reward)
        advantages = discounted_reward - value
        self.actor_update([state, action, advantages])
        self.critic_update([state, discounted_reward])
        print("update")

    def explore(self, act_police,episode=100):
        """增加explore的目的是 使用一个人为策略，收集一些靠谱的数据"""
        tqdm_e = tqdm(range(episode))
        env = game.GameState()
        print("explore")

        for i in tqdm_e:
            done = 0
            state = env.reset()
            act_police.reset()
            # state = np.squeeze(im_processor(state))
            state_stack = np.stack([state for i in range(STACK_NUM)], axis=2)
            s=deque()
            a=deque()
            r=deque()
            d=deque()
            next_s=deque()

            while not done:
                action = act_police.step()
                state_newaxis = state_stack[np.newaxis,:]
                action_array = np.array([0,0])
                action_array[action] = 1
                next_im,reward,done = env.step(action_array)
                # next_im = im_processor(next_im)

                next_state_stack = np.append(next_im,state_stack[..., :-1], axis=2)
                action_onehot = to_categorical(action, self.n_action)

                s.append(state_stack)
                a.append(action_onehot)
                r.append(reward)
                d.append(done)
                next_s.append(next_state_stack)

            self.update(s, r, d, a, next_s)


    def train(self,episode):

        with graph.as_default():
            tqdm_e = tqdm(range(episode))
            env = game.GameState()
            s=deque()
            a=deque()
            r=deque()
            d=deque()
            next_s=deque()
            for i in tqdm_e:
                state = env.reset()
                cum_r = 0
                done = False
                # state = np.squeeze(im_processor(state))
                state_stack = np.stack([state for i in range(STACK_NUM)], axis=2)

                while not done:
                    state_newaxis = state_stack[np.newaxis,:]
                    action = self.actor.explore(state_newaxis)
                    action_array = np.array([0,0])
                    action_array[action] = 1
                    next_im,reward,done = env.step(action_array)
                    # next_im = im_processor(next_im)
                    next_state_stack = np.append(next_im,state_stack[..., :-1], axis=2)
                    action_onehot = to_categorical(action, self.n_action)

                    s.append(state_stack)
                    a.append(action_onehot)
                    r.append(reward)
                    d.append(done)
                    next_s.append(next_state_stack)
                    state_stack = next_state_stack
                    cum_r += reward


                self.cum_r.append(cum_r)
                tqdm_e.set_description("Score: " + str(cum_r))
                tqdm_e.refresh()

                # train

                self.update(s, r, d, a, next_s)
                s=deque()
                a=deque()
                r=deque()
                d=deque()
                next_s=deque()

                if (i > 10000) &  (not(i % 50000)):
                    self.save_model(f"{i}-eps-.h5")

            self.save_model(f"final-{i}-eps-.h5")

    def save_model(self, save_name):
        path = self.model_path
        if not os.path.exists(path):
            os.makedirs(path)
        self.actor.model.save(os.path.join(path, save_name))
