import game.wrapped_flappy_bird as game
import numpy as np
import matplotlib.pyplot as plt
from A2C.A2C import A2C
from utils.net import conv_shared
from utils.im_processor import im_processor
from collections import deque
import time

env = game.GameState()


state = env.reset()
state = im_processor(state)



d = 0
while not d:
    a = np.array([0,0])
    a[np.random.randint(2)] = 1
    state, reward, d = env.step(a)

# state = cv2.resize(state, (80, 80))
    plt.imshow(state)
    plt.show()
    # time.sleep(0.5)

reward

state.shape


state_shape = state.shape

state = np.squeeze(state)
stack_num = 5
state_stack = np.stack([state for i in range(stack_num)], axis=2)
state_stack.shape



a2c = A2C(state_shape=(80,80,5), n_action=2, net=conv_shared)
# a2c.actor.model.summary()
# a2c.critic.model.summary()
state[np.newaxis,:].shape

state_stack = state_stack[np.newaxis,:]

state_stack[...,1:].shape

state_stack[...,0] = state
state_stack.shape

d = deque()
d.append(state_stack)

np.array(d).shape

np.append(state[...,np.newaxis], state_stack[..., 1:], axis=2).shape


a2c.actor.explore(state_stack)
a2c.actor.action_prob(state)



a2c.train(episode=100)
a2c.random_act(100)

np.random.randint(2)
# del game_state

#
# # feature
# x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
# ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
# s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
