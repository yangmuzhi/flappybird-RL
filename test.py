import game.wrapped_flappy_bird as game
import numpy as np
import matplotlib.pyplot as plt

from A3C.A3C import A3C
from A2C_vec.A2C import A2C
from utils.net import conv_shared, simple_net
import time

env = game.GameState()
#
# state = env.reset()
#
# state
# state.shape
# a3c = A3C(state_shape=3, n_action=2, net=simple_net)
# a3c.trainAsy(game.GameState, episodes=10)


a2c = A2C(state_shape=3, n_action=2, net=simple_net, epsilon=0.5)
a2c.actor.model.summary()
a2c.critic.model.summary()
# state[np.newaxis,:].shape
#
# a2c.actor.explore(state[np.newaxis,:])
# a2c.actor.action_prob(state[np.newaxis,:])
# a2c.explore()
a2c.train(env,episode=10)


# del game_state

#
# # feature
# x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
# ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
# s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
