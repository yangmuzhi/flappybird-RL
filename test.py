import game.wrapped_flappy_bird as game
import numpy as np

import matplotlib.pyplot as plt
from A2C.A2C import A2C
from utils.net import conv_shared
from utils.im_processor import im_processor

game_state = game.GameState()


# action [1,0], [0,1]
# action = np.array([1,0])

state = game_state.reset()
# 使用原始图片试试

# state = cv2.resize(state, (80, 80))
plt.imshow(state)
plt.show()

state = im_processor(state)
state_shape = state.shape

a2c = A2C(state_shape=state_shape, n_action=2, net=conv_shared)
a2c.actor.model.summary()
a2c.critic.model.summary()

# state = state[np.newaxis,:]
# a2c.actor.explore(state)

a2c.train(episode=1000)


# del game_state

#
# # feature
# x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
# ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
# s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
