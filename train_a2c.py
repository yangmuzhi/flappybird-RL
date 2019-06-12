# import matplotlib.pyplot as plt
from A2C.A2C import A2C
from utils.net import conv_shared
# from utils.im_processor import im_processor

a2c = A2C(state_shape=(80,80,1), n_action=2, net=conv_shared)
a2c.actor.model.summary()
a2c.critic.model.summary()

a2c.train(episode=100000)
