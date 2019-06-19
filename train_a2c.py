# import matplotlib.pyplot as plt
from A2C.A2C import A2C
from utils.net import conv_shared
# from utils.im_processor import im_processor

a2c = A2C(state_shape=(80,80,5), n_action=2, net=conv_shared)
# a2c.actor.model.summary()
# a2c.critic.model.summary()


class police:
    @staticmethod
    def reset():
        police.pi = [1] + [0]*10 + [1] +[0]*20 + [1] +[0]*15 + [1] +[0]*20 + [1]+ [0]*100
    @staticmethod
    def step():
        a = police.pi[0]
        police.pi.pop(0)
        return a




a2c.explore(act_police=police,episode=50)
a2c.train(episode=1000)
