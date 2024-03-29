from A2C_vec.A2C import A2C
import time
from threading import Thread
from multiprocessing import cpu_count
from tqdm import tqdm
import numpy as np
from keras.utils import to_categorical
import tensorflow as tf
import gym


class A3C(A2C):
    def __init__(self, state_shape, n_action, net, n_threads=None, model_path='model/a3c'):
        self.state_shape = state_shape
        self.n_action = n_action
        if n_threads == None:
            self.n_threads = cpu_count()
        else:
            self.n_threads = n_threads

        super(A3C, self).__init__(state_shape,n_action,net,model_path=model_path)

    def trainAsy(self, env_name, episodes):
        """异步地使用A2C的train方法
        """
        envs = [env_name() for i in range(self.n_threads)]
        threads = [Thread(target=self.train, args=(envs[i], episodes)) for i in range(self.n_threads)]
        for t in threads:
            t.start()
            time.sleep(1)
        [t.join() for t in threads]
