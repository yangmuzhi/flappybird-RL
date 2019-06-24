# Training a flappybird AI using reinforcement learning


# How to use it

## git clone

```{python}

git clone https://github.com/yangmuzhi/flappybird-RL.git
cd flappybird
python train_a3c.py

```
## environment

env is in the game file.

```{python}

import game.wrapped_flappy_bird as game
import numpy as np
env = game.GameState()
state = env.reset()
action = np.zeros(2)
action[np.random.randint(2)] = 1
state, reward, done = env.step(action)

```
## result



## todo

- [x] training  framework
- [x] save trained models
- [x] better readme
