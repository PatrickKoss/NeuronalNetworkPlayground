import random
import time
from collections import deque

import gym
import keras
import numpy as np
from PIL import Image
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential
from keras.optimizers import Adam

# Factor for resizing the current state of gym step which is a image
IMG_RESIZE_FACTOR = 0.25


class Assault:
    def __init__(self, env):
        '''init all variables'''
        self.env = env
        self.epsilon = 1.0
        self.memory = deque(maxlen=2000)
        self.gamma = 0.85
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 1e-3
        self.tau = .125
        # if you want to train the model from scratch uncomment the lines below
        # self. model = self.create_model()
        # self.target_model = self.create_model()
        # load in the train models for retraining
        self.model = keras.models.load_model("assaultModel-resized.h5")
        self.target_model = keras.models.load_model("assaultModel-resized.h5")

    def create_model(self):
        '''standard model for image analysing'''
        model = Sequential()
        state_shape = self.env.observation_space.shape
        resized_shape = (int(state_shape[0] * IMG_RESIZE_FACTOR), int(state_shape[1] * IMG_RESIZE_FACTOR)) + (1,)
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=resized_shape))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(7))
        model.compile(loss='mean_squared_error', optimizer=Adam(lr=self.learning_rate),
                      metrics=['accuracy'])
        return model

    def act(self, state):
        '''predict an action according to an reinforcement method'''
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        state = get_resized_state(state)
        return np.argmax(self.model.predict(state)[0])

    def remember(self, state, action, reward, new_state, done):
        '''for reinforcement learning you need a memory. This method stores all necessary variables.'''
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        '''train the model with previous states. How many can be configured in batch_size'''
        batch_size = 32
        if len(self.memory) < batch_size:
            return
        # get random sample from the memory
        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            state = get_resized_state(state)
            target = self.target_model.predict(state)
            if done:
                target[0][action] = reward
            else:
                # resize the output picture of gym assault
                new_state = get_resized_state(new_state)
                # calculate the Q_future value. It is the value of the action
                Q_future = max(self.target_model.predict(new_state)[0])
                # calculate a score according to a reinforcement formula. target is our label for training our model.
                target[0][action] = reward + Q_future * self.gamma
            self.model.fit(state, target, epochs=1, verbose=0)

    def target_train(self):
        '''for our target model we need to adjust the weights according to a reinforcement formula'''
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save_model(self, fn):
        '''save the model to a passed path'''
        self.model.save(fn)


def start_train():
    env = gym.make("Assault-v0")
    # train our model for 200 games
    trails = 200
    # train 2000 steps on each game max
    trails_len = 2000

    dqn_agent = Assault(env=env)

    for trial in range(trails):
        cur_state = env.reset()
        for step in range(trails_len):
            # get an action from the prediction of our model
            action = dqn_agent.act(cur_state)
            # get new_state, reward, done on the predicted action
            new_state, reward, done, _ = env.step(action)
            # store this information in memory
            dqn_agent.remember(cur_state, action, reward, new_state, done)

            # once the action is predicted, train the model on last actions
            dqn_agent.replay()
            # adjust the weights on our target model
            dqn_agent.target_train()

            # set the current state to the new state
            cur_state = new_state
            print("step: {}".format(step))
            if done:
                break
        # after each game save the model
        dqn_agent.save_model("assaultModel-resized.h5")

        print("trial: {}".format(trial))

    dqn_agent.save_model("assaultModel-resized.h5")


def let_play():
    '''let our model play the assault play and watch it'''
    env = gym.make("Assault-v0")
    cur_state = env.reset()
    done = False
    model = keras.models.load_model("assaultModel-resized.h5")
    while not done:
        env.render()

        cur_state = get_resized_state(cur_state)
        action = np.argmax(model.predict(cur_state)[0])

        cur_state, reward, done, _ = env.step(action)

        time.sleep(0.05)


def get_resized_state(state):
    '''this method resize and grayscale the output image of the gym assault'''
    res_state = Image.fromarray(state).resize(
        size=(int(state.shape[1] * IMG_RESIZE_FACTOR), int(state.shape[0] * IMG_RESIZE_FACTOR))).convert('L')
    res_state = np.array(res_state)
    state_shape = (-1,) + res_state.shape + (1,)
    res_state = res_state.reshape(state_shape)
    return res_state


# meaning of the taken actions, represented by numbers
ACTION_MEANING = {
    0: "NOOP",
    1: "FIRE",
    2: "UP",
    3: "RIGHT",
    4: "LEFT",
    5: "DOWN",
    6: "UPRIGHT",
    7: "UPLEFT",
    8: "DOWNRIGHT",
    9: "DOWNLEFT",
    10: "UPFIRE",
    11: "RIGHTFIRE",
    12: "LEFTFIRE",
    13: "DOWNFIRE",
    14: "UPRIGHTFIRE",
    15: "UPLEFTFIRE",
    16: "DOWNRIGHTFIRE",
    17: "DOWNLEFTFIRE",
}

if __name__ == '__main__':
    # start_train()
    let_play()
