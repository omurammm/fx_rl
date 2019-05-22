#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random
from keras.models import Sequential, Model
from keras.layers import Conv2D, Flatten, Dense, Input, LSTM, concatenate
import numpy as np
from collections import deque
import tensorflow as tf
import time
from env import Env
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import argparse


from logging import getLogger, StreamHandler, DEBUG, FileHandler
logger = getLogger('testtest')
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)
fh = FileHandler('test.log')
logger.addHandler(fh)
logger.propagate = False

logger.debug('hello')


ENV_NAME = "fx_dqn"

# spread = 0.003 # yen
spread = 0

num_episodes = 300
initial_replay_size = 5000
batch_size = 32
replay_memory_size = 20000
act_interval = 1
train_interval = 1
target_update_interval = 2000


epsilon_optimizer = 1.5e-7
momentum = 0.95
learning_rate = 0.00025


epsilon_init = 1.0
epsilon_fin = 0.05
test_epsilon = 0.05
exploration_steps = 100000
gamma = 0.99

len_input = 100
TRAIN = True
LOAD = False
save_interval = 5000000
save_path = 'saved_networks/' + ENV_NAME






class Agent:
    def __init__(self, num_actions, len_input, args):
        self.num_actions = num_actions
        self.len_input = len_input
        self.num_features = 2+len(args.features)


        self.epsilon = epsilon_init
        self.epsilon_step = (epsilon_init - epsilon_fin) / exploration_steps
        self.t = 0
        self.repeated_action = 0

        # Parameters used for summary
        self.total_reward = 0
        self.total_max_q = 0
        self.total_loss = 0
        self.duration = 0
        self.episode = 0

        self.start = 0
        
        self.replay_memory = deque()
        
        self.s, self.q_values, q_network = self.network()
        q_network_weights = q_network.trainable_weights
        
        self.st, self.target_q_values, target_network = self.network()
        target_network_weights = target_network.trainable_weights

        # Define target network update operation
        self.update_target_network = [target_network_weights[i].assign(q_network_weights[i]) for i in range(len(target_network_weights))]
        
        
        # Define loss and gradient update operation
        self.action, self.y, self.loss, self.grad_update = self.training_op(q_network_weights)
        
        self.sess = tf.InteractiveSession()

        self.saver = tf.train.Saver(q_network_weights)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        self.sess.run(tf.global_variables_initializer())

        # Load network
        if LOAD:
            self.load_network()


        # Initialize target network
        self.sess.run(self.update_target_network)
        
        
    def network(self):
        # model = Sequential()
        # model.add(Dense(400, activation='relu', input_dim=self.len_input))
        # model.add(Dense(300, activation='relu'))
        # model.add(Dense(self.num_actions, activation='linear'))

        chart = Input(shape=(self.num_features,self.len_input))
        lstm = LSTM(8)(chart)
        output = Dense(3)(lstm)

        s = tf.placeholder(tf.float32, [None, self.num_features, self.len_input])
        model = Model(inputs=chart, outputs=output)
        q_values = model(s)
        
        return s, q_values, model
    
    def training_op(self, q_network_weights):
        action = tf.placeholder(tf.int64, [None])
        y = tf.placeholder(tf.float32, [None])

        # Convert action to one hot vector. shape=(BATCH_SIZE, num_actions)
        action_one_hot = tf.one_hot(action, self.num_actions, 1.0, 0.0)
        #shape = (BATCH_SIZE,)
        q_value = tf.reduce_sum(tf.multiply(self.q_values, action_one_hot), reduction_indices=1)

        # Clip the error, the loss is quadratic when the error is in (-1, 1), and linear outside of that region
        error = tf.abs(y - q_value)
        quadratic_part = tf.clip_by_value(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = tf.reduce_mean(0.5 * tf.square(quadratic_part) + linear_part)

        optimizer = tf.train.RMSPropOptimizer(learning_rate, momentum=momentum, epsilon=epsilon_optimizer)
        grad_update = optimizer.minimize(loss, var_list=q_network_weights)

        return action, y, loss, grad_update

        
    def get_action(self, s):
        action = self.repeated_action
        if self.t % act_interval == 0:
            if self.epsilon >= random.random() or self.t < initial_replay_size:
                action = random.randrange(self.num_actions)
            else:
                action = np.argmax(self.q_values.eval(feed_dict={self.s: [np.float32(s)]}))
            self.repeated_action = action
        return action

    def test_get_action(self, s):
        action = self.repeated_action
        if self.t % act_interval == 0:
            if test_epsilon >= random.random():
                action = random.randrange(self.num_actions)
            else:
                action = np.argmax(self.q_values.eval(feed_dict={self.s: [np.float32(s)]}))
            self.repeated_action = action
        return action

    def load_network(self):
        checkpoint = tf.train.get_checkpoint_state(save_path)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print('Successfully loaded: ' + checkpoint.model_checkpoint_path)
        else:
            print('Training new network...')
        
    def run(self, s, action, R, terminal, s_):

        self.total_reward += R

        R = np.sign(R)

        self.replay_memory.append((s, action, R, s_, terminal))
        if len(self.replay_memory) > replay_memory_size:
            self.replay_memory.popleft()
            
        if self.t >= initial_replay_size:
            if self.t % train_interval == 0:
                self.train()

            if self.t % target_update_interval == 0:
                self.sess.run(self.update_target_network)

            # Save network
            if self.t % save_interval == 0:
                path = self.saver.save(self.sess, save_path + '/' + ENV_NAME, global_step=(self.t))
                print('Successfully saved: ' + path)

        self.total_max_q += np.max(self.q_values.eval(feed_dict={self.s: [np.float32(s)]}))
        self.duration += 1          
  
        if terminal:
            #Debug
            elapsed = time.time() - self.start
            if self.t < initial_replay_size:
                mode = 'random'
            elif initial_replay_size <= self.t < initial_replay_size + exploration_steps:
                mode = 'explore'
            else:
                mode = 'exploit'
            
            text = 'EPISODE: {0:6d} / TIMESTEP: {1:8d} / DURATION: {2:5d} / EPSILON: {3:.5f} / TOTAL_REWARD: {4:3.2f} / AVG_MAX_Q: {5:2.4f} / AVG_LOSS: {6:.5f} / MODE: {7} / STEP_PER_SECOND: {8:.1f}'.format(
                self.episode + 1, self.t, self.duration, self.epsilon,
                self.total_reward, self.total_max_q / float(self.duration),
                self.total_loss / (float(self.duration) / float(train_interval)), mode, self.duration/elapsed)
            print(text)

            with open('fx_output.txt','a') as f:
                f.write(text+"\n")

            self.total_reward = 0
            self.total_max_q = 0
            self.total_loss = 0
            self.duration = 0
            self.episode += 1

        self.t += 1
        if self.epsilon > epsilon_fin and self.t >= initial_replay_size:
            self.epsilon -= self.epsilon_step
            
    def train(self):
        s_batch = []
        action_batch = []
        R_batch = []
        next_s_batch = []
        terminal_batch = []
        y_batch = []

        # Sample random minibatch of transition from replay memory
        minibatch = random.sample(self.replay_memory, batch_size)
        for data in minibatch:
            s_batch.append(data[0])
            action_batch.append(data[1])
            R_batch.append(data[2])
            next_s_batch.append(data[3])
            terminal_batch.append(data[4])

        # Convert True to 1, False to 0
        terminal_batch = np.array(terminal_batch) + 0
        target_q_values_batch = self.target_q_values.eval(feed_dict={self.st: np.float32(np.array(next_s_batch))})
        y_batch = R_batch + (1 - terminal_batch) * gamma * np.max(target_q_values_batch, axis=1)
        loss, _ = self.sess.run([self.loss, self.grad_update], feed_dict={
            self.s: np.float32(np.array(s_batch)),
            self.action: action_batch,
            self.y: y_batch
        })

        self.total_loss += loss

        


def load_chart(target, start, end, dates=None):

    if target=='fx':
        start = '20180103' if start==0 else start
        end = '20190424' if end==0 else end
        return load_fx(start, end, dates)


    elif 'gold' in target or 'silver' in target:

        return load_material(start, end, material=target, dates=dates)

    elif target in ['wpm']:

        return load_stock(start, end, stock=target, dates=dates)

def load_fx(start="20180103", end=None):
    f = open("USDJPY20010102_20180626.txt")
    line = f.readline()
    line = f.readline()

    price = []
    while line:
        if start in line:
            print('from',start)
            while line:
                price.append(float(line.split(",")[-2]))
                if end in line:
                    print('to', end)
                    break
                line = f.readline()
        line = f.readline()
    f.close()
    return price



def load_material(start, end, material='goldUSD', dates=None):
    # d = datetime.datetime.strptime(start, '%Y%m%d')
    # start = d.strftime('%Y/%-m/%-d')
    # d = datetime.datetime.strptime(end, '%Y%m%d')
    # end = d.strftime('%Y/%-m/%-d')
    if material=='goldJPY':
        df = pd.read_csv('goldJPY_19950403_20190424.csv',names=('date','price'))
    elif material=='goldUSD':
        df = pd.read_csv('goldUSD_19950102_20190515.csv',names=('date','price'))
    elif material=='silverUSD':
        df = pd.read_csv('silverUSD_19950102_20190515.csv',names=('date','price'))
    else:
        assert False, 'No chart'

    df['price'] = df['price'].astype(str)

    # change date
    for i in range(len(df)):
        # print(i)
        d = df['date'][i]
        date = datetime.datetime.strptime(d, '%Y/%m/%d')
        df['date'][i] = date.strftime('%Y%m%d')

    if dates:
        df = df[df['date'].isin(dates)]

    start_idx = df[df['date']==start].index[0] if start else 0
    end_idx = df[df['date']==end].index[0] if end else len(df)-1
    start_loc = df.index.get_loc(start_idx)
    end_loc = df.index.get_loc(end_idx)
    df = df[end_loc:start_loc+1]
    price = df['price']
    chart = price.str.replace(',','').astype(float)
    chart = list(chart/max(chart))
    dates = list(df['date'])

    return chart, dates

def load_stock(start, end, stock, dates=None):
    if stock == 'wpm':
        df = pd.read_csv('WPM_20090520_20190520.csv')
    df['date'] = df['date'].astype(str)

    if dates:
        df = df[df['date'].isin(dates)]


    start_idx = df[df['date']==start].index[0] if start else 0
    end_idx = df[df['date']==end].index[0] if end else len(df)-1
    start_loc = df.index.get_loc(start_idx)
    end_loc = df.index.get_loc(end_idx)
    df = df[start_loc:end_loc+1]
    price = df['close'].astype(str)
    chart = price.str.replace(',','').astype(float)
    chart = list(chart/max(chart))
    dates = list(df['date'])
    return chart, dates



def plot_pips(rewards):
    rewards = np.array(rewards)
    plt.plot(rewards)
    plt.savefig('pips')
    plt.show()


def convert_action(action):
    if action==0:
        return -1
    elif action==1:
        return 0
    elif action==2:
        return 1
    # elif action==3:
    #     return 1
    # else:
    #     return 3

def run_test(env, agent, picture=0):
    terminal = False
    s = env.reset()
    rewards = [0]
    while not terminal:
        action = agent.test_get_action(s)
        s_, R, terminal = env.step(action)
        rewards.append(rewards[-1]+R)
        # agent.run(s, action, R, terminal, s_)
        s = s_

    if picture:
        plot_pips(rewards)
    return rewards

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=str, default='20140101')
    parser.add_argument('--end', type=str, default='20190515')
    parser.add_argument('--target', type=str, default='goldUSD')
    parser.add_argument('--features', nargs='*', default=[])

    args = parser.parse_args()
    print(vars(args))


    print("Data Loading...")
    # all len : 6155990, 2010 - len : 3091974
    # chart = load_chart()
    chart, dates = load_chart(args.target, args.start, args.end)
    num_train = int(len(chart)*0.9)
    features = []
    features_test = []
    for feature in args.features:
        print(feature)
        feature_chart, _ = load_chart(feature, args.start, args.end, dates=dates)
        features.append(feature_chart[:num_train])
        features_test.append(feature_chart[num_train:])

    print("End!!")

    # print(max(chart), max(features[0]))
    print(np.array(features).shape)
    print(np.array(features_test).shape)
    print("num_train :", num_train)
    env = Env(chart[:num_train], features, len_input, spread)
    env_test = Env(chart[num_train:], features_test, len_input, spread)
    agent = Agent(env.action_space.n, len_input, args)
    if TRAIN:
        for _ in range(num_episodes):
            agent.start = time.time()
            terminal = False
            s = env.reset()
            while not terminal:
                action = agent.get_action(s)
                action = convert_action(action)
                s_, R, terminal = env.step(action)
                agent.run(s, action, R, terminal, s_)
                s = s_
            rewards = run_test(env_test,agent)
            print('Test reward:', rewards[-1])

    # env_test = Env(chart[num_train:], len_input, spread)
    #for _ in range(10):
    terminal = False
    s = env_test.reset()
    rewards = [0]

    while not terminal:
        action = agent.test_get_action(s)
        s_, R, terminal = env_test.step(action)
        rewards.append(rewards[-1]+R)
        agent.run(s, action, R, terminal, s_)
        s = s_
    plot_pips(rewards)

if __name__ == '__main__':
    main()
