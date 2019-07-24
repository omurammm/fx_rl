#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
#  Train
#  $
#
#  Test
#  $ python dqn.py --TRAIN 0 --LOAD 1 --save_name plat_1step/plat_1step-3600000
#
#
#


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
import glob
from SumTree import SumTree

# from logging import getLogger, StreamHandler, DEBUG, FileHandler
# logger = getLogger('testtest')
# handler = StreamHandler()
# handler.setLevel(DEBUG)
# logger.setLevel(DEBUG)
# logger.addHandler(handler)
# fh = FileHandler('test.log')
# logger.addHandler(fh)
# logger.propagate = False
#
# logger.debug('hello')

#-------------------- MEMORY --------------------------
class Memory:   # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.max_p = 1

    def _getPriority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, sample):
        p = self._getPriority(error)
        self.tree.add(p, sample)

    def add_p(self, p, sample):
        self.tree.add(p, sample)


    def sample(self, n):
        batch = []
        idx_batch = []
        segment = self.tree.total() / n

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append(data)
            idx_batch.append(idx)

        return batch, idx_batch

    def update(self, idx, error):
        p = self._getPriority(error)
        if p > self.max_p:
            self.max_p = p
        self.tree.update(idx, p)



class Agent:
    def __init__(self,
                 args,
                 num_actions,
                 # frame_width = 84,  # Resized frame width
                 # frame_height = 84,  # Resized frame height
                 # state_length = 4,  # Number of most recent frames to produce the input to the network
                 anealing_steps = 2000000, # Number of steps over which the initial value of epsilon is linearly annealed to its final value
                 initial_epsilon = 1.0,  # Initial value of epsilon in epsilon-greedy
                 final_epsilon = 0.05,  # Final value of epsilon in epsilon-greedy
                 epsilon_test = 0.05,
                 target_update_interval = 6000,  # The frequency with which the target network is updated
                 act_interval = 1,  # The agent sees only every () input
                 train_interval = 1,  # The agent selects 4 actions between successive updates
                 batch_size = 32,  # Mini batch size
                 lr = 0.00025,  # Learning rate used by RMSProp
                 # MOMENTUM = 0.95  # Momentum used by RMSProp
                 # MIN_GRAD = 0.01  # Constant added to the squared gradient in the denominator of the RMSProp update
                 # save_interval = 300000,  # The frequency with which the network is saved
                 # no_op_steps = 30,  # Maximum number of "do nothing" actions to be performed by the agent at the start of an episode
                 # initial_beta = 0.4,
                 ):

        self.prioritized = args.prioritized
        self.double = args.double
        # self.dueling = args.dueling
        self.n_step = args.n_step

        self.initial_memory_size = args.initial_memory_size
        self.replay_memory_size = args.replay_memory_size
        self.gamma = args.gamma
        self.gamma_n = args.gamma ** args.n_step



        self.num_actions = num_actions
        self.len_input = args.len_input
        self.num_features = 2+len(args.features)

        # self.frame_width = frame_width
        # self.frame_height = frame_height
        # self.state_length = state_length
        self.anealing_steps = anealing_steps
        self.target_update_interval = target_update_interval
        self.act_interval = act_interval
        self.train_interval = train_interval
        self.batch_size = batch_size
        self.lr = lr
        # self.no_op_steps = no_op_steps

        self.epsilon = initial_epsilon
        self.epsilon_fin = final_epsilon
        self.epsilon_test = epsilon_test
        self.anealing_by_step = (initial_epsilon - final_epsilon) / anealing_steps
        # self.beta = initial_beta
        # self.beta_step = (1 - initial_epsilon) / args.num_episodes
        self.t = 0
        self.repeated_action = 0

        self.save_name = args.save_name
        self.save_path = args.save_path
        self.save_interval = args.save_interval





        # Parameters used for summary
        self.total_reward = 0
        self.total_max_q = 0
        self.total_loss = 0
        self.duration = 0
        self.episode = 0

        self.start = 0

        # Create replay memory
        #self.replay_memory = deque()

        if self.prioritized:
            self.memory = Memory(args.replay_memory_size)
        else:
            self.memory = deque()

        self.buffer = []
        self.R = 0

        # Dueling Network
        # if self.dueling:
        #     # Create q network
        #     self.s, self.q_values, q_network = self.build_dueling_network()
        #     q_network_weights = q_network.trainable_weights
        #
        #     # Create target network
        #     self.st, self.target_q_values, target_network = self.build_dueling_network()
        #     target_network_weights = target_network.trainable_weights
        #
        # else:

        # Create q network
        self.s, self.q_values, q_network = self.build_network()
        q_network_weights = q_network.trainable_weights

        # Create target network
        self.st, self.target_q_values, target_network = self.build_network()
        target_network_weights = target_network.trainable_weights

        # Define target network update operation
        self.update_target_network = [target_network_weights[i].assign(q_network_weights[i]) for i in range(len(target_network_weights))]

        # Define loss and gradient update operation
        self.a, self.y, self.error, self.loss, self.grad_update = self.build_training_op(q_network_weights)

        self.sess = tf.InteractiveSession()


        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(q_network_weights)


        if args.LOAD:
            self.load_network()


        # Initialize target network
        self.sess.run(self.update_target_network)

        
    def build_network(self):
        # model = Sequential()
        # model.add(Dense(400, activation='relu', input_dim=self.len_input))
        # model.add(Dense(300, activation='relu'))
        # model.add(Dense(self.num_actions, activation='linear'))

        chart = Input(shape=(self.num_features,self.len_input))
        lstm = LSTM(3)(chart)
        # output = Dense(3)(lstm)

        s = tf.placeholder(tf.float32, [None, self.num_features, self.len_input])
        model = Model(inputs=chart, outputs=lstm)
        q_values = model(s)
        
        return s, q_values, model

    def huber_loss(self, x, delta=1.0):
        return tf.where(
            tf.abs(x) < delta,
            tf.square(x) * 0.5,
            delta * (tf.abs(x) - 0.5 * delta)
        )

    def build_training_op(self, q_network_weights):
        a = tf.placeholder(tf.int64, [None])
        y = tf.placeholder(tf.float32, [None])
        #w = tf.placeholder(tf.float32, [None])

        # Convert action to one hot vector. shape=(BATCH_SIZE, num_actions)
        a_one_hot = tf.one_hot(a, self.num_actions, 1.0, 0.0)
        # shape = (BATCH_SIZE,)
        q_value = tf.reduce_sum(tf.multiply(self.q_values, a_one_hot), reduction_indices=1)

        # Clip the error, the loss is quadratic when the error is in (-1, 1), and linear outside of that region
        td_error = y - q_value
        errors = self.huber_loss(td_error)
        loss = tf.reduce_mean(errors)
        # error_is = (w / tf.reduce_max(w)) * error

        optimizer = tf.train.RMSPropOptimizer(self.lr, momentum=0.95, epsilon=0.01)
        grad_update = optimizer.minimize(loss, var_list=q_network_weights)

        return a, y, tf.abs(td_error), loss, grad_update

        
    def get_action(self, s):
        action = self.repeated_action
        if self.t % self.act_interval == 0:
            if self.epsilon >= random.random() or self.t < self.initial_memory_size:
                action = random.randrange(self.num_actions)
            else:
                action = np.argmax(self.q_values.eval(feed_dict={self.s: [np.float32(s)]}))
            self.repeated_action = action
        return action

    def test_get_action(self, s):
        action = self.repeated_action
        if self.t % self.act_interval == 0:
            if self.epsilon_test >= random.random():
                action = random.randrange(self.num_actions)
            else:
                action = np.argmax(self.q_values.eval(feed_dict={self.s: [np.float32(s)]}))
            self.repeated_action = action
        return action

    def load_network(self):
        if os.path.isdir(self.save_path):
            checkpoint = tf.train.get_checkpoint_state(self.save_path)
            if checkpoint and checkpoint.model_checkpoint_path:
                self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
                print('Successfully loaded: ' + checkpoint.model_checkpoint_path)
            else:
                print('Training new network...')
        else:
            self.saver.restore(self.sess, self.save_path)
            print('Successfully loaded: ' + self.save_path)



        
    def run(self, state, action, reward, terminal, next_state):
        # print(self.t,"\r")
        # print(reward)

        # Clip all positive rewards at 1 and all negative rewards at -1, leaving 0 rewards unchanged
        raw_reward = reward
        reward = np.sign(reward)

        if (not self.prioritized) and len(self.memory) > self.replay_memory_size:
            self.memory.popleft()

        #if self.t < INITIAL_REPLAY_SIZE:
        #self.memory.add(1, (state, action, reward, next_state, terminal))
        self.buffer.append((state, action, reward, next_state, terminal))

        self.R = (self.R + reward * self.gamma_n) / self.gamma
        # print(self.R, reward, action)
        #print(self.memory.tree.tree[199:])
        #print(self.memory.max_p)
        if self.t < self.initial_memory_size:
            if terminal:      # terminal state
                while len(self.buffer) > 0:
                    n = len(self.buffer)
                    s, a, r, s_, done= self.get_sample(n)
                    if self.prioritized:
                        self.memory.add_p(self.memory.max_p, (s, a, r, s_, done))
                    else:
                        self.memory.append((s, a, r, s_, done))
                    self.R = (self.R - self.buffer[0][2]) / self.gamma
                    self.buffer.pop(0)
                self.R = 0

            if len(self.buffer) >= self.n_step:
                s, a, r, s_, done = self.get_sample(self.n_step)
                if self.prioritized:
                    self.memory.add_p(self.memory.max_p, (s, a, r, s_, done))
                else:
                    self.memory.append((s, a, r, s_, done))
                self.R = self.R - self.buffer[0][2]
                self.buffer.pop(0)



        if self.t >= self.initial_memory_size:

            if terminal:      # terminal state
                while len(self.buffer) > 0:
                    n = len(self.buffer)
                    s, a, r, s_, done= self.get_sample(n)
                    if self.prioritized:
                        self.memory.add_p(self.memory.max_p, (s, a, r, s_, done))
                    else:
                        self.memory.append((s, a, r, s_, done))
                    self.R = (self.R - self.buffer[0][2]) / self.gamma
                    self.buffer.pop(0)
                self.R = 0

            if len(self.buffer) >= self.n_step:
                s, a, r, s_, done = self.get_sample(self.n_step)
                if self.prioritized:
                    self.memory.add_p(self.memory.max_p, (s, a, r, s_, done))
                else:
                    self.memory.append((s, a, r, s_, done))
                self.R = self.R - self.buffer[0][2]
                self.buffer.pop(0)

            # Train network
            if self.t % self.train_interval == 0:
                self.train_network()

            # Update target network
            if self.t % self.target_update_interval == 0:
                print("update",len(self.sess.run(self.update_target_network)))

            # Save network
            if self.t % self.save_interval == 0:
                path = self.saver.save(self.sess, self.save_path+'/'+self.save_name, global_step=(self.t))
                print('Successfully saved: ' + path)

        self.total_reward += raw_reward
        self.total_max_q += np.max(self.q_values.eval(feed_dict={self.s: [np.float32(state)]}))
        self.duration += 1
###

        if terminal:
            #Debug
            elapsed = time.time() - self.start
            if self.t < self.initial_memory_size:
                mode = 'random'
            elif self.initial_memory_size <= self.t < self.initial_memory_size + self.anealing_steps:
                mode = 'explore'
            else:
                mode = 'exploit'
            
            text = 'EPISODE: {0:6d} / TIMESTEP: {1:8d} / DURATION: {2:5d} / EPSILON: {3:.5f} / TOTAL_REWARD: {4:3.2f} / AVG_MAX_Q: {5:2.4f} / AVG_LOSS: {6:.5f} / MODE: {7} / STEP_PER_SECOND: {8:.1f}'.format(
                self.episode + 1, self.t, self.duration, self.epsilon,
                self.total_reward, self.total_max_q / float(self.duration),
                self.total_loss / (float(self.duration) / float(self.train_interval)), mode, self.duration/elapsed)
            print(text)

            # with open('log/{}.txt'.format(self.save_name),'a') as f:
            f_log.write(text+"\n")

            self.total_reward = 0
            self.total_max_q = 0
            self.total_loss = 0
            self.duration = 0
            self.episode += 1

        self.t += 1
        if self.epsilon > self.epsilon_fin and self.t >= self.initial_memory_size:
            self.epsilon -= self.anealing_by_step


    def train_network(self):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        terminal_batch = []
        w_batch = []

        if self.prioritized:
            minibatch, idx_batch = self.memory.sample(self.batch_size)
        else:
            minibatch = random.sample(self.memory, self.batch_size)

        for data in minibatch:
            state_batch.append(data[0])
            action_batch.append(data[1])
            reward_batch.append(data[2])
            #shape = (BATCH_SIZE, 4, 32, 32)
            next_state_batch.append(data[3])
            terminal_batch.append(data[4])

        # Convert True to 1, False to 0
        terminal_batch = np.array(terminal_batch) + 0
        target_q_values_batch = self.target_q_values.eval(feed_dict={self.st: np.float32(np.array(next_state_batch))})

        # DDQN
        if self.double:
            actions = np.argmax(self.q_values.eval(feed_dict={self.s: np.float32(np.array(next_state_batch))}), axis=1)
            target_q_values_batch = np.array([target_q_values_batch[i][action] for i, action in enumerate(actions)])
            y_batch = reward_batch + (1 - terminal_batch) * self.gamma_n * target_q_values_batch
        else:
            y_batch = reward_batch + (1 - terminal_batch) * self.gamma * np.max(target_q_values_batch, axis=1)

        # IS weight
        #for idx in idx_batch:
        #    wi = (NUM_REPLAY_MEMORY * self.memory.tree.tree[idx])**(-self.beta)
        #    w_batch.append(wi)

        error_batch = self.error.eval(feed_dict={
            self.s: np.float32(np.array(state_batch)),
            self.a: action_batch,
            self.y: y_batch
        })

        #error_is_batch = self.error_is.eval(feed_dict={
        #   self.a: action_batch,
        #    self.y: y_batch,
        #    self.w: w_batch
        #})

        # Memory update
        if self.prioritized:
            for i in range(self.batch_size):
                self.memory.update(idx_batch[i],error_batch[i])

        loss, _ = self.sess.run([self.loss, self.grad_update], feed_dict={
            self.s: np.float32(np.array(state_batch)),
            self.a: action_batch,
            self.y: y_batch
            #self.w: w_batch
        })

        # print(reward_batch)


        self.total_loss += loss

    def get_sample(self, n):
        s, a, _, _, _ = self.buffer[0]
        _, _, _, s_, done = self.buffer[n-1]

        return s, a, self.R, s_, done





def load_chart(target, start, end, dates=None):

    if target=='fx':
        start = '20180103' if start==0 else start
        end = '20190424' if end==0 else end
        return load_fx(start, end, dates)

    elif 'gold' in target or 'silver' in target:
        return load_material(start, end, material=target, dates=dates)

    elif target in ['wpm']:
        return load_stock(start, end, stock=target, dates=dates)

    elif target.isdecimal():
        return load_tocom(start, end, commodity_num=int(target), dates=dates)

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
    # chart = list(chart*100/max(chart))
    chart = list(chart)
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
    # chart = list(chart/max(chart))
    chart = list(chart)
    dates = list(df['date'])
    return chart, dates

def get_dateAndprice(commodity_num):
    csvs = sorted(glob.glob('tocom_data/*/*'))
    cols = ('date', 'deal-type', 'commodity', 'contract-month', 'strike-price', 'open', 'high', 'low', 'close', 'settlement-price', 'volume', 'turnover')
    df_return = pd.DataFrame()
    for csv_f in csvs:
        df = pd.read_csv(csv_f, names=cols)
        df_com = df[df['commodity']==commodity_num]
        df_add = df_com[['date', 'open', 'high', 'low', 'close']]
        df_return = pd.concat([df_return, df_add])
    df_return = df_return.dropna()
    df_return = df_return.drop_duplicates(subset='date')
    df_return = df_return.reset_index(drop=True)
    return df_return

def load_tocom(start, end, commodity_num=18, dates=None):
    print('Loading TOCOM data...')
    df = get_dateAndprice(commodity_num)
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
    # chart = list(chart/max(chart))
    chart = list(chart)
    dates = list(df['date'])

    # pad 0 with 1 before value
    for i in range(len(chart)):
        if chart[i] ==0:
            chart[i]=chart[i-1]
    if 0 in chart:
        assert False, 'Data includes 0'
    print(commodity_num,'len :',len(chart))
    return chart, dates





def plot_pips(rewards, filename):
    rewards = np.array(rewards)
    for i in range(len(rewards)):
        plt.plot(rewards[i])
    plt.savefig(filename)
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

    parser.add_argument('--TRAIN', type=int, default=1)
    parser.add_argument('--LOAD', type=int, default=0)
    parser.add_argument('--save_interval', type=int, default=200000)
    parser.add_argument('--save_name', type=str, default='test')

    parser.add_argument('--target', type=str, default='11')
    parser.add_argument('--features', nargs='*', default=[])
    parser.add_argument('--start', type=str, default='19980106')
    parser.add_argument('--end', type=str, default='20181010')

    parser.add_argument('--spread', type=float, default=0)
    parser.add_argument('--len_input', type=int, default=100)
    parser.add_argument('--test_split', type=float, default=0.05)
    parser.add_argument('--series', type=str, default='diff')


    parser.add_argument('--prioritized', type=int, default=1, help='prioritized experience replay')
    parser.add_argument('--double', type=int, default=1, help='Double-DQN')
    # parser.add_argument('--dueling', type=int, default=1, help='Dueling Network')
    parser.add_argument('--n_step', type=int, default=3, help='n step bootstrap target')
    # parser.add_argument('--env_name', type=str, default='Alien-v0', help='Environment of Atari2600 games')
    # parser.add_argument('--train', type=int, default=1, help='train mode or test mode')
    # parser.add_argument('--gui', type=int, default=0, help='decide whether you use GUI or not')
    # parser.add_argument('--load', type=int, default=0, help='loading saved network')
    # parser.add_argument('--network_path', type=str, default=0, help='used in loading and saving (default: \'saved_networks/<env_name>\')')
    parser.add_argument('--replay_memory_size', type=int, default=100000, help='replay memory size')
    parser.add_argument('--initial_memory_size', type=int, default=50000, help='Learner waits until replay memory stores this number of transition')
    parser.add_argument('--num_episodes', type=int, default=800, help='number of episodes each agent plays')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')



    args = parser.parse_args()



    print(vars(args))




    args.save_path = 'saved_networks/{}'.format(args.save_name)


    if not args.LOAD:
        if (os.path.exists(args.save_path) or os.path.exists('log/{}.txt'.format(args.save_name))):
            assert False, 'the file exists'

        os.makedirs('log', exist_ok=True)
        os.makedirs(args.save_path, exist_ok=True)


    global f_log
    f_log_name = 'log/{}.txt'.format(args.save_name)
    os.makedirs(os.path.dirname(f_log_name), exist_ok=True)
    f_log = open(f_log_name, 'w')
    f_log.write(str(vars(args))+'\n\n')

    print("Data Loading...")
    print('target:', args.target)
    chart, dates = load_chart(args.target, args.start, args.end)
    num_train = int(len(chart)*(1-args.test_split))
    features = []
    features_test = []
    for feature in args.features:
        print('feature:', feature)
        feature_chart, _ = load_chart(feature, args.start, args.end, dates=dates)
        features.append(feature_chart[:num_train])
        features_test.append(feature_chart[num_train-args.len_input+1:])

    print("End!!")

    # print(max(chart), max(features[0]))

    print("num_train :", num_train)
    env = Env(chart[:num_train], features, args.len_input, args.spread, series=args.series)
    env_test = Env(chart[num_train-args.len_input+1:], features_test, args.len_input, args.spread, series=args.series)
    agent = Agent(args, env.action_space.n)
    if args.TRAIN:
        for _ in range(args.num_episodes):
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
            # with open('log/{}.txt'.format(args.save_name),'a') as f:
            f_log.write('Test reward: '+str(rewards[-1])+"\n")


    # env_test = Env(chart[num_train:], len_input, spread)
    #for _ in range(10):


    rewards_all = []

    for _ in range(10):
        rewards = [0]
        s = env_test.reset()
        terminal = False
        while not terminal:
            action = agent.test_get_action(s)
            s_, R, terminal = env_test.step(action)
            rewards.append(rewards[-1]+R)
            agent.run(s, action, R, terminal, s_)
            s = s_
        rewards_all.append(rewards)
    plot_pips(rewards_all, 'log/'+args.save_name)

if __name__ == '__main__':
    main()
