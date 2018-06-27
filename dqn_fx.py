#!/usr/bin/env python
# -*- coding: utf-8 -*-


import csv
import random
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense
import numpy as np
from collections import deque
import tensorflow as tf
import time
from env_fx import Env_FX

num_episodes = 10000
initial_replay_size = 10000
batch_size = 32
replay_memory_size = 200000
act_interval = 1
train_interval = 1
target_update_interval = 10000


epsilon_optimizer = 1.5e-7
momentum = 0.95
learning_rate = 0.00025


epsilon_init = 1.0
epsilon_fin = 0.1
exploration_steps = 100000
gamma = 0.99

len_input = 10
TRAIN = True


class Agent:
    def __init__(self, num_actions, len_input):
        self.num_actions = num_actions
        self.len_input = len_input+1
        
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


        self.sess.run(tf.global_variables_initializer())


        # Initialize target network
        self.sess.run(self.update_target_network)
        
        
    def network(self):
        model = Sequential()
        model.add(Dense(10, activation='relu', input_dim=self.len_input))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(self.num_actions, activation='linear'))
        
        s = tf.placeholder(tf.float32, [None, self.len_input])
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
        
    def run(self, s, action, R, terminal, s_):
        R = np.sign(R)
        
        self.replay_memory.append((s, action, R, s_, terminal))
        if len(self.replay_memory) > replay_memory_size:
            self.replay_memory.popleft()
            
        if self.t >= initial_replay_size:
            if self.t % train_interval == 0:
                self.train()

            if self.t % target_update_interval == 0:
                self.sess.run(self.update_target_network)

        self.total_reward += R
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
            
            text = 'EPISODE: {0:6d} / TIMESTEP: {1:8d} / DURATION: {2:5d} / EPSILON: {3:.5f} / TOTAL_REWARD: {4:3.0f} / AVG_MAX_Q: {5:2.4f} / AVG_LOSS: {6:.5f} / MODE: {7} / STEP_PER_SECOND: {8:.1f}'.format(
                self.episode + 1, self.t, self.duration, self.epsilon,
                self.total_reward, self.total_max_q / float(self.duration),
                self.total_loss / (float(self.duration) / float(train_interval)), mode, self.duration/elapsed)
            print(text)

            with open('fx_output.txt','a') as f:
                f.write(text)

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

        


def load_chart():
    f = open("USDJPY.csv",encoding="shiftjis")
    reader = csv.reader(f)
    next(reader)

    price = []
    for row in reader:
        price.append(float(row[4]))
    f.close()
    return price

def main():
    chart = load_chart()
    env = Env_FX(chart, len_input)
    agent = Agent(env.action_space.n, len_input)
    if TRAIN:
        for _ in range(num_episodes):
            agent.start = time.time()
            terminal = False
            s = env.reset()
            while not terminal:
                action = agent.get_action(s)
                s_, R, terminal = env.step(action)
                #print(terminal)
                agent.run(s, action, R, terminal, s_)
                s = s_


if __name__ == '__main__':
    main()
