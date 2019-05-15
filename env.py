#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import numpy as np

class Env:
    def __init__(self, chart, len_input, spread, action_type='move', reward_type='unrealized', positions=[-1,0,1]):
        self.chart_raw, self.chart_diff = self._arrange_chart(chart, len_input)
        self.action_space = ActionSpace(len(positions))
        self.len_input = len_input
        self.spread = spread

        self.idx = 0
        self.position = 0
        self.position_price = 0

        # 'move' or 'do'  : only 'move' now
        self.action_type = action_type

        # 'unrealized' or 'realized'
        self.reward_type = reward_type

        self.positions = positions

    def _arrange_chart(self, chart, len_input):
        chart_diff_ = list(np.array(chart[1:]) - np.array(chart[:-1]))
        chart = chart[1:]
        chart_raw = []
        chart_diff = []
        for i in range(len(chart)-len_input+1):
            chart_raw.append(chart[i:i+len_input])
            chart_diff.append(chart_diff_[i:i+len_input])
        return chart_raw, chart_diff

    # def deal(self, to_position, price):
    #     self.position = to_position
    #     self.position_price = price

    def reset(self):
        self.idx = 0
        self.position = 0
        return self.chart_diff[0] + [0]


    def step(self, action):

        pre_position = self.position
        # action_type == 'move'
        self.position = action

        pre_price = self.chart_raw[self.idx][-2]
        price = self.chart_raw[self.idx][-1]

        post_price = self.chart_raw[self.idx+1][-1] if (self.idx+1)!=len(self.chart_raw) else price

        if self.reward_type == 'unrealized':
            commission = np.abs(self.position-pre_position) * self.spread
            reward = (post_price - price) * self.position - commission

        elif self.reward_type == 'realized':
            if self.position - pre_position == 0:
                reward = 0
            else:
                # TODO: 買い増し、売り増しどうしよう、とりあえず一旦決済してから増す
                # if (np.sign(self.position)==np.sign(pre_position)) and (np.abs(self.position)-np.abs(pre_position)>0):

                commission = np.abs(pre_position) * self.spread
                reward = (price - self.position_price) * pre_position - commission

                self.position_price = price

        self.idx += 1
        if self.idx == len(self.chart_raw):
            return (self.chart_diff[self.idx-1]+[self.position], reward, True)
        else:
            return (self.chart_diff[self.idx]+[self.position], reward, False)



    # def step_legacy(self, action):
    #     reward = 0
    #     price = self.chart_raw[self.idx][-1]
    #     # buy : 1
    #     if action == 1:
    #         if self.position == 0:
    #             self.deal(1, price)
    #
    #         elif self.position == 1:
    #             pass
    #
    #         elif self.position == 3:
    #             reward = (price-self.spread - self.position_price) * 2
    #             self.position = 1
    #             #self.deal(1, price)
    #
    #         # position : -1 or -3 => 1
    #         # $2 sell
    #         elif self.position < 0:
    #             reward = (price - self.position_price) * self.position
    #             self.deal(1, price)
    #
    #     # buy : 3
    #     if action == 2:
    #         if self.position == 0:
    #             self.deal(3, price)
    #
    #         elif self.position == 1:
    #             # TODO: 買い増し、売り増しどうしよう、とりあえず一旦決済してから増す
    #             reward = (price-self.spread - self.position_price) * self.position
    #             self.deal(3, price)
    #
    #         elif self.position == 3:
    #             pass
    #             # self.position = 1
    #             # reward = (price - self.position_price) * 2
    #
    #         # position : -1 or -3 => 1
    #         elif self.position < 0:
    #             reward = (price - self.position_price) * self.position
    #             self.deal(3, price)
    #
    #     # sell : -1
    #     if action == 3:
    #         price -= self.spread
    #
    #         if self.position == 0:
    #             self.deal(-1, price)
    #
    #         elif self.position > 0:
    #             reward = (price - self.position_price) * self.position
    #             self.deal(-1, price)
    #
    #         elif self.position == -1:
    #             pass
    #         elif self.position == -3:
    #             reward = (price+self.spread - self.position_price) * (-2)
    #             self.position = -1
    #             #self.deal(-1, price)
    #
    #
    #     # sell : -3
    #     if action == 4:
    #         price -= self.spread
    #         if self.position == 0:
    #             self.deal(-3, price)
    #
    #         elif self.position > 0:
    #             reward = (price - self.position_price) * self.position
    #             self.deal(-3, price)
    #
    #         elif self.position == -1:
    #             # TODO: 買い増し、売り増しどうしよう、とりあえず一旦決済してから増す
    #             reward = (price+self.spread - self.position_price) * self.position
    #             self.deal(-3, price)
    #
    #         elif self.position == -3:
    #             pass
    #
    #     # stay
    #     if action == 0:
    #         if self.position == 0:
    #             pass
    #
    #         elif self.position < 0:
    #             reward = (price - self.position_price) * self.position
    #             self.deal(0, 0)
    #
    #         elif self.position > 0:
    #             reward = (price-self.spread - self.position_price) * self.position
    #             self.deal(0, 0)
    #
    #     self.idx += 1
    #     if self.idx == len(self.chart_raw):
    #         return (self.chart_diff[self.idx-1]+[self.position], reward, True)
    #     else:
    #         return (self.chart_diff[self.idx]+[self.position], reward, False)
    #
    #


            # def step(self, action):
    #     reward = 0
    #     price = self.chart_raw[self.idx][-1]
    #     # buy
    #     if action == 1:
    #         if self.position < 0:
    #             reward = -self.position - price
    #             self.position = 0
    #         elif self.position == 0:
    #             self.position = price
    #             #reward = - self.spread
    #     # sell
    #     elif action == 2:
    #         if self.position > 0:
    #             reward = (price - self.spread) - self.position
    #             self.position = 0
    #         elif self.position == 0:
    #             self.position = - (price - self.spread)
    #             #reward = - self.spread
    #
    #     self.idx += 1
    #     if self.idx == len(self.chart_raw):
    #         return (self.chart_diff[self.idx-1]+[np.sign(self.position)], reward, True)
    #     else:
    #         return (self.chart_diff[self.idx]+[np.sign(self.position)], reward, False)



class ActionSpace:
    def __init__(self, n):
        self.n = n

    def sample(self):
        return random.randrange(self.n)




