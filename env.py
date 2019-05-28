#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import numpy as np

class Env:
    def __init__(self,
                 chart,
                 feature_charts,
                 len_input,
                 spread,
                 action_type='move',
                 reward_type='unrealized',
                 positions=[-1,0,1],
                 raw_or_diff='diff'):

        self.chart_raw, self.chart_diff = self._arrange_chart(chart, len_input)
        self.feature_charts = []
        self.raw_or_diff = raw_or_diff
        for feature in feature_charts:
            raw, diff = self._arrange_chart(feature, len_input)
            if raw_or_diff=='raw':
                self.feature_charts.append(raw)
            elif raw_or_diff=='diff':
                self.feature_charts.append(diff)

        if raw_or_diff=='raw':
            self.chart = self.chart_raw
        elif raw_or_diff=='diff':
            self.chart = self.chart_diff

        self.feature_charts = np.array(self.feature_charts)

        self.action_space = ActionSpace(len(positions))

        self.len_input = len_input
        self.spread = spread

        self.idx = 0
        self.position = 0
        self.position_price = 0
        self.terminal = False

        # 'move' or 'do'  : only 'move' now
        self.action_type = action_type

        # 'unrealized' or 'realized'
        self.reward_type = reward_type

        self.positions = positions

        self.position_history = np.zeros(len_input)

    def _arrange_chart(self, chart, len_input):
        chart_diff_ = list(np.array(chart[1:]) - np.array(chart[:-1]))
        chart = chart[1:]
        chart_raw = []
        chart_diff = []
        for i in range(len(chart)-len_input+1):
            chart_raw.append(chart[i:i+len_input])
            chart_diff.append(chart_diff_[i:i+len_input])
        return np.array(chart_raw), np.array(chart_diff)

    # def deal(self, to_position, price):
    #     self.position = to_position
    #     self.position_price = price

    def reset(self):
        self.idx = 0
        self.position = 0
        self.terminal = False
        if len(self.feature_charts)!=0:
            state = np.concatenate((np.stack((self.chart[self.idx], self.position_history)),self.feature_charts[:,self.idx,:]))
        else:
            state = np.stack((self.chart[self.idx], self.position_history))
        return state


    def step(self, action):
        pre_position = self.position
        # action_type == 'move'
        self.position = action

        self.position_history = np.concatenate((self.position_history[1:], [self.position]))

        pre_price = self.chart_raw[self.idx][-2]
        price = self.chart_raw[self.idx][-1]

        post_price = self.chart_raw[self.idx+1][-1] if (self.idx+1)!=len(self.chart_raw) else price


        # TODO: 買い売り両方でspreadがかかる場合のみ実装されている。FXには使えない
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


        if self.idx == len(self.chart)-1:
            self.terminal = True
        else:
            self.idx += 1

        if len(self.feature_charts)!=0:
            state = np.concatenate((np.stack((self.chart[self.idx], self.position_history)),self.feature_charts[:,self.idx,:]))
        else:
            state = np.stack((self.chart[self.idx], self.position_history))
        return (state, reward, self.terminal)



class ActionSpace:
    def __init__(self, n):
        self.n = n

    def sample(self):
        return random.randrange(self.n)




