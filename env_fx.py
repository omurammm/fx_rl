#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random

class Env_FX:
    def __init__(self, chart, len_input):
        self.chart = self._arrange_chart(chart, len_input)
        self.action_space = ActionSpace(3)
        self.len_input = len_input

        self.idx = 0
        self.position = 0

    def _arrange_chart(self, chart, len_input):
        chart_input = []
        for i in range(len(chart)-len_input+1):
            chart_input.append(chart[i:i+len_input])
        return chart_input

    def reset(self):
        self.idx = 0
        self.position = 0
        return self.chart[0]

    def step(self, action):
        reward = 0
        price = self.chart[self.idx][-1]
        # buy
        if action == 1:
            if self.position < 0:
                reward = int((-self.position - price)*100)
                self.position = 0
            elif self.position == 0:
                self.position = price
        # sell
        elif action == 2:
            if self.position > 0:
                reward = int((price - self.position)*100)
                self.position = 0
            elif self.position == 0:
                self.position = -price

        self.idx += 1
        if self.idx == len(self.chart):
            return (self.chart[self.idx-1], reward, True)
        else:
            return (self.chart[self.idx], reward, False)



class ActionSpace:
    def __init__(self, n):
        self.n = n

    def sample(self):
        return random.randrange(self.n)




