#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author:   zhangkai
Created:  2022-06-18 11:36:00
"""

import matplotlib.pyplot as plt
import numpy as np


def visualize_learning_rate(scheduler, epochs):
    lrs = [[] for _ in range(len(scheduler.optimizer.param_groups))]
    for epoch in range(epochs):
        for lst, dct in zip(lrs, scheduler.optimizer.param_groups):
            lst.append(dct['lr'])
        scheduler.step()

    lists = []
    for lr in lrs:
        lists.append(list(range(epochs)))
        lists.append(lr)
    plt.figure(figsize=(12, 4))
    lines = plt.plot(*lists)
    plt.setp(lines[0], linewidth=3)
    plt.title('Learning rate change')
    plt.xlabel('Epochs')
    plt.ylabel('Learning rate')
    plt.xticks(np.arange(0, epochs + 1, 10), rotation=45)
    plt.yticks(np.arange(0.01, 0.11, 0.01))
    plt.grid(True)
    plt.show()
