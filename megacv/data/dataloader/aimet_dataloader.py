#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author:   zhangkai
Created:  2023-01-10 14:40:08
"""


class AimetDataLoader:

    def __init__(self, dataloaders, maxlen=1000):
        self.dataloaders = dataloaders
        self.step = 0
        self.maxlen = maxlen

    def reset(self):
        self.step = 0
        for dataloader in self.dataloaders:
            dataloader.reset()

    def __len__(self):
        return self.maxlen

    def __iter__(self):
        return self

    def __next__(self):
        if self.step > self.maxlen:
            raise StopIteration

        for i, dataloader in enumerate(self.dataloaders):
            if self.step % len(self.dataloaders) == i:
                data = dataloader.get_batch()
                if data is not None:
                    self.step += 1
                    return data["img"]

        raise StopIteration
