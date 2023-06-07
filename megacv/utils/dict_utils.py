#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author:   zhangkai
Created:  2022-04-18 11:26:41
"""

import collections


class Dict(dict):

    """Use properties to access key values in the dictionary

    if the key not in dict, return `None`
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            self.__setitem__(key, value)

    def dict(self):
        return DictUnwrapper(self)

    def __delattr__(self, key):
        try:
            del self[key]
        except Exception:
            pass

    def __getattr__(self, key):
        try:
            return self[key]
        except Exception:
            pass

    def __setitem__(self, key, value):
        super().__setitem__(key, DictWrapper(value))

    __setattr__ = __setitem__

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __or__(self, other):
        if not isinstance(other, (Dict, dict)):
            return NotImplemented
        new = Dict(other)
        new.update(self)
        return new

    def __ior__(self, other):
        self.update(other)
        return self


class DefaultDict(collections.defaultdict):

    def __delattr__(self, key):
        try:
            del self[key]
            return True
        except Exception:
            return False

    def __getattr__(self, key):
        return self[key]


def DictWrapper(*args, **kwargs):
    if args and len(args) == 1:
        if isinstance(args[0], collections.defaultdict):
            return DefaultDict(args[0].default_factory, args[0])
        elif isinstance(args[0], dict):
            return Dict(args[0])
        elif isinstance(args[0], (tuple, list)):
            return type(args[0])(map(DictWrapper, args[0]))
        else:
            return args[0]
    elif args:
        return type(args)(map(DictWrapper, args))
    else:
        return Dict(**kwargs)


def DictUnwrapper(doc):
    if isinstance(doc, DefaultDict):
        return collections.defaultdict(doc.default_factory, doc)
    if isinstance(doc, Dict):
        return dict(map(lambda x: (x[0], DictUnwrapper(x[1])), doc.items()))
    if isinstance(doc, (tuple, list)):
        return type(doc)(map(DictUnwrapper, doc))
    return doc
