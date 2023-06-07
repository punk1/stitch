#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""The Stopwatch utility.

Author:   zhangkai
Created:  2022-04-06 17:03:32
"""

import time


class Stopwatch:
    """The matlab-style Stopwatch."""

    def __init__(self, start=False):
        """Initialize.

        Args:
          start (bool): whether start the clock immediately.
        """
        self.beg = None
        self.end = None
        self.duration = 0.0

        if start:
            self.tic()

    def tic(self):
        """Start the Stopwatch."""
        self.beg = time.time()
        self.end = None

    def toc(self):
        """Stop the Stopwatch."""
        if self.beg is None:
            raise RuntimeError("Please run tic before toc.")
        self.end = time.time()
        return self.end - self.beg

    def toc2(self):
        """Record duration, and Restart the Stopwatch."""
        delta = self.toc()
        self.tic()
        return delta

    def acc(self):
        """Accumulates the duration."""
        delta = self.toc()
        self.duration += delta
        self.tic()
        return delta

    def reset(self):
        """Reset the whole Stopwatch."""
        self.tic()
        self.duration = 0.0
