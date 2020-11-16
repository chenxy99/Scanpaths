import copy
import os
from typing import Any, Dict, Optional, Union, Type

import torch
from torch import nn, optim

import json
from os.path import join

class RecordManager(object):
    def __init__(
        self,
        serialization_dir: str,
        filename_prefix: str = "history_record",
    ):

        self._serialization_dir = serialization_dir
        self._filename_prefix = filename_prefix
        self.record = dict()
        self.record['epoch'] = -1
        self.record['iteration'] = -1
        self.record['best_metric'] = None

    def load(self):
        with open(join(self._serialization_dir, self._filename_prefix + '.json'), 'r') as f:
            self.record = json.load(f)

    def save(self, epoch, iteration, best_metric):
        self.record['epoch'] = epoch
        self.record['iteration'] = iteration
        self.record['best_metric'] = best_metric
        with open(join(self._serialization_dir, self._filename_prefix + '.json'), 'w') as f:
            json.dump(self.record, f, indent=2)

    def init_record(self):
        with open(join(self._serialization_dir, self._filename_prefix + '.json'), 'w') as f:
            json.dump(self.record, f, indent=2)

    def get_epoch(self):
        return self.record['epoch']

    def get_iteration(self):
        return self.record['iteration']

    def get_best_metric(self):
        return self.record['best_metric']
