# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from . import BaseWrapperDataset


class PrependTokenDataset(BaseWrapperDataset):

    def __init__(self, dataset, token=None, label_token=None):
        super().__init__(dataset)
        self.token = token
        self.label_token = label_token
        if token is not None:
            self._sizes = np.array(dataset.sizes) + 1
        else:
            self._sizes = dataset.sizes

        if label_token is not None:
            self._sizes = self._sizes + 1

    def __getitem__(self, idx):
        item = self.dataset[idx]
        if self.label_token is not None:
            item = torch.cat([item.new([self.label_token]), item])
        if self.token is not None:
            item = torch.cat([item.new([self.token]), item])
        return item

    @property
    def sizes(self):
        return self._sizes

    def num_tokens(self, index):
        n = self.dataset.num_tokens(index)
        if self.token is not None:
            n += 1
        if self.label_token is not None:
            n += 1
        return n

    def size(self, index):
        n = self.dataset.size(index)
        if self.token is not None:
            n += 1
        if self.label_token is not None:
            n += 1
        return n
