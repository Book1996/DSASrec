import numpy as np
import torch
import torch.utils.data
import random


def collate_fn(batch):
    batch = np.array(batch, dtype=object)

    batch_seq = torch.LongTensor(batch[:, 0].tolist())
    batch_pos = torch.LongTensor(batch[:, 1].tolist())
    batch_neg = torch.LongTensor(batch[:, 2].tolist())
    batch_target = torch.LongTensor(batch[:, 3].tolist())
    return [batch_seq, batch_pos, batch_neg, batch_target]


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, max_len, dataset, item_num):
        self.dataset = dataset
        self.max_len = max_len
        self.len = len(self.dataset)
        self.item_num = item_num
        self.all_set = set([i for i in range(1,  self.item_num + 1)])

    def __getitem__(self, index):
        seq = self.dataset[index][0]  # 1
        pos = self.dataset[index][1]
        rated = set(self.dataset[index][2])
        neg = random.sample(self.all_set - rated, self.max_len)
        target = self.dataset[index][3]
        return [seq, pos, neg, target]

    def __len__(self):
        return self.len


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, max_len, dataset, item_num):
        self.dataset = dataset
        self.max_len = max_len
        self.len = len(self.dataset)
        self.item_num = item_num
        self.all_set = set([i for i in range(1,  self.item_num + 1)])

    def __getitem__(self, index):
        seq = self.dataset[index][0]  # 1
        pos = self.dataset[index][1]
        rated = set(self.dataset[index][2])
        neg = random.sample(self.all_set - rated, 100)
        target = self.dataset[index][3]
        return [seq, pos, neg, target]

    def __len__(self):
        return self.len
