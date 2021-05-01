'''
This script handling the training process.
'''
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import numpy as np
from tqdm import tqdm


class Session(object):
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def train(self, dataloader, optimizer):
        mean_loss = 0
        for i, batch_samples in enumerate(tqdm(dataloader)):
            optimizer.zero_grad()
            for j in range(len(batch_samples)):
                batch_samples[j] = batch_samples[j].to(self.device)
            pos_loss, neg_loss, score = self.model.loss(batch_samples)
            loss = (pos_loss + neg_loss).mean()
            loss.backward()
            optimizer.step()
            mean_loss = mean_loss + (loss.item() - mean_loss) / (i + 1)
        return mean_loss

    def test(self, dataloader):
        NDCG = 0.0
        HT = 0.0
        valid_user = 0.0
        total_test = 10000
        self.model.eval()
        for i, batch_samples in enumerate(tqdm(dataloader)):
            for j in range(len(batch_samples)):
                batch_samples[j] = batch_samples[j].to(self.device)
            pos_loss, neg_loss, predictions, attn_weight = self.model.prediction(batch_samples)
            predictions = -predictions.cpu().detach().numpy()
            rank = predictions.argsort(1).argsort(1)[:, 0]
            T, _ = predictions.shape
            valid_user += T
            for j in rank:
                if j < 10:
                    NDCG += 1 / np.log2(j + 2)
                    HT += 1
            if valid_user > total_test:
                break
        self.model.train()
        result = {'NDCG': NDCG / valid_user, 'HT': HT / valid_user}
        return result
