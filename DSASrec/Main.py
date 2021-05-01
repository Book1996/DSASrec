"""
======================
@author:Book
@time:2021/3/5:13:39
======================
"""
import torch
from Dataset import TrainDataset, collate_fn, TestDataset
from torch.utils.data import DataLoader
import pickle
import os
import numpy as np
from Models import DSASRec
from Session import Session
import csv
seed = 2021
if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = '7'
    USE_CUDA = torch.cuda.is_available()

    # dataset_name = 'Steam'
    dataset_name = 'Movielens'
    # dataset_name = 'Beauty'
    # dataset_name = 'Video_Games'
    with open('data/' + dataset_name + '.pkl', 'rb') as f:
        train_set = pickle.load(f)
        test_set = pickle.load(f)
    with open('data/' + dataset_name + 'infor_size.pkl', 'rb') as f:
        user_count, item_count = pickle.load(f)
    print(user_count, item_count)
    model_name = "SASrec"
    max_len = 50
    ffn_dim = 50
    emb_dim = 50
    learning_rate = 0.001
    epoch = 500
    layers = 2
    dropout = [0.5, 0.0]
    batch_size = 64
    decay = 1e-6
    if dataset_name == 'Steam':
        dropout = [0.5, 0.0]
    if dataset_name == 'Movielens':
        dropout = [0.2, 0.0]
        max_len = 200
        ffn_dim = 50
    load = False
    save = True
    exp_name = 'ffb_dim' + str(ffn_dim) + '_dp' + str(dropout)+ '_emb_dim' + str(emb_dim) + '_decay' + str(decay) + '_' + dataset_name + "_" + model_name + '_l' + str(
        layers)
    print(exp_name)
    train_data = TrainDataset(max_len, train_set, item_count)
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=5, shuffle=True, collate_fn=collate_fn)
    test_data = TestDataset(max_len, test_set, item_count)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=5, shuffle=True, collate_fn=collate_fn)
    device = torch.device('cuda' if USE_CUDA else 'cpu')
    model = DSASRec(max_len, item_count, emb_dim, ffn_dim, layers, dropout).to(device)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if load:
        model.load_state_dict(torch.load(exp_name + '_parameter.pkl', map_location=torch.device('cpu')))
    session = Session(model, device)
    with open('log/'+dataset_name+'/'+exp_name + '_log.csv', 'w+', newline='') as csvFile:
        csv_writer = csv.writer(csvFile)
        max_HT = 0
        csv_writer.writerow(['epoch', 'loss', 'recall', 'ndcg'])
        for e in range(epoch):
            loss = session.train(train_loader, optimizer)
            print("epoch: {}, loss: {}".format(e, loss))
            ret = session.test(test_loader)
            print(ret)
            row = [e, loss, ret['HT'], ret['NDCG']]
            csv_writer.writerow(row)
            if max_HT < ret['HT'] and save:
                max_HT = ret['HT']
                torch.save(model.state_dict(), exp_name + '_parameter.pkl')




