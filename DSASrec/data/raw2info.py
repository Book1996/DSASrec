import pickle
import pandas as pd
from random import sample
import random
import numpy as np
import csv

def to_df_amz(file_path):
    with open(file_path, 'r') as fin:
        df = {}
        i = 0
        for line in fin:
            df[i] = eval(line)
            i += 1
        df = pd.DataFrame.from_dict(df, orient='index')
        return df


def to_df_stm(file_path):
    with open(file_path, 'r') as fin:
        df = {}
        i = 0
        for line in fin:
            df[i] = eval(line)
            i += 1
        df = pd.DataFrame.from_dict(df, orient='index')
        df.rename(columns={'username': 'reviewerID', 'product_id': 'asin', 'date': 'unixReviewTime'}, inplace=True)
        return df


def to_df_mvl(file_path):
    names = ['reviewerID', 'asin', 'rating', 'unixReviewTime']
    df = pd.read_csv(pre_name + dataset_name + '.dat', sep='::', engine='python', header=None, names=names)
    return df


def build_map(df, col_name):
    key = sorted(df[col_name].unique().tolist())
    m = dict(zip(key, range(len(key))))
    df[col_name] = df[col_name].map(lambda x: m[x] + 1)
    return m, key


if __name__ == '__main__':
    # dataset_name = 'Beauty'
    # dataset_name = 'Steam'
    # dataset_name = 'Video_Games'
    dataset_name = 'Movielens'
    pre_name = 'reviews_'
    # suf_name = '.json'
    suf_name = '.dat'
    review_df = to_df_mvl(pre_name + dataset_name + suf_name)
    # Remove redundant columns
    review_df = review_df[['reviewerID', 'asin', 'unixReviewTime']]
    # cold start up
    print(review_df.head())
    filter_u = []
    filter_i = []
    for reviewerID, col in review_df.groupby('reviewerID'):
        if len(col['asin']) < 5:
            filter_u.append(reviewerID)
    for asin, col in review_df.groupby('asin'):
        if len(col['reviewerID']) < 5:
            filter_i.append(asin)
    review_df = review_df[~review_df['reviewerID'].isin(filter_u)]
    review_df = review_df[~review_df['asin'].isin(filter_i)]
    print(review_df.shape)
    # duplicates
    review_df = review_df.drop_duplicates(subset=['reviewerID', 'asin'], keep="first")
    print(review_df.shape)
    #   ont_hot
    asin_map, asin_key = build_map(review_df, 'asin')
    revi_map, revi_key = build_map(review_df, 'reviewerID')
    review_df = review_df.sort_values(['reviewerID', 'unixReviewTime'])
    review_df = review_df.reset_index(drop=True)
    #   get the attributte of dataset
    user_count, item_count, example_count = len(revi_map), len(asin_map), review_df.shape[0]
    print(user_count, item_count, example_count)
    with open(dataset_name + 'infor.pkl', 'wb') as f:
        pickle.dump(review_df, f, pickle.HIGHEST_PROTOCOL)
    with open(dataset_name + 'infor_size.pkl', 'wb') as f:
        pickle.dump((user_count, item_count), f, pickle.HIGHEST_PROTOCOL)
