import pickle


def pad(x, max_len=50):
    if len(x) > max_len:
        x = x[-max_len:]
    else:
        pad_num = max_len - len(x)
        x = [0] * pad_num + x
    return x


if __name__ == '__main__':
    # dataset_name = 'Steam'
    # dataset_name = 'Movielens'
    # dataset_name = 'Beauty'
    dataset_name = 'Video_Games'
    max_len = 50
    with open(dataset_name + 'infor.pkl', 'rb') as f:
        review_df = pickle.load(f)
    with open(dataset_name + 'infor_size.pkl', 'rb') as f:
        user_count, item_count = pickle.load(f)
    print(review_df.head())
    train_set = []
    test_set = []
    for reviewerID, col in review_df.groupby('reviewerID'):
        behaviour = col["asin"].tolist()
        if len(behaviour) == 1:
            continue
        if len(behaviour) < 3:
            seq = pad(behaviour[:-1])
            pos = pad(behaviour[1:])
            target = 0
            sample_train = [seq, pos, behaviour, target]
            train_set.append(sample_train)
        else:
            seq = pad(behaviour[:-2])
            pos = pad(behaviour[1:-1])
            target = 0
            sample_train = [seq, pos, behaviour[:-1], target]
            train_set.append(sample_train)
            seq = pad(behaviour[:-1])
            pos = pad(behaviour[1:])
            target = behaviour[-1]
            sample_test = [seq, pos, behaviour, target]
            test_set.append(sample_test)

    print(len(train_set), len(test_set))
    with open(dataset_name + '.pkl', 'wb') as f:
        pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)
