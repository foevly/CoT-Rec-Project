import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset


class ItemSequenceDataset(Dataset):
    def __init__(self, filepath, max_length):
        df = pd.read_csv(filepath,  # 或 filepath 变量
                         encoding="utf-8-sig",  # 去掉可能的 BOM
                         dtype=str)
        df.rename(columns=lambda c: c.strip(), inplace=True)  # 列名去空格

        for col in ["user_id", "item_id"]:
            df[col] = (df[col].astype(str).str.strip())  # 去首尾空格
            df[col] = pd.to_numeric(df[col], errors="coerce")  # 非数字→NaN
        df = df.dropna(subset=["user_id", "item_id"]).copy()
        df[["user_id", "item_id"]] = df[["user_id", "item_id"]].astype("int64")
        # df = pd.read_csv(filepath, names=['user_id', 'item_id']).astype("int64")
        self.num_users, self.num_items = df['user_id'].max() + 1, df['item_id'].max() + 1
        
        self.all_records = [[] for _ in range(self.num_users)]
        for _, row in tqdm(df.iterrows()):
            user_id, item_id = row.iloc[0], row.iloc[1]
            self.all_records[user_id].append(item_id)
        
        print('# Users:', self.num_users)
        print('# Items:', self.num_items)
        print('# Interactions:', len(df))

        lens = [len(s) for s in self.all_records]
        print("min_len:", min(lens), "  #short(<2):", sum(l < 2 for l in lens))

        X_train, y_train = [], []
        X_valid, y_valid = [], []
        X_test, y_test = [], []

        for seq in tqdm(self.all_records):
            train_seq = seq[:-2]
            if len(train_seq) < max_length:
                X_train.append((max_length - len(train_seq) + 1) * [self.num_items] + train_seq[:-1])
                y_train.append((max_length - len(train_seq) + 1) * [self.num_items] + train_seq[1:])
            else:
                for i in range(len(train_seq) - max_length):
                    X_train.append(train_seq[i:i+max_length])
                    y_train.append(train_seq[i+1:i+max_length+1])

            valid_seq = seq[:-1]
            if len(valid_seq) - 1 < max_length:
                X_valid.append((max_length - len(valid_seq) + 1) * [self.num_items] + valid_seq[:-1])
            else:
                X_valid.append(valid_seq[-(max_length+1):-1])
            y_valid.append(valid_seq[-1])

            test_seq = seq
            if len(test_seq) - 1 < max_length:
                X_test.append((max_length - len(test_seq) + 1) * [self.num_items] + test_seq[:-1])
            else:
                X_test.append(test_seq[-(max_length+1):-1])
            y_test.append(test_seq[-1])

        self.X_train, self.y_train = torch.tensor(X_train), torch.tensor(y_train)
        self.X_valid, self.y_valid = torch.tensor(X_valid), torch.tensor(y_valid)
        self.X_test, self.y_test = torch.tensor(X_test), torch.tensor(y_test)

        print('Data loading completed.')

    def __len__(self):
        return len(self.X_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]
