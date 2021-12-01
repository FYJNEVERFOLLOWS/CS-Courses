import numpy as np
from collections import Counter

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
# torch.set_printoptions(profile="full")

def build_dict(words, max_words=50000):
    word_count = Counter()
    for w in words:
        word_count[w] += 1
    ls = word_count.most_common(max_words)
    num_words = len(ls) + 1
    # return word2idx, idx2word and num_words, respectively
    return {w[0]: index+1 for (index, w) in enumerate(ls)}, {index+1 : w[0] for (index, w) in enumerate(ls)}, num_words

def encode(text, word_to_idx):
    return [word_to_idx.get(t, -1) for t in text]

vocab_path = "/Users/fuyanjie/Desktop/PG/AI/NLP/exp_hw_ZeweiChu/bobsue.voc.txt"
with open(vocab_path, "r") as f:
    text = f.read()
f.close()
vocab = text.split('\n')
dict_word2idx, dict_idx2word, vocab_size = build_dict(vocab[:-1])
print(dict_word2idx)
print(dict_idx2word)

class BobSue_Dataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path
        self.total_x, self.total_y = self._read_file()

    def __getitem__(self, index):
        return torch.tensor(self.total_x[index], dtype=torch.long), torch.tensor(self.total_y[index], dtype=torch.long).squeeze() # dtype is torch.long

    def __len__(self):
        return len(self.total_x)

    def _read_file(self):
        total_x = []
        total_y = []
        feats, labels = self.load_data(self.data_path)
        print(f'max_len {self.max_len}')
        for idx, feat in enumerate(feats):
            feat_encoded = encode(feat, word_to_idx=dict_word2idx)
            feat_encoded = F.pad(torch.tensor(feat_encoded, dtype=torch.long), (20 - len(feat_encoded), 0)) # 填充至序列长度为20
            # feat_one_hot = F.one_hot(feat_encoded, num_classes=vocab_size)
            # feat_one_hot.shape: torch.Size([20, 1499])
            # feat_encoded.shape: torch.Size([20])
            total_x.append(feat_encoded)
            # label_encoded: [idx], label_encoded.shape: [1]
            label_encoded = encode(labels[idx], word_to_idx=dict_word2idx)
            label_encoded = F.pad(torch.tensor(label_encoded, dtype=torch.long), (20 - len(label_encoded), 0)) # 填充至序列长度为20
            # label_one_hot = F.one_hot(torch.tensor(label_encoded, dtype=torch.long), num_classes=vocab_size)
            # label_one_hot.shape: torch.Size([1, 1499])
            # label_encoded.shape: torch.Size([20])
            total_y.append(label_encoded)

        return total_x, total_y

    def load_data(self, path):
        feats = []
        labels = []
        self.max_len = 0
        with open(path, "r") as f:
            text = f.read()
        f.close()
        sentences = text.split('\n')[:-1]
        for sen in sentences:
            words = sen.split(' ')
            self.max_len = max(self.max_len, len(words))
            for i in range(1, len(words)):
                feats.append(words[:i])
                labels.append(words[1:i+1])
        return feats, labels

class Prevsent_Dataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path
        self.total_x, self.total_y = self._read_file()

    def __getitem__(self, index):
        return torch.tensor(self.total_x[index], dtype=torch.long), torch.tensor(self.total_y[index], dtype=torch.long).squeeze() # dtype is torch.long

    def __len__(self):
        return len(self.total_x)

    def _read_file(self):
        total_x = []
        total_y = []
        feats, labels = self.load_data(self.data_path)
        print(f'max_len {self.max_len}')
        for idx, feat in enumerate(feats):
            feat_encoded = encode(feat, word_to_idx=dict_word2idx)
            feat_encoded = F.pad(torch.tensor(feat_encoded, dtype=torch.long), (20 - len(feat_encoded), 0)) # 填充至序列长度为20
            # feat_one_hot = F.one_hot(feat_encoded, num_classes=vocab_size)
            # feat_one_hot.shape: torch.Size([20, 1499])
            # feat_encoded.shape: torch.Size([20])
            total_x.append(feat_encoded)
            # label_encoded: [idx], label_encoded.shape: [1]
            label_encoded = encode(labels[idx], word_to_idx=dict_word2idx)
            label_encoded = F.pad(torch.tensor(label_encoded, dtype=torch.long), (20 - len(label_encoded), 0)) # 填充至序列长度为20
            # label_one_hot = F.one_hot(torch.tensor(label_encoded, dtype=torch.long), num_classes=vocab_size)
            # label_one_hot.shape: torch.Size([1, 1499])
            # label_encoded.shape: torch.Size([20])
            total_y.append(label_encoded)

        return total_x, total_y

    def load_data(self, path):
        feats = []
        labels = []
        self.max_len = 0
        with open(path, "r") as f:
            text = f.read()
        f.close()
        sentences = text.split('\n')[:-1]
        for sen in sentences:
            words = sen.split(' ')
            self.max_len = max(self.max_len, len(words))
            for i in range(1, len(words)):
                feats.append(words[:i])
                labels.append(words[1:i+1])
        return feats, labels

# if __name__ == '__main__':
#     train_txt_path = "/Users/fuyanjie/Desktop/PG/AI/NLP/exp_hw_ZeweiChu/bobsue.lm.train.txt"
#     # feat_one_hot = F.one_hot(torch.tensor([1, 2, 4]), num_classes=5)
#     # print(feat_one_hot)
#     # print(feat_one_hot.shape)
#     # pad = nn.ZeroPad2d((0,0,0,20-feat_one_hot.size(0)))
#     # feat_one_hot = pad(feat_one_hot)
#     # print(feat_one_hot)
#     # print(feat_one_hot.shape)
#
#     train_data = DataLoader(BobSue_Dataset(train_txt_path), batch_size=4, shuffle=True,
#                             num_workers=0)  # train_data.shape (batch_x, batch_y)
#     print(len(train_data))  # len(train_data) is samples / batch_size
#     print(next(iter(train_data))[0].shape, next(iter(train_data))[1].shape)