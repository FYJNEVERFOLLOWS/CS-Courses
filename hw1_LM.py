# homework1: Language Model
import time

import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import utils
from torch.utils.data import DataLoader

train_data_path = "/Users/fuyanjie/Desktop/PG/AI/NLP/exp_hw_ZeweiChu/bobsue.lm.train.txt"
dev_data_path = "/Users/fuyanjie/Desktop/PG/AI/NLP/exp_hw_ZeweiChu/bobsue.lm.dev.txt"
test_data_path = "/Users/fuyanjie/Desktop/PG/AI/NLP/exp_hw_ZeweiChu/bobsue.lm.test.txt"

vocab_path = "/Users/fuyanjie/Desktop/PG/AI/NLP/exp_hw_ZeweiChu/bobsue.voc.txt"
device = torch.device('cpu')


# design model
class LSTM(nn.Module):
    def __init__(self, embedding_size, vocab_size, output_size):
        super(LSTM, self).__init__()
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size)
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=embedding_size, num_layers=1,batch_first=True)  # utilize the LSTM model in torch.nn
        self.fc = nn.Linear(embedding_size, output_size)  # add fully-connected layer to reshape the output

    def forward(self, a):
        # a.shape: torch.Size([64, 20])
        a_embedded = self.embed(a.long())
        # a_embedded.shape: torch.Size([64, 20, 200])

        # the input shape of self.lstm should be (N,L,input_size)
        x, self.hidden = self.lstm(input=a_embedded)
        # x.shape: torch.Size([64, 20, 200])
        output = self.fc(x)
        # output.shape: torch.Size([64, 20, 1499])

        return output

vocab_size = 1499
# embedding_size = hidden_size = 200
model = LSTM(embedding_size=200, vocab_size=vocab_size, output_size=vocab_size)

# construct loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, verbose=True)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

train_data = DataLoader(utils.BobSue_Dataset(train_data_path), batch_size=64,
                        shuffle=True, num_workers=0)  # train_data is a tuple: (batch_x, batch_y)
dev_data = DataLoader(utils.BobSue_Dataset(dev_data_path), batch_size=64,
                    shuffle=True, num_workers=0)  # test_data is a tuple: (batch_x, batch_y)
test_data = DataLoader(utils.BobSue_Dataset(test_data_path), batch_size=64,
                    shuffle=True, num_workers=0)  # test_data is a tuple: (batch_x, batch_y)

with open(vocab_path, "r") as f:
    text = f.read()
f.close()
vocab = text.split('\n')
dict_word2idx, dict_idx2word, vocab_size = utils.build_dict(vocab[:-1])
print(f'dict_word2idx {dict_word2idx}')
print(f'dict_idx2word {dict_idx2word}')
def main():
    best_loss = 9999999.0
    for epoch in range(10):
        # Train
        running_loss = 0.0
        iter = 0

        # training cycle forward, backward, update
        model.train()
        for (batch_x, batch_y) in train_data:
            # 获得一个批次的数据和标签(inputs, labels)
            batch_x = batch_x.to(device)  # batch_x.shape [B, seq_len] [B, 20]
            batch_y = batch_y.to(device)  # batch_y.shape [B, seq_len] [B, 20]

            # 获得模型预测结果
            output = model(batch_x)  # output.shape [B, seq_len, vocab_size]
            label = batch_y[:, -1] # label.shape [B]
            pred = output[:, -1, :] # pred.shape torch.Size([B, vocab_size])
            # 代价函数
            loss = criterion(pred, label)  # averaged loss on batch_y

            running_loss += loss.item()
            if iter % 1000 == 0:
                now_loss = running_loss / 1000
                if now_loss < best_loss:
                    # 模型保存
                    # torch.save(model, './model_save/best_model_epoch%d_loss_%f.pth' % (epoch, loss.item()))
                    best_loss = now_loss
                # scheduler.step(now_loss)
                print('[%d, %5d] loss: %.5f' % (epoch + 1, iter + 1, now_loss), flush=True)
                running_loss = 0.0

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 一个iter以一个batch为单位
            iter += 1
        # scheduler.step()
        # torch.cuda.empty_cache()

        # Validate
        cnt_acc = 0
        total = 0
        wrong_pred = {}
        with torch.no_grad():
            model.eval()
            for (batch_x, batch_y) in dev_data:
                batch_x = batch_x.to(device)  # batch_x.shape [B, seq_len] [B, 20]
                batch_y = batch_y.to(device)  # batch_y.shape [B, seq_len] [B, 20]
                # batch_y.shape[0] = batch_size
                total += batch_y.size(0)

                # 获得模型预测结果
                output = model(batch_x)  # output.shape [B, seq_len, vocab_size]

                label = batch_y[:, -1].numpy()
                pred = output[:, -1, :].numpy()
                pred = np.argmax(pred, axis=1)
                cnt_acc += np.sum(label == pred)
                wrong_pred_indices = np.where(label != pred)[0]
                for idx in wrong_pred_indices:
                    pair = (pred[idx], label[idx]) # type of key: tuple
                    wrong_pred[pair] = wrong_pred.get(pair, 0) + 1

        print(f'cnt_acc {cnt_acc} total {total}')
        print('accuracy on val set: %.2f %% ' % (100.0 * cnt_acc / total), flush=True)
        print(f'wrong_pred \n {wrong_pred}')
        # Evaluate
        cnt_acc = 0
        total = 0
        with torch.no_grad():
            model.eval()
            for (batch_x, batch_y) in test_data:
                batch_x = batch_x.to(device)  # batch_x.shape [B, seq_len] [B, 20]
                batch_y = batch_y.to(device)  # batch_y.shape [B, seq_len] [B, 20]
                # batch_y.shape[0] = batch_size
                total += batch_y.size(0)

                # 获得模型预测结果
                output = model(batch_x)  # output.shape [B, seq_len, vocab_size]

                label = batch_y[:, -1].numpy()
                pred = output[:, -1, :].numpy()
                pred = np.argmax(pred, axis=1)
                cnt_acc += np.sum(label == pred)
                wrong_pred_indices = np.where(label != pred)[0]
                for idx in wrong_pred_indices:
                    pair = (pred[idx], label[idx]) # type of key: tuple
                    wrong_pred[pair] = wrong_pred.get(pair, 0) + 1
        print(f'cnt_acc {cnt_acc} total {total}')
        print('accuracy on test set: %.2f %% ' % (100.0 * cnt_acc / total), flush=True)
        ordered_wrong_pred = sorted(wrong_pred.items(), key=lambda x:x[1], reverse=True)
        print(f'ordered_wrong_pred[:35] \n {ordered_wrong_pred[:35]}')

        for idx, w_pred in enumerate(ordered_wrong_pred[:35]):
            print(f'Rank {idx + 1}')
            print(f'wrong pred {dict_idx2word[w_pred[0][0]]} label {dict_idx2word[w_pred[0][1]]} times {w_pred[1]}')



if __name__ == '__main__':
    main()


