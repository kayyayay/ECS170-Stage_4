import torch
import torch.nn as nn
import numpy as np

from code.base_class.method import method


class Method_RNN_Generation(method, nn.Module):

    def __init__(self, mName=None, mDescription=None, model_type="RNN"):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.embedding_dim = 128
        self.dropout = nn.Dropout(0.3)
        self.hidden_dim = 256
        self.lr = 1e-3
        self.epochs = 50
        self.batch_size = 128
        self.model_type = model_type

        self.loss_fn = nn.CrossEntropyLoss()

    # BUILD MODEL
    # -------------------
    def build(self, vocab_size):
        self.embedding = nn.Embedding(vocab_size, self.embedding_dim)

        if self.model_type == 'RNN':
            self.rnn = nn.RNN(
                self.embedding_dim,
                self.hidden_dim,
                batch_first=True
            )

        elif self.model_type == 'LSTM':
            self.rnn = nn.LSTM(
                self.embedding_dim,
                self.hidden_dim,
                batch_first=True
            )

        elif self.model_type == 'GRU':
            self.rnn = nn.GRU(
                self.embedding_dim,
                self.hidden_dim,
                batch_first=True
            )

        self.fc = nn.Linear(self.hidden_dim, vocab_size)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    # -------------------
    # FORWARD
    # -------------------
    def forward(self, x):

        x = self.embedding(x)

        if self.model_type == 'LSTM':
            out, (h, c) = self.rnn(x)
        else:
            out, h = self.rnn(x)

        out = out[:, -1, :]  # (batch, hidden)

        out = self.fc(out)  # (batch, vocab)

        return out

    # -------------------
    # TRAIN
    # -------------------
    def train_model(self, X, y):

        X = torch.LongTensor(X)
        y = torch.LongTensor(y)

        self.train()

        for epoch in range(self.epochs):

            perm = torch.randperm(X.size(0))

            total_loss = 0

            for i in range(0, X.size(0), self.batch_size):

                idx = perm[i:i+self.batch_size]
                xb, yb = X[idx], y[idx]

                self.optimizer.zero_grad()

                pred = self.forward(xb)
                loss = self.loss_fn(pred, yb)

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

    # -------------------
    # TEXT GENERATION
    # -------------------
    def generate(self, start_words, vocab, idx2word, max_len=30):

        self.eval()

        words = start_words.lower().split()
        seq = [vocab.get(w, 1) for w in words]

        for _ in range(max_len):

            x = torch.LongTensor(seq[-20:]).unsqueeze(0)

            with torch.no_grad():
                out = self.forward(x)
                probs = torch.softmax(out, dim=1)
                next_word = torch.multinomial(probs[0], 1).item()

            seq.append(next_word)

            if next_word == 0:
                break

        return " ".join([idx2word[i] for i in seq])