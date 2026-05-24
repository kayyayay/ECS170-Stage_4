import torch
import torch.nn as nn
import numpy as np

from code.base_class.method import method
from code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy

class Method_RNN_Classification(method, nn.Module):

    def __init__(self, mName=None, mDescription=None, model_type = "RNN"):

        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

# The hyperparameters used
        self.learning_rate = 1e-3
        self.max_epoch = 8
        self.batch_size = 128
        self.model_type = model_type
        self.embedding_dim = 128
        self.hidden_dim = 256

        self.loss_list = []
        self.acc_list = []

        # These will be set after dataset loading
        self.vocab_size = None

# Model layers
        self.embedding = None
        self.rnn = None
        self.fc = None

        self.loss_fn = nn.CrossEntropyLoss()

# After the vocab is known, we build the model
    def build_model(self, vocab_size):

        self.vocab_size = vocab_size

        self.dropout = nn.Dropout(0.3)

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=self.embedding_dim,
            padding_idx=0
        )

        if self.model_type == 'RNN':

            self.rnn = nn.RNN(
                input_size=self.embedding_dim,
                hidden_size=self.hidden_dim,
                batch_first=True
            )

        elif self.model_type == 'LSTM':

            self.rnn = nn.LSTM(
                input_size=self.embedding_dim,
                hidden_size=self.hidden_dim,
                batch_first=True
            )

        elif self.model_type == 'GRU':

            self.rnn = nn.GRU(
                input_size=self.embedding_dim,
                hidden_size=self.hidden_dim,
                batch_first=True
            )

        self.fc = nn.Linear(self.hidden_dim, 2)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)


    def forward(self, x):

        x = self.embedding(x)
        x = self.dropout(x)

        if self.model_type == 'LSTM':
            out, (hidden, _) = self.rnn(x)
        else:
            out, hidden = self.rnn(x)

        hidden = hidden[-1]
        out = self.fc(hidden)

        return out

# Train the model
    def train_model(self, X, y):

        X_t = torch.LongTensor(np.array(X))
        y_t = torch.LongTensor(np.array(y))

        self.train()

        evaluator = Evaluate_Accuracy('train_acc', '')

        for epoch in range(self.max_epoch):

            perm = torch.randperm(X_t.size(0))

            epoch_loss = 0
            batches = 0

            for i in range(0, X_t.size(0), self.batch_size):

                idx = perm[i:i + self.batch_size]

                X_b = X_t[idx]
                y_b = y_t[idx]

                self.optimizer.zero_grad()

                outputs = self.forward(X_b)
                loss = self.loss_fn(outputs, y_b)

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                batches += 1

            avg_loss = epoch_loss / batches
            self.loss_list.append(avg_loss)

            # Accuracy check
            self.eval()

            with torch.no_grad():
                preds = []

                for i in range(0, X_t.size(0), 256):
                    batch = X_t[i:i + 256]
                    preds.append(self.forward(batch).argmax(dim=1))

                train_pred = torch.cat(preds)

            self.train()

            evaluator.data = {
                'true_y': y_t,
                'pred_y': train_pred
            }

            acc = evaluator.evaluate()['accuracy']
            self.acc_list.append(acc)

            print(f"Epoch {epoch+1}/{self.max_epoch} | Loss: {avg_loss:.4f} | Acc: {acc:.4f}")

    # Testing the model
    def test(self, X):

        self.eval()

        X_t = torch.LongTensor(np.array(X))

        preds = []

        with torch.no_grad():

            for i in range(0, X_t.size(0), 256):
                batch = X_t[i:i + 256]
                preds.append(self.forward(batch).argmax(dim=1))

        return torch.cat(preds).numpy()

# Running the pipeline
    def run(self):

        self.loss_list = []
        self.acc_list = []

        print("RNN method running...")
        print("--start training...")

        train_X = self.data['train']['X']
        train_y = self.data['train']['y']

        test_X = self.data['test']['X']
        test_y = self.data['test']['y']

        vocab_size = self.data.get('vocab_size', 10000)
        self.build_model(vocab_size)

        self.train_model(train_X, train_y)

        print("--start testing...")
        pred_y = self.test(test_X)

        return {
            'pred_y': pred_y,
            'true_y': test_y,
            'loss_list': self.loss_list,
            'acc_list': self.acc_list,
        }
        print("Debug Check")
        print("Train labels unique:", np.unique(train_y))
        print("Test labels unique:", np.unique(test_y))

        sample_out = self.forward(torch.LongTensor(train_X[:5]))
        print("Sample logits:", sample_out)
        print("Sample predictions:", sample_out.argmax(dim=1))
