from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import torch
import unicodedata
import string
import random
import argparse
import time

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math

import torch.nn as nn
import torch

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

n_hidden = 128



class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2)
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        # print(input_size)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        # print(x.shape)
        # print(c_n.shape, h_n.shape, x.shape)
        out = (self.linear(out[-1]))
        # print(x[-1].shape)
        out = self.softmax(out)
        return out

class MyLSTM(nn.Module):
    def __init__(self,input_size, hidden_size, proj_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size= hidden_size

        #input
        self.ii = nn.Linear(input_size, hidden_size)
        self.hi = nn.Linear(hidden_size, hidden_size)

        #forget gate
        self.i2f = nn.Linear(input_size, hidden_size)
        self.hf = nn.Linear(hidden_size, hidden_size)
        #cell
        self.ig = nn.Linear(input_size, hidden_size)
        self.hg = nn.Linear(hidden_size, hidden_size)

        #output gate
        self.io = nn.Linear(input_size, hidden_size)
        self.ho = nn.Linear(hidden_size, hidden_size)
        self.output_proj = nn.Linear(hidden_size, proj_size)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x, h0_c0=None):
        B, L, _=x.shape
        hidden_seq=[]

        if h0_c0 is None:
            h_t1 = torch.zeros(B,self.hidden_size).to(x.device)
            c_t1 = torch.zeros(B,self.hidden_size).to(x.device)
        else:
            h_t1, c_t1 = h0_c0

        for t in range(L):
            x_t = x[:, t, :]  #current input

        i_t = torch.sigmoid(self.ii(x_t) + self.hi(h_t1))
        f_t = torch.sigmoid(self.i2f(x_t) + self.hf(h_t1))
        g_t = torch.tanh(self.ig(x_t) + self.hg(h_t1))
        o_t = torch.sigmoid(self.io(x_t) + self.ho(h_t1))
        c_t1 = f_t * c_t1 + i_t * g_t
        h_t1 = o_t * torch.tanh(c_t1)

        hidden_seq.append(h_t1.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1)
        h_t1 = self.softmax(self.output_proj(h_t1))
        return hidden_seq, (h_t1, c_t1)
  
def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def findFiles(path):
    return glob.glob(path)

# print(findFiles('data/names/*.txt'))


all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)


# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


# print(unicodeToAscii('Ślusàrski'))

# Build the category_lines dictionary, a list of names per language
category_lines = {}
all_categories = []


# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]


for filename in findFiles('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

# print(category_lines['Italian'][:5])


# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)


# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor


# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor


def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i


def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]


def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor

def train(args, category_tensor, line_tensor):
    # hidden = rnn.initHidden()
    #
    # rnn.zero_grad()
    # print(line_tensor.shape)
    # for i in range(line_tensor.size()[0]):
    #     output, hidden = rnn(line_tensor[i], hidden)
    optimizer.zero_grad()
    output = rnn(line_tensor)
    loss = criterion(output, category_tensor)
    loss.backward()
    optimizer.step()

    # Add parameters' gradients to their values, multiplied by learning rate
    # for p in rnn.parameters():
    #     p.data.add_(p.grad.data, alpha=-args.learning_rate)

    return output, loss.item()


def main(args, current_loss):
    correct_num = 0
    for iter in range(1, n_iters + 1):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        output, loss = train(args, category_tensor, line_tensor)
        current_loss += loss

        guess, guess_i = categoryFromOutput(output)
        if guess == category:
            correct_num += 1

        # Print iter number, loss, name and guess
        if iter % print_every == 0:
            # guess, guess_i = categoryFromOutput(output)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

        # Add current loss avg to list of losses
        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            print(f"iter: {iter}; loss: {current_loss / plot_every}")
            print(f"iter: {iter}; accurancy: {correct_num / plot_every}")
            all_accurancy.append(correct_num / plot_every)
            correct_num = 0
            current_loss = 0


# Just return an output given a line
def evaluate(line_tensor):
    # hidden = rnn.initHidden()
    # for i in range(line_tensor.size()[0]):
    #     output, hidden = rnn(line_tensor[i], hidden)
    output = rnn(line_tensor)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=4, type=int, help="The batch size of training")
    parser.add_argument("--device", default='mps', type=str, help="The training device")
    parser.add_argument("--learning_rate", default=0.08, type=float, help="learning rate")
    parser.add_argument("--epochs", default=20, type=int, help="Training epoch")
    parser.add_argument("--logdir", default="./log", type=str)
    parser.add_argument("--hidden", default=128, type=int, help="The number of hidden state")

    args = parser.parse_args()

    for i in range(10):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        print('category =', category, '/ line =', line)

    criterion = nn.NLLLoss()

    # rnn = RNN(n_letters, args.hidden, n_categories)
    rnn = LSTM(n_letters, args.hidden, n_categories)#.to(args.device)
    # rnn = MyLSTM(n_letters, args.hidden, n_categories)#.to(args.device)
    optimizer = torch.optim.SGD(rnn.parameters(), lr=args.learning_rate)
    print(rnn)

    current_loss = 0
    all_losses = []
    all_accurancy = []
    n_iters = 300000
    print_every = 5000
    plot_every = 1000

    start = time.time()

    main(args, current_loss)

    plt.figure()
    plt.plot(all_losses)
    plt.savefig('./log/loss_RNN_3.png')
    plt.figure()
    plt.plot(all_accurancy)
    plt.savefig('./log/acc_RNN_3.png')

    confusion = torch.zeros(n_categories, n_categories)
    n_confusion = 10000

    # Go through a bunch of examples and record which are correctly guessed
    for i in range(n_confusion):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        output = evaluate(line_tensor)
        guess, guess_i = categoryFromOutput(output)
        category_i = all_categories.index(category)
        confusion[category_i][guess_i] += 1

    # Normalize by dividing every row by its sum
    for i in range(n_categories):
        confusion[i] = confusion[i] / confusion[i].sum()

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + all_categories, rotation=90)
    ax.set_yticklabels([''] + all_categories)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # sphinx_gallery_thumbnail_number = 2
    plt.show()
    fig.savefig('log/confusion_m_RNN_3.png')
