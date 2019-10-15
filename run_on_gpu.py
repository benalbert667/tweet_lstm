import torch
import torch.nn as nn
import torch.optim as optim
from custom_nets import LayeredLSTM
from random import shuffle
# %%

with open('trump_tweets.txt', 'r', encoding='UTF-8') as file:
    data = file.readlines()
# categories = set(_ for d in data for _ in d)
categories = ('\n', ' ', '!', '"', '#', '$', '%', '&', '\'', '(', ')', '+', ',', '-', '.', '/',
              '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '=', '?', '@',
              'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
              'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '_', 'a', 'b', 'c', 'd', 'e',
              'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
              'v', 'w', 'x', 'y', 'z', '<END>')
for i in range(len(data)):
    data[i] = data[i].strip()
    chrs = set(data[i])
    for erase in chrs - set(categories):
        data[i] = data[i].replace(erase, '')

n_letters = len(categories)


gpu = torch.device('cuda:0')


def input_target_pair(s):
    if s in categories:  # inputting letter by letter
        inp = torch.zeros(1, 1, n_letters)
        inp[0, 0, categories.index(s)] = 1
        return inp.to(gpu)
    inp = torch.zeros(len(s), 1, n_letters)
    tar = []
    for i, c in enumerate(s):
        if i > 0:
            tar.append(categories.index(c))
        inp[i, 0, categories.index(c)] = 1
    tar.append(len(categories) - 1)
    tar = torch.LongTensor(tar)
    tar.unsqueeze_(-1)
    return inp.to(gpu), tar.to(gpu)


for n_layers in (3, 6, 9, 12, 15):
    print('Training lstm of {} layers...'.format(n_layers))
    net = LayeredLSTM(n_letters, n_letters, num_layers=n_layers).to(gpu)
    criterion = nn.NLLLoss()
    lr = 0.005 / (10 * (n_layers/3))
    optimizer = optim.Adam(net.parameters(), lr=lr)

    for epoch in range(2 + n_layers//3):
        count = 0
        shuffle(data)
        for d in data:
            seq_len = len(d)
            input_tensor, target_tensor = input_target_pair(d)
            net.init_new_seq()

            optimizer.zero_grad()

            loss = None

            for i in range(seq_len):
                output = net(input_tensor[i])
                l = criterion(output, target_tensor[i])
                if loss is None:
                    loss = l
                else:
                    loss += l

            if loss is None:
                continue

            loss.backward()

            optimizer.step()

            count += 1
            if not count % 30:
                print('|', end='')

        print('\nepoch {} done'.format(epoch))
    print()

    torch.save(net, 'nets/trump{}.pt'.format(n_layers))

