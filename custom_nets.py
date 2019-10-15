import torch
import torch.nn as nn


class LayeredLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LayeredLSTM, self).__init__()

        self.i_s = input_size
        self.h_s = hidden_size
        self.n_layers = num_layers

        self.hidden_states = [None for _ in range(self.n_layers)]
        self.cell_states = [None for _ in range(self.n_layers)]
        self.lstms = nn.ModuleList([LSTM1(self.i_s, self.h_s) for _ in range(self.n_layers)])

    def init_new_seq(self, random=False):
        for i in range(self.n_layers):
            self.hidden_states[i], self.cell_states[i] = (_ for _ in self.lstms[i].init_states(random=random))

    def forward(self, x):
        h = x
        for i in range(self.n_layers):
            lstm, hidden, cell = self.lstms[i], self.hidden_states[i], self.cell_states[i]
            y, h, c = lstm(h, hidden.view(1, -1), cell)
            self.hidden_states[i], self.cell_states[i] = h, c
        return y


class LSTM1(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM1, self).__init__()

        self.i_s = input_size
        self.h_s = hidden_size

        self.f_gate = nn.Sequential(nn.Linear(self.i_s + self.h_s, self.h_s, bias=True), nn.Sigmoid())
        self.i_gate = nn.Sequential(nn.Linear(self.i_s + self.h_s, self.h_s, bias=True), nn.Sigmoid())
        self.o_gate = nn.Sequential(nn.Linear(self.i_s + self.h_s, self.h_s, bias=True), nn.Sigmoid())
        self.c_gate = nn.Sequential(nn.Linear(self.i_s + self.h_s, self.h_s, bias=True), nn.Tanh())
        self.tanh = nn.Tanh()
        self.softmax = nn.LogSoftmax(dim=1)

    def init_states(self, random=False):
        if random:
            return torch.rand(self.h_s), torch.rand(self.h_s)
        return torch.zeros(self.h_s), torch.zeros(self.h_s)

    def forward(self, x, h, cs):  # input, hidden state, cell state
        x = torch.cat((x, h), 1)
        f, i, o, c = self.f_gate(x), self.i_gate(x), self.o_gate(x), self.c_gate(x)
        cs = (cs * f) + (i * c)
        h = o * self.tanh(cs)
        return self.softmax(h), h, cs


class LSTM2(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM2, self).__init__()

        self.i_s = input_size
        self.h_s = hidden_size

        self.f_gate = nn.Sequential(
            nn.Linear(self.i_s + self.h_s, self.i_s*2//3 + self.h_s, bias=True),
            nn.Linear(self.i_s*2//3 + self.h_s, self.i_s//3 + self.h_s, bias=True),
            nn.Linear(self.i_s//3 + self.h_s, self.h_s, bias=True),
            nn.Sigmoid())
        self.i_gate = nn.Sequential(
            nn.Linear(self.i_s + self.h_s, self.i_s*2//3 + self.h_s, bias=True),
            nn.Linear(self.i_s*2//3 + self.h_s, self.i_s//3 + self.h_s, bias=True),
            nn.Linear(self.i_s//3 + self.h_s, self.h_s, bias=True),
            nn.Sigmoid())
        self.o_gate = nn.Sequential(
            nn.Linear(self.i_s + self.h_s, self.i_s*2//3 + self.h_s, bias=True),
            nn.Linear(self.i_s*2//3 + self.h_s, self.i_s//3 + self.h_s, bias=True),
            nn.Linear(self.i_s//3 + self.h_s, self.h_s, bias=True),
            nn.Sigmoid())
        self.c_gate = nn.Sequential(
            nn.Linear(self.i_s + self.h_s, self.i_s*2//3 + self.h_s, bias=True),
            nn.Linear(self.i_s*2//3 + self.h_s, self.i_s//3 + self.h_s, bias=True),
            nn.Linear(self.i_s//3 + self.h_s, self.h_s, bias=True),
            nn.Tanh())
        self.tanh = nn.Tanh()
        self.softmax = nn.LogSoftmax(dim=1)

    def init_states(self, random=False):
        if random:
            return torch.rand(self.h_s), torch.rand(self.h_s)
        return torch.zeros(self.h_s), torch.zeros(self.h_s)

    def forward(self, x, h, cs):  # input, hidden state, cell state
        x = torch.cat((x, h), 1)
        f, i, o, c = self.f_gate(x), self.i_gate(x), self.o_gate(x), self.c_gate(x)
        cs = (cs * f) + (i * c)
        h = o * self.tanh(cs)
        return self.softmax(h), h, cs
