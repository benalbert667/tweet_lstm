import torch
import torch.nn as nn


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

    def init_states(self):
        return torch.zeros(self.h_s), torch.zeros(self.h_s)

    def forward(self, x, h, cs):  # input, hidden state, cell state
        x = torch.cat((x, h), 1)
        f, i, o, c = self.f_gate(x), self.i_gate(x), self.o_gate(x), self.c_gate(x)
        cs = (cs * f) + (i * c)
        return o * self.tanh(cs), cs
