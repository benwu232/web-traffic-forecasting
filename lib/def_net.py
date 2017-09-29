import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim


class TsRnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.cell_num = 100
        self.lstm1 = nn.LSTMCell(1, self.cell_num)
        self.lstm2 = nn.LSTMCell(self.cell_num, 1)

    def forward(self, input, future=0):
        outputs = []
        h_t = Variable(torch.zeros(input.size(1), self.cell_num).double().cuda(), requires_grad=False)
        c_t = Variable(torch.zeros(input.size(1), self.cell_num).double().cuda(), requires_grad=False)
        h_t2 = Variable(torch.zeros(input.size(1), 1).double().cuda(), requires_grad=False)
        c_t2 = Variable(torch.zeros(input.size(1), 1).double().cuda(), requires_grad=False)

        for i, input_t in enumerate(input):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(c_t, (h_t2, c_t2))
            #outputs += [c_t2]
        for i in range(future):# if we should predict the future
            h_t, c_t = self.lstm1(c_t2, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(c_t, (h_t2, c_t2))
            outputs += [c_t2]
        outputs = torch.stack(outputs, 1).squeeze(2)
        #outputs = torch.stack(outputs, 1)
        #outputs = outputs.permute(1, 0)
        return outputs

class TsLstm3(nn.Module):
    def __init__(self, feature_num=1, cell_num=100):
        super().__init__()
        self.cell_num = cell_num
        self.lstm1 = nn.LSTMCell(feature_num, cell_num)
        self.lstm2 = nn.LSTMCell(cell_num, cell_num)
        self.lstm3 = nn.LSTMCell(cell_num, 1)

    def forward(self, input, future=0):
        outputs = []
        h_t = Variable(torch.zeros(input.size(1), self.cell_num).double().cuda(), requires_grad=False)
        c_t = Variable(torch.zeros(input.size(1), self.cell_num).double().cuda(), requires_grad=False)
        h_t2 = Variable(torch.zeros(input.size(1), self.cell_num).double().cuda(), requires_grad=False)
        c_t2 = Variable(torch.zeros(input.size(1), self.cell_num).double().cuda(), requires_grad=False)
        h_t3 = Variable(torch.zeros(input.size(1), 1).double().cuda(), requires_grad=False)
        c_t3 = Variable(torch.zeros(input.size(1), 1).double().cuda(), requires_grad=False)

        for i, input_t in enumerate(input):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(c_t, (h_t2, c_t2))
            h_t3, c_t3 = self.lstm3(c_t2, (h_t3, c_t3))
            #outputs += [c_t2]
        for i in range(future):# if we should predict the future
            h_t, c_t = self.lstm1(c_t3, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(c_t, (h_t2, c_t2))
            h_t3, c_t3 = self.lstm3(c_t2, (h_t3, c_t3))
            outputs += [c_t3]
        outputs = torch.stack(outputs, 1).squeeze(2)
        #outputs = torch.stack(outputs, 1)
        #outputs = outputs.permute(1, 0)
        return outputs



class SparseAutoEncoder(torch.nn.Module):
    def __init__(self, feature_size, hidden_size, l1weight):
        super(SparseAutoEncoder, self).__init__()
        self.lin_encoder = nn.Linear(feature_size, hidden_size)
        self.lin_decoder = nn.Linear(hidden_size, feature_size)
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.l1weight = l1weight
        self.rnn = nn.GRU(self.filters, self.output_channel, 1, batch_first=True, bidirectional=self.bidirectional, dropout=0.5)

    def encode(self):
        pass



    def forward(self, input):
        # encoder
        x = input.view(-1, self.feature_size)
        x = self.lin_encoder(x)
        x = F.relu(x)

        # sparsity penalty
        x = L1Penalty.apply(x, self.l1weight)

        # decoder
        x = self.lin_decoder(x)
        x = F.sigmoid(x)
        return x.view_as(input)
