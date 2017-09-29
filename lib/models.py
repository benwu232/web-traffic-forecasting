import random
import sklearn

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
from torch.optim import *
import torch.nn.functional as F
from tqdm import tqdm, trange
import tensorboard_logger as tb_logger
import tensorboard_logger as tblg
import yaml
from lib.foundation import *

SOS_token = 0
EOS_token = 1
USE_CUDA = False
USE_CUDA = True

class SmapeLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets):
        numerator = torch.abs(predictions - targets) * 200.0
        denominator = torch.abs(predictions) + torch.abs(targets)
        # for kaggle, avoid 0 / 0
        denominator[numerator<1e-2] = 1.0
        smape = (numerator / denominator).mean()
        return smape


class SparseLoss(torch.nn.Module):
    def __init__(self, sparsity=0.05, active_threshold=-0.9, labda=2.0):
        super().__init__()
        self.sparsity = sparsity
        self.active_threshold = active_threshold
        self.labda = labda

    def forward(self, inputs):
        in_size = inputs.size()
        input_num = in_size[0] * in_size[1] * in_size[2]
        active_num = (inputs - self.active_threshold).sign().sum()
        activeness = active_num / input_num
        sparsity_ = 1 - self.sparsity
        activeness_ = 1 - activeness
        sparse_loss = self.sparsity * torch.log(self.sparsity / activeness) + sparsity_ * torch.log(sparsity_ / activeness_)
        #sparse_loss = nn.KLDivLoss(activeness, self.sparsity)

        return self.labda * sparse_loss


class GaussianNoise(nn.Module):
    def __init__(self, stddev):
        super().__init__()
        self.stddev = stddev

    def forward(self, din):
        if self.training:
            return din + torch.autograd.Variable(torch.randn(din.size()).cuda() * self.stddev)
        return din


class ResBac(nn.Module):
    expansion = 1

    def __init__(self, filters_in, filters_out, kernel_size=3, padding=0, dilation=1, stride=1, dropout_rate=0.5,
                 using_bn=False, using_act=True, res=True):
        super(ResBac, self).__init__()
        self.filters_in = filters_in
        self.filters_out = filters_out
        self.stride = stride
        self.res = res
        self.using_bn = using_bn
        if self.using_bn:
            self.bn = nn.BatchNorm1d(filters_in)
        self.using_act = using_act
        self.conv1d = nn.Conv1d(filters_in, filters_out,
                               kernel_size=kernel_size, stride=stride,
                               padding=padding, dilation=dilation, bias=True)
        self.act = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.change_dim = nn.Conv1d(self.filters_in, self.filters_out, padding=0,
                                    kernel_size=1, stride=self.stride, bias=False)

    def forward(self, din):
        x = din

        if self.using_bn:
            x = self.bn(x)

        if self.using_act:
            x = self.act(x)

        x = self.conv1d(x)

        if self.res:
            tmp = din
            if self.stride > 1 or self.filters_in != self.filters_out:
                tmp = self.change_dim(din)
            #print(x.size(), tmp.size())
            x += tmp

        x = self.dropout(x)

        return x

def freeze_model(model, freeze=True):
    active = not freeze
    for param in model.parameters():
        param.requires_grad = active

def active_model(model, active=True):
    freeze_model(model, not active)

class EncoderRnn(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout=0.1, bidirectional=False):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        #self.conv1 = ResBac(self.input_size, self.hidden_size, kernel_size=3, stride=1, using_bn=False, padding=1, res=False)
        self.rnn1 = nn.GRU(input_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=self.bidirectional)
        self.rnn2 = nn.GRU(hidden_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=self.bidirectional)

    def forward(self, input_seqs, hidden=None):
        outputs, hidden = self.rnn1(input_seqs, hidden)
        if self.bidirectional:
            outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:] # Sum bidirectional outputs
        outputs, hidden = self.rnn2(outputs, hidden)
        if self.bidirectional:
            outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:] # Sum bidirectional outputs
        return outputs, hidden


class DecoderRnn(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout=0.1, bidirectional=False):
        super().__init__()

        self.input_size = output_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_direction = 1
        if self.bidirectional == True:
            self.num_direction = 2

        #self.conv1 = ResBac(self.input_size, self.hidden_size, kernel_size=3, stride=1, using_bn=False, padding=1, res=False)
        self.rnn1 = nn.GRU(self.input_size, self.hidden_size, n_layers, dropout=self.dropout, bidirectional=self.bidirectional)
        self.rnn2 = nn.GRU(hidden_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=self.bidirectional)
        #self.rnn3 = nn.GRU(hidden_size, hidden_size, dropout=self.dropout, bidirectional=self.bidirectional)
        self.fc = nn.Linear(self.hidden_size * self.num_direction, self.output_size)

    def forward(self, x, h=None):
        #print(input_seqs.size(), hidden.size())
        if x.size()[-1] != self.hidden_size:
            x, h = self.rnn1(x, h)
            if self.bidirectional:
                x = x[:, :, :self.hidden_size] + x[:, :, self.hidden_size:] # Sum bidirectional outputs
        x, h = self.rnn2(x, h)
        #if self.bidirectional:
        #    x = x[:, :, :self.hidden_size] + x[:, :, self.hidden_size:] # Sum bidirectional outputs
        #x, h = self.rnn3(x, h)
        x = F.dropout(x, 0.5)
        x = self.fc(x)
        #outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:] # Sum bidirectional outputs
        return x, h


#Endoer with partial activation while training
class EncoderPA3(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout=0.1, bidirectional=False, activation_rate=0.3):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.activation_rate = activation_rate

        #self.conv1 = ResBac(self.input_size, self.hidden_size, kernel_size=3, stride=1, using_bn=False, padding=1, res=False)
        self.rnn1 = nn.GRU(input_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=self.bidirectional)
        self.rnn2 = nn.GRU(hidden_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=self.bidirectional)
        self.rnn3 = nn.GRU(hidden_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=self.bidirectional)

    def forward(self, input_seqs, hidden=None):
        outputs, hidden = self.rnn1(input_seqs, hidden)
        if self.bidirectional:
            outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:] # Sum bidirectional outputs

        outputs, hidden = self.rnn2(outputs, hidden)
        if self.bidirectional:
            outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:] # Sum bidirectional outputs

        outputs, hidden = self.rnn3(outputs, hidden)
        if self.bidirectional:
            outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:] # Sum bidirectional outputs

        return outputs, hidden

    def random_active(self):
        activation = True if random.random() < self.activation_rate else False
        active_model(self.rnn1, activation)
        activation = True if random.random() < self.activation_rate else False
        active_model(self.rnn2, activation)
        activation = True if random.random() < self.activation_rate else False
        active_model(self.rnn3, activation)

    def active_nn(self, **kwargs):
        if 'activation1' in kwargs:
            active_model(self.rnn1, kwargs['activation1'])
        if 'activation2' in kwargs:
            active_model(self.rnn2, kwargs['activation2'])
        if 'activation3' in kwargs:
            active_model(self.rnn3, kwargs['activation3'])


class EncoderPA2(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout=0.1, bidirectional=False, activation_rate=0.3):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.activation_rate = activation_rate

        #self.conv1 = ResBac(self.input_size, self.hidden_size, kernel_size=3, stride=1, using_bn=False, padding=1, res=False)
        self.rnn1 = nn.GRU(input_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=self.bidirectional)
        self.rnn2 = nn.GRU(hidden_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=self.bidirectional)

    def forward(self, input_seqs, hidden=None):
        outputs, hidden = self.rnn1(input_seqs, hidden)
        if self.bidirectional:
            outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:] # Sum bidirectional outputs

        outputs, hidden = self.rnn2(outputs, hidden)
        if self.bidirectional:
            outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:] # Sum bidirectional outputs

        return outputs, hidden

    def random_active(self):
        activation = True if random.random() < self.activation_rate else False
        active_model(self.rnn1, activation)
        activation = True if random.random() < self.activation_rate else False
        active_model(self.rnn2, activation)

    def active_nn(self, **kwargs):
        if 'activation1' in kwargs:
            active_model(self.rnn1, kwargs['activation1'])
        if 'activation2' in kwargs:
            active_model(self.rnn2, kwargs['activation2'])


class DecoderPA2(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout=0.1, bidirectional=False, activation_rate=0.3):
        super().__init__()

        self.input_size = output_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_direction = 1
        if self.bidirectional == True:
            self.num_direction = 2

        self.activation_rate = activation_rate

        self.rnn2 = nn.GRU(hidden_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=self.bidirectional)
        self.fc = nn.Linear(self.hidden_size * self.num_direction, self.output_size)

    def forward(self, x, h=None):
        x, h = self.rnn2(x, h)
        x = self.fc(x)
        return x, h

    def random_active(self):
        activation = True if random.random() < self.activation_rate else False
        active_model(self.rnn2, activation)

    def active_nn(self, **kwargs):
        if 'activation2' in kwargs:
            active_model(self.rnn2, kwargs['activation2'])


class DecoderPA3(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout=0.1, bidirectional=False, activation_rate=0.3):
        super().__init__()

        self.input_size = output_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_direction = 1
        if self.bidirectional == True:
            self.num_direction = 2

        self.activation_rate = activation_rate

        #self.conv1 = ResBac(self.input_size, self.hidden_size, kernel_size=3, stride=1, using_bn=False, padding=1, res=False)
        self.rnn1 = nn.GRU(self.input_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=self.bidirectional)
        self.rnn2 = nn.GRU(hidden_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=self.bidirectional)
        self.rnn3 = nn.GRU(hidden_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=self.bidirectional)
        self.fc1 = nn.Linear(self.hidden_size * self.num_direction, self.hidden_size * self.num_direction // 2)
        self.fc2 = nn.Linear(self.hidden_size * self.num_direction // 2, self.output_size)

    def forward(self, x, h=None):
        #print(input_seqs.size(), hidden.size())
        if x.size()[-1] != self.hidden_size:
            x, h = self.rnn1(x, h)
            if self.bidirectional:
                x = x[:, :, :self.hidden_size] + x[:, :, self.hidden_size:] # Sum bidirectional outputs
        x, h = self.rnn2(x, h)
        if self.bidirectional:
            x = x[:, :, :self.hidden_size] + x[:, :, self.hidden_size:] # Sum bidirectional outputs
        x, h = self.rnn3(x, h)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, 0.5)
        x = self.fc2(x)
        x = F.relu(x)
        return x, h

    def random_active(self):
        activation = True if random.random() < self.activation_rate else False
        active_model(self.rnn1, activation)
        activation = True if random.random() < self.activation_rate else False
        active_model(self.rnn1, activation)
        activation = True if random.random() < self.activation_rate else False
        active_model(self.rnn1, activation)
        activation = True if random.random() < self.activation_rate else False
        active_model(self.fc1, activation)
        activation = True if random.random() < self.activation_rate else False
        active_model(self.fc2, activation)

    def active_nn(self, **kwargs):
        if 'activation1' in kwargs:
            active_model(self.rnn1, kwargs['activation1'])
        if 'activation2' in kwargs:
            active_model(self.rnn2, kwargs['activation2'])
        if 'activation3' in kwargs:
            active_model(self.rnn3, kwargs['activation3'])


class EncoderPA5(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout=0.1, bidirectional=False, activation_rate=0.2):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.activation_rate = activation_rate

        #self.conv1 = ResBac(self.input_size, self.hidden_size, kernel_size=3, stride=1, using_bn=False, padding=1, res=False)
        self.rnn1 = nn.GRU(input_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=self.bidirectional)
        self.rnn2 = nn.GRU(hidden_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=self.bidirectional)
        self.rnn3 = nn.GRU(hidden_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=self.bidirectional)
        self.rnn4 = nn.GRU(hidden_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=self.bidirectional)
        self.rnn5 = nn.GRU(hidden_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=self.bidirectional)

    def forward(self, input_seqs, hidden=None):
        activation = True if random.random() < self.activation_rate else False
        active_model(self.rnn1, activation)
        outputs, hidden = self.rnn1(input_seqs, hidden)
        if self.bidirectional:
            outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:] # Sum bidirectional outputs

        activation = True if random.random() < self.activation_rate else False
        active_model(self.rnn2, activation)
        outputs, hidden = self.rnn2(outputs, hidden)
        if self.bidirectional:
            outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:] # Sum bidirectional outputs

        activation = True if random.random() < self.activation_rate else False
        active_model(self.rnn3, activation)
        outputs, hidden = self.rnn3(outputs, hidden)
        if self.bidirectional:
            outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:] # Sum bidirectional outputs

        activation = True if random.random() < self.activation_rate else False
        active_model(self.rnn4, activation)
        outputs, hidden = self.rnn4(outputs, hidden)
        if self.bidirectional:
            outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:] # Sum bidirectional outputs

        activation = True if random.random() < self.activation_rate else False
        active_model(self.rnn5, activation)
        outputs, hidden = self.rnn5(outputs, hidden)
        if self.bidirectional:
            outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:] # Sum bidirectional outputs

        return outputs, hidden

from sru import SRU, SRUCell
# Using SRU
class EncoderSRU3(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout=0.1, bidirectional=False, activation_rate=0.3):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.activation_rate = activation_rate

        #self.conv1 = ResBac(self.input_size, self.hidden_size, kernel_size=3, stride=1, using_bn=False, padding=1, res=False)
        self.rnn1 = SRU(input_size, hidden_size, num_layers=n_layers, dropout=self.dropout, rnn_dropout=self.dropout, use_tanh=1, use_relu=0, bidirectional=self.bidirectional)
        self.rnn2 = SRU(hidden_size, hidden_size, num_layers=n_layers, dropout=self.dropout, rnn_dropout=self.dropout, use_tanh=1, use_relu=0, bidirectional=self.bidirectional)
        self.rnn3 = SRU(hidden_size, hidden_size, num_layers=n_layers, dropout=self.dropout, rnn_dropout=self.dropout, use_tanh=1, use_relu=0, bidirectional=self.bidirectional)

    def forward(self, input_seqs, hidden=None):
        outputs, hidden = self.rnn1(input_seqs, hidden)
        if self.bidirectional:
            outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:] # Sum bidirectional outputs

        outputs, hidden = self.rnn2(outputs, hidden)
        if self.bidirectional:
            outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:] # Sum bidirectional outputs

        outputs, hidden = self.rnn3(outputs, hidden)
        if self.bidirectional:
            outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:] # Sum bidirectional outputs

        return outputs, hidden

    def random_active(self):
        activation = True if random.random() < self.activation_rate else False
        active_model(self.rnn1, activation)
        activation = True if random.random() < self.activation_rate else False
        active_model(self.rnn2, activation)
        activation = True if random.random() < self.activation_rate else False
        active_model(self.rnn3, activation)

    def active_nn(self, **kwargs):
        if 'activation1' in kwargs:
            active_model(self.rnn1, kwargs['activation1'])
        if 'activation2' in kwargs:
            active_model(self.rnn2, kwargs['activation2'])
        if 'activation3' in kwargs:
            active_model(self.rnn3, kwargs['activation3'])


class DecoderSRU3(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout=0.1, bidirectional=False, activation_rate=0.3):
        super().__init__()

        self.input_size = output_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_direction = 1
        if self.bidirectional == True:
            self.num_direction = 2

        self.activation_rate = activation_rate

        #self.conv1 = ResBac(self.input_size, self.hidden_size, kernel_size=3, stride=1, using_bn=False, padding=1, res=False)
        self.rnn1 = SRU(self.input_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=self.bidirectional)
        self.rnn2 = SRU(hidden_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=self.bidirectional)
        self.rnn3 = SRU(hidden_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=self.bidirectional)
        self.fc1 = nn.Linear(self.hidden_size * self.num_direction, self.hidden_size * self.num_direction // 2)
        self.fc2 = nn.Linear(self.hidden_size * self.num_direction // 2, self.output_size)

    def forward(self, x, h=None):
        #print(input_seqs.size(), hidden.size())
        if x.size()[-1] != self.hidden_size:
            x, h = self.rnn1(x, h)
            if self.bidirectional:
                x = x[:, :, :self.hidden_size] + x[:, :, self.hidden_size:] # Sum bidirectional outputs
        x, h = self.rnn2(x, h)
        if self.bidirectional:
            x = x[:, :, :self.hidden_size] + x[:, :, self.hidden_size:] # Sum bidirectional outputs
        x, h = self.rnn3(x, h)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, 0.5)
        x = self.fc2(x)
        x = F.relu(x)
        return x, h

    def random_active(self):
        activation = True if random.random() < self.activation_rate else False
        active_model(self.rnn1, activation)
        activation = True if random.random() < self.activation_rate else False
        active_model(self.rnn2, activation)
        activation = True if random.random() < self.activation_rate else False
        active_model(self.rnn3, activation)
        activation = True if random.random() < self.activation_rate else False
        active_model(self.fc1, activation)

    def active_nn(self, **kwargs):
        if 'activation1' in kwargs:
            active_model(self.rnn1, kwargs['activation1'])
        if 'activation2' in kwargs:
            active_model(self.rnn2, kwargs['activation2'])
        if 'activation3' in kwargs:
            active_model(self.rnn3, kwargs['activation3'])




class EncDec(object):
    def __init__(self, model_pars):
        self.model_dir = '../output/models'
        if model_pars['enc_file'] is None:
            self.encoder = self.init_encoder(model_pars['encoder'])
        else:
            self.load_encoder(model_pars['enc_file'])
        active_model(self.encoder)
        self.enc_freeze_span = [0, 0]
        if 'enc_freeze_span' in model_pars:
            self.enc_freeze_span = model_pars['enc_freeze_span']

        if model_pars['dec_file'] is None:
            self.decoder = self.init_decoder(model_pars['decoder'])
        else:
            self.load_decoder(model_pars['dec_file'])

        self.clip = model_pars['clip']
        self.lr = model_pars['lr']
        self.train_batch_per_epoch = model_pars['train_batch_per_epoch']
        self.validate_batch_per_epoch = model_pars['validate_batch_per_epoch']
        self.timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        self.model_file = None

        self.set_loss_fn(model_pars['loss_fn'])

        self.encoder_optimizer = self.set_optimizer(self.encoder, model_pars['encoder']['optimizer'])
        self.decoder_optimizer = self.set_optimizer(self.decoder, model_pars['decoder']['optimizer'])

        if USE_CUDA:
            self.encoder.cuda()
            self.decoder.cuda()

    def set_optimizer(self, model, optim_pars):
        if optim_pars['type'] == 'SGD':
            optimizer = SGD(model.parameters(), lr=self.lr, weight_decay=optim_pars['l2_scale'], momentum=optim_pars['momentum'], dampening=optim_pars['dampening'], nesterov=optim_pars['nesterov'])
        elif optim_pars['type'] == 'Adadelta':
            optimizer = Adadelta(model.parameters(), lr=self.lr, rho=optim_pars['rho'], weight_decay=optim_pars['l2_scale'], eps=optim_pars['epsilon'])
        elif optim_pars['type'] == 'Adam':
            optimizer = Adam(model.parameters(), lr=self.lr, betas=(optim_pars['beta1'], optim_pars['beta2']), eps=optim_pars['epsilon'], weight_decay=optim_pars['l2_scale'])
        elif optim_pars['type'] == 'RMSprop':
            optimizer = RMSprop(model.parameters(), lr=self.lr, alpha=optim_pars['rho'], eps=optim_pars['epsilon'], weight_decay=optim_pars['l2_scale'], momentum=optim_pars['momentum'], centered=optim_pars['centered'])
        return optimizer

    def set_loss_fn(self, type='L1Loss'):
        if type == 'L1Loss':
            self.criterion = nn.L1Loss(size_average=False)
        elif type == 'SMAPE':
            self.criterion = SmapeLoss()

    def init_encoder(self, pars):
        self.encoder = None
        if pars['type'] == 'simple':
            self.encoder = EncoderRnn(input_size=pars['input_size'],
                                      hidden_size=pars['hidden_size'],
                                      n_layers=pars['n_layers'],
                                      dropout=pars['dropout'],
                                      bidirectional=pars['bidirectional']
                                      )
        elif pars['type'] == 'pa3':
            self.encoder = EncoderPA3(input_size=pars['input_size'],
                                      hidden_size=pars['hidden_size'],
                                      n_layers=pars['n_layers'],
                                      dropout=pars['dropout'],
                                      bidirectional=pars['bidirectional'],
                                      activation_rate=pars['activation_rate']
                                      )
        elif pars['type'] == 'pa5':
            self.encoder = EncoderPA5(input_size=pars['input_size'],
                                      hidden_size=pars['hidden_size'],
                                      n_layers=pars['n_layers'],
                                      dropout=pars['dropout'],
                                      bidirectional=pars['bidirectional'],
                                      activation_rate=pars['activation_rate']
                                      )
        elif pars['type'] == 'pa2':
            self.encoder = EncoderPA2(input_size=pars['input_size'],
                                      hidden_size=pars['hidden_size'],
                                      n_layers=pars['n_layers'],
                                      dropout=pars['dropout'],
                                      bidirectional=pars['bidirectional'],
                                      activation_rate=pars['activation_rate']
                                      )
        elif pars['type'] == 'sru3':
            self.encoder = EncoderSRU3(input_size=pars['input_size'],
                                      hidden_size=pars['hidden_size'],
                                      n_layers=pars['n_layers'],
                                      dropout=pars['dropout'],
                                      bidirectional=pars['bidirectional'],
                                      activation_rate=pars['activation_rate']
                                      )
        return self.encoder

    def init_decoder(self, pars):
        self.decoder = None
        if pars['type'] == 'simple':
            self.decoder = DecoderRnn(hidden_size=pars['hidden_size'],
                                      output_size=pars['output_size'],
                                      n_layers=pars['n_layers'],
                                      dropout=pars['dropout'],
                                      bidirectional=pars['bidirectional']
                                      )
        elif pars['type'] == 'pa2':
            self.decoder = DecoderPA2(hidden_size=pars['hidden_size'],
                                      output_size=pars['output_size'],
                                      n_layers=pars['n_layers'],
                                      dropout=pars['dropout'],
                                      bidirectional=pars['bidirectional'],
                                      activation_rate=pars['activation_rate']
                                      )
        elif pars['type'] == 'pa3':
            self.decoder = DecoderPA3(hidden_size=pars['hidden_size'],
                                      output_size=pars['output_size'],
                                      n_layers=pars['n_layers'],
                                      dropout=pars['dropout'],
                                      bidirectional=pars['bidirectional'],
                                      activation_rate=pars['activation_rate']
                                      )
        elif pars['type'] == 'sru3':
            self.decoder = DecoderSRU3(hidden_size=pars['hidden_size'],
                                       output_size=pars['output_size'],
                                      n_layers=pars['n_layers'],
                                      dropout=pars['dropout'],
                                      bidirectional=pars['bidirectional'],
                                      activation_rate=pars['activation_rate']
                                      )

        return self.decoder

    def save_model(self, enc_file, dec_file):
        torch.save(self.encoder, os.path.join(self.model_dir, enc_file))
        torch.save(self.decoder, os.path.join(self.model_dir, dec_file))

    def load_encoder(self, enc_file):
        self.encoder = torch.load(os.path.join(self.model_dir, enc_file))

    def load_decoder(self, dec_file):
        self.decoder = torch.load(os.path.join(self.model_dir, dec_file))

    def load_models(self, model_files):
        enc_file = model_files[0]
        dec_file = model_files[1]
        self.encoder = torch.load(os.path.join(self.model_dir, enc_file))
        self.decoder = torch.load(os.path.join(self.model_dir, dec_file))

        #if USE_CUDA:
        #    self.encoder.cuda()
        #    self.decoder.cuda()

    def set_train(self, mode=True):
        self.encoder.train(mode)
        self.decoder.train(mode)

    def train_batch(self, input_batches, target_batches):
        input_batches = Variable(input_batches)
        target_batches = Variable(target_batches)

        # Zero gradients of both optimizers
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        #target_batches = target_batches.float()

        # Run words through encoder
        if USE_CUDA:
            input_batches = input_batches.cuda()

        input_batches = self.transform(input_batches)
        encoder_outputs, encoder_hidden = self.encoder(input_batches, hidden=None)

        decoder_outputs, decoder_hidden = self.decoder(encoder_outputs, encoder_hidden)
        predictions = self.inv_transform(decoder_outputs)

        # Move new Variables to CUDA
        if USE_CUDA:
            target_batches = target_batches.cuda()

        # Loss calculation and backpropagation
        loss = self.criterion(predictions, target_batches)

        loss.backward()

        # Clip gradient norms
        enc_clip = torch.nn.utils.clip_grad_norm(self.encoder.parameters(), self.clip)
        dec_clip = torch.nn.utils.clip_grad_norm(self.decoder.parameters(), self.clip)

        # Update parameters with optimizers
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.data[0], enc_clip, dec_clip


    def train(self, train_bb):
        self.set_train(True)
        epoch_loss = 0
        #train epoch

        tq = tqdm(range(1, self.train_batch_per_epoch+1), unit='batch')
        #for batch_cnt in range(self.train_batch_per_epoch):
        for batch_cnt in tq:
            tq.set_description('Batch %i/%i' % (batch_cnt, self.train_batch_per_epoch))
            #input_seq, _, _, target_seq = self.train_bb.build_batch()
            input_seq, _, _, target_seq = train_bb.build_batch()

            # Run the train function
            batch_loss, enc_clip, dec_clip = self.train_batch(input_seq, target_seq)
            epoch_loss += batch_loss
            tq.set_postfix(train_loss=round(epoch_loss/batch_cnt, 3), enc_clip=round(enc_clip, 4), dec_clip=round(dec_clip, 4))
        epoch_loss /= self.train_batch_per_epoch
        return epoch_loss

    def validate_batch(self, input_batches, target_batches):
        return 0

    def validate(self, validate_bb):
        self.set_train(False)
        epoch_loss = 0
        #validate epoch
        for batch_cnt in range(self.validate_batch_per_epoch):
            #input_seq, _, _, target_seq = self.validate_bb.build_batch()
            input_seq, _, _, target_seq = validate_bb.build_batch()

            batch_loss = self.validate_batch(input_seq, target_seq)
            epoch_loss += batch_loss
        epoch_loss /= self.validate_batch_per_epoch
        return epoch_loss

    def reconfig_model(self, config_file):
        with open(config_file, 'r') as f:
            pars = yaml.safe_load(f)
        active_model(self.encoder)
        self.set_optimizer(self.encoder, pars['encoder']['optimizer'])
        active_model(self.decoder)
        self.set_optimizer(self.decoder, pars['decoder']['optimizer'])

    def run_train(self, train_bb, validate_bb, epochs=10, **kwargs):
        self.timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        base = kwargs['base']
        config_file = kwargs['config_file']
        kf = ''
        if 'kf' in kwargs:
            kf = '_kf{}_'.format(kwargs['kf'])
        prefix = 'wtf_' + self.timestamp + kf

        save_freq = 0
        if 'save_freq' in kwargs:
            save_freq = int(kwargs['save_freq'])

        tblg.configure('../output/tblog/{}'.format(self.timestamp), flush_secs=10)

        #tq = tqdm(range(1, epochs + 1), unit='epoch')
        for epoch in range(1, epochs + 1):
            self.reconfig_model(config_file)
            if self.enc_freeze_span[0] <= epoch <= self.enc_freeze_span[1]:
                print('Freeze encoder')
                logging.info('Freeze encoder')
                freeze_model(self.encoder)
            else:
                print('Active encoder')
                logging.info('Active encoder')
                active_model(self.encoder)

            print(dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f"))
            print('************************ Training epoch {} *******************************'.format(epoch))
            logging.info('************************ Training epoch {} *******************************'.format(epoch))
            #tq.set_description('Epoch %i/%i' % (epoch, epochs))
            train_loss = self.train(train_bb)
            tblg.log_value('train_loss', train_loss, epoch)
            logging.info('Training loss: {}'.format(train_loss))
            #print('Epoch {}, training loss: {}'.format(epoch, train_loss[0]))
            validate_score = self.validate(validate_bb)
            tblg.log_value('validate_smape', validate_score, epoch)
            print('Validation Smape score: {}'.format(validate_score))
            logging.info('Validation Smape score: {}'.format(validate_score))
            #tq.set_postfix(train_loss=train_loss, validate_smape=validate_score)

            if save_freq > 0 and epoch % save_freq == 0:
                enc_model_file = prefix + str(epoch) + '_enc.pth'
                dec_model_file = prefix + str(epoch) + '_dec.pth'
                print('Save model to files: {}, {}'.format(enc_model_file, dec_model_file))
                logging.info('Save model to files: {}, {}'.format(enc_model_file, dec_model_file))
                self.save_model(enc_model_file, dec_model_file)

            print('\n\n')
            logging.info('\n\n')


    def predict_batch(self, input_batches, predict_seq_len):
        return 0

    def predict(self, predict_bb, predict_seq_len):
        self.set_train(False)

        predict_results = []
        batch_cnt = 0
        for batch_data, _, _ in predict_bb.build_batch_once(target_len=predict_seq_len):
            batch_cnt += 1
            if batch_cnt % 100 == 0:
                print('Predicted %d samples' % (batch_cnt*batch_data.size()[1]))
            #print(batch_data.size())
            pred_batch = self.predict_batch(batch_data, predict_seq_len)
            predict_results.append(pred_batch.cpu().data.numpy())

        #predict_results = torch.cat(predict_results)
        #predict_results = predict_results.cpu().data.numpy()
        predict_results = np.concatenate(predict_results)
        #predict_results = predict_results.clip(0)
        return predict_results


class Seq2Seq(EncDec):
    def __init__(self, model_pars):
        super().__init__(model_pars)

        self.teacher_forcing_ratio = model_pars['teacher_forcing_ratio']
        self.keep_hidden = model_pars['keep_hidden']

    def train_batch(self, input_batches, target_batches):
        input_batches = Variable(input_batches)
        target_batches = Variable(target_batches)

        # Zero gradients of both optimizers
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        if USE_CUDA:
            input_batches = input_batches.cuda()

        input_batches[:, :, 0] = self.transform(input_batches[:, :, 0].unsqueeze(2))
        encoder_outputs, encoder_hidden = self.encoder(input_batches, hidden=None)

        # Prepare input and output variables
        decoder_input = encoder_outputs[-1].unsqueeze(0)
        decoder_hidden = encoder_hidden[:self.decoder.n_layers*self.decoder.num_direction] # Use last (forward) hidden state from encoder

        target_len = target_batches.size(0)
        batch_size = input_batches.size(1)
        all_decoder_outputs = Variable(torch.zeros(target_len, batch_size, self.decoder.output_size))

        # Move new Variables to CUDA
        if USE_CUDA:
            all_decoder_outputs = all_decoder_outputs.cuda()
            target_batches = target_batches.cuda()

        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False
        for t in range(target_len):
            #decoder_output, decoder_hidden, decoder_attn = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            if self.keep_hidden:
                decoder_output, _ = self.decoder(decoder_input, decoder_hidden)
            else:
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

            #print(all_decoder_outputs[t].size(), decoder_output[0].size())
            all_decoder_outputs[t] = decoder_output[0]
            if use_teacher_forcing:
                decoder_input = target_batches[t].view(1, -1, 1) # Teacher forcing: Next input is current target
            else:
                decoder_input = decoder_output      # Next input is current prediction

        # Loss calculation and backpropagation
        #predictions = all_decoder_outputs.squeeze(2).permute(1, 0)
        #targets = target_batches.permute(1, 0)
        predictions = all_decoder_outputs.squeeze(2)
        #loss = self.criterion(predictions, target_batches) / (target_len * batch_size)
        #predictions = torch.exp(predictions) - 1.0
        predictions = self.inv_transform(predictions)
        loss = self.criterion(predictions, target_batches)

        loss.backward()

        # Clip gradient norms
        enc_clip = torch.nn.utils.clip_grad_norm(self.encoder.parameters(), self.clip)
        dec_clip = torch.nn.utils.clip_grad_norm(self.decoder.parameters(), self.clip)

        # Update parameters with optimizers
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.data[0], enc_clip, dec_clip

    def validate_batch(self, input_batches, target_batches):
        input_batches = Variable(input_batches, volatile=True)
        target_batches = Variable(target_batches, volatile=True)
        #ValidateLoss = SmapeLoss()
        if USE_CUDA:
            input_batches = input_batches.cuda()

        #input_batches = self.transform(input_batches)
        input_batches[:, :, 0] = self.transform(input_batches[:, :, 0].unsqueeze(2))
        encoder_outputs, encoder_hidden = self.encoder(input_batches, hidden=None)

        # Prepare input and output variables
        #decoder_input = Variable(torch.FloatTensor([SOS_token] * self.batch_size).view(1, -1, 1))
        decoder_input = encoder_outputs[-1].unsqueeze(0)
        #decoder_hidden = encoder_hidden[:self.decoder.n_layers].squeeze(0) # Use last (forward) hidden state from encoder
        decoder_hidden = encoder_hidden[:self.decoder.n_layers*self.decoder.num_direction] # Use last (forward) hidden state from encoder

        target_len = target_batches.size(0)
        batch_size = input_batches.size(1)
        all_decoder_outputs = Variable(torch.zeros(target_len, batch_size, self.decoder.output_size))

        # Move new Variables to CUDA
        if USE_CUDA:
            #decoder_input = decoder_input.cuda()
            all_decoder_outputs = all_decoder_outputs.cuda()

        # Run through decoder one time step at a time
        for t in range(target_len):
            #decoder_output, decoder_hidden, decoder_attn = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            if self.keep_hidden:
                decoder_output, _ = self.decoder(decoder_input, decoder_hidden)
            else:
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

            #print(all_decoder_outputs[t].size(), decoder_output[0].size())
            all_decoder_outputs[t] = decoder_output[0]
            #decoder_input = target_batches[t].view(1, -1, 1) # Next input is current target
            decoder_input = decoder_output      # Next input is current prediction

        # Loss calculation and backpropagation
        predictions = all_decoder_outputs.squeeze(2)
        predictions = self.inv_transform(predictions)
        predictions = predictions.permute(1, 0)
        targets = target_batches.permute(1, 0)
        #loss = ValidateLoss(predictions, targets)
        smape_score = smape_np(predictions.data.cpu().numpy(), targets.data.numpy())

        return smape_score

    '''
    def run_train(self, train_bb, validate_bb, epochs=10, **kwargs):
        self.timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        kf = ''
        if 'kf' in kwargs:
            kf = '_kf{}_'.format(kwargs['kf'])
        prefix = 'wtf_' + self.timestamp + kf

        save_freq = 0
        if 'save_freq' in kwargs:
            save_freq = int(kwargs['save_freq'])

        tblg.configure('../output/tblog/{}'.format(self.timestamp), flush_secs=10)

        #tq = tqdm(range(1, epochs + 1), unit='epoch')
        for epoch in range(1, epochs + 1):
            if self.enc_freeze_span[0] <= epoch < self.enc_freeze_span[1]:
                freeze_model(self.encoder)
            else:
                active_model(self.encoder)

            print(dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f"))
            print('************************ Training epoch {} *******************************'.format(epoch))
            logging.info('************************ Training epoch {} *******************************'.format(epoch))
            #tq.set_description('Epoch %i/%i' % (epoch, epochs))
            train_loss = self.train(train_bb)
            tblg.log_value('train_loss', train_loss, epoch)
            logging.info('Training loss: {}'.format(train_loss))
            #print('Epoch {}, training loss: {}'.format(epoch, train_loss[0]))
            validate_score = self.validate(validate_bb)
            tblg.log_value('validate_smape', validate_score, epoch)
            print('Validation Smape score: {}'.format(validate_score))
            logging.info('Validation Smape score: {}'.format(validate_score))
            #tq.set_postfix(train_loss=train_loss, validate_smape=validate_score)

            if save_freq > 0 and epoch % save_freq == 0:
                enc_model_file = prefix + str(epoch) + '_enc.pth'
                dec_model_file = prefix + str(epoch) + '_dec.pth'
                print('Save model to files: {}, {}'.format(enc_model_file, dec_model_file))
                logging.info('Save model to files: {}, {}'.format(enc_model_file, dec_model_file))
                self.save_model(enc_model_file, dec_model_file)

            print('\n\n')
            logging.info('\n\n')
    '''


    def predict_batch(self, input_batches, predict_seq_len):
        input_batches = Variable(input_batches, volatile=True)
        if USE_CUDA:
            input_batches = input_batches.cuda()

        #input_batches = self.transform(input_batches)
        input_batches[:, :, 0] = self.transform(input_batches[:, :, 0].unsqueeze(2))
        encoder_outputs, encoder_hidden = self.encoder(input_batches, hidden=None)

        # Prepare input and output variables
        #decoder_input = Variable(torch.FloatTensor([SOS_token] * self.batch_size).view(1, -1, 1))
        decoder_input = encoder_outputs[-1].unsqueeze(0)
        #decoder_hidden = encoder_hidden[:self.decoder.n_layers].squeeze(0) # Use last (forward) hidden state from encoder
        decoder_hidden = encoder_hidden[:self.decoder.n_layers*self.decoder.num_direction] # Use last (forward) hidden state from encoder

        batch_size = input_batches.size(1)
        all_decoder_outputs = Variable(torch.zeros(predict_seq_len, batch_size, self.decoder.output_size), volatile=True)

        # Move new Variables to CUDA
        if USE_CUDA:
            all_decoder_outputs = all_decoder_outputs.cuda()

        # Run through decoder one time step at a time
        for t in range(predict_seq_len):
            #decoder_output, decoder_hidden, decoder_attn = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            if self.keep_hidden:
                decoder_output, _ = self.decoder(decoder_input, decoder_hidden)
            else:
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            #print(all_decoder_outputs[t].size(), decoder_output[0].size())
            all_decoder_outputs[t] = decoder_output[0]
            #decoder_input = target_batches[t].view(1, -1, 1) # Next input is current target
            decoder_input = decoder_output

        predictions = all_decoder_outputs.squeeze(2)
        predictions = self.inv_transform(predictions).permute(1, 0)
        return predictions

    def predict(self, predict_bb, predict_seq_len):
        predict_results = super().predict(predict_bb, predict_seq_len)
        return predict_results.clip(0)


class Autoencoder(EncDec):
    def __init__(self, model_pars):
        super().__init__(model_pars)

    def train_batch(self, input_batches):
        pass

    def train(self, train_bb):
        self.set_train(True)
        epoch_loss = 0
        #train epoch

        tq = tqdm(range(1, self.train_batch_per_epoch+1), unit='batch')
        #for batch_cnt in range(self.train_batch_per_epoch):
        for batch_cnt in tq:
            tq.set_description('Batch %i/%i' % (batch_cnt, self.train_batch_per_epoch))
            #input_seq, _, _, target_seq = self.train_bb.build_batch()
            input_seq, _, _ = train_bb.build_batch()

            # Run the train function
            batch_loss, enc_clip, dec_clip = self.train_batch(input_seq)
            epoch_loss += batch_loss
            tq.set_postfix(train_loss=round(epoch_loss/batch_cnt, 3), enc_clip=round(enc_clip, 4), dec_clip=round(dec_clip, 4))
        epoch_loss /= self.train_batch_per_epoch
        return epoch_loss

    def run_train(self, train_bb, validate_bb, epochs=10, **kwargs):
        self.timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        prefix = 'wtf_' + self.timestamp + '_ae'
        save_freq = 0
        if 'save_freq' in kwargs:
            save_freq = int(kwargs['save_freq'])

        tblg.configure('../output/tblog/{}'.format(self.timestamp), flush_secs=10)

        #tq = tqdm(range(1, epochs + 1), unit='epoch')
        for epoch in range(1, epochs + 1):
            print(dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f"))
            print('************************ Training epoch {} *******************************'.format(epoch))
            logging.info('************************ Training epoch {} *******************************'.format(epoch))
            #tq.set_description('Epoch %i/%i' % (epoch, epochs))
            train_loss = self.train(train_bb)
            tblg.log_value('train_loss', train_loss, epoch)
            logging.info('Training loss: {}'.format(train_loss))

            if save_freq > 0 and epoch % save_freq == 0:
                enc_model_file = prefix + str(epoch) + '_enc.pth'
                dec_model_file = prefix + str(epoch) + '_dec.pth'
                print('Save model to files: {}, {}'.format(enc_model_file, dec_model_file))
                logging.info('Save model to files: {}, {}'.format(enc_model_file, dec_model_file))
                self.save_model(enc_model_file, dec_model_file)

            print('\n\n')
            logging.info('\n\n')


from torch.autograd import Function
'''
class L1Penalty(Function):
    @staticmethod
    def forward(ctx, input, l1weight):
        ctx.save_for_backward(input)
        ctx.l1weight = l1weight
        return input

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_variables
        grad_input = input.clone().sign().mul(ctx.l1weight)
        grad_input += grad_output
        return grad_input
'''


class SparseAE(Autoencoder):
    def __init__(self, model_pars):
        super().__init__(model_pars)
        self.sparsity = model_pars['sparsity']
        self.labda = model_pars['labda']
        self.active_threshold = model_pars['active_threshold']
        #self.SparseCombo = SparseCombo(self.encoder, self.decoder, sparsity=self.l1weight)
        self.SparseLoss = SparseLoss(self.sparsity, self.active_threshold, self.labda)
        self.SmapeLoss = SmapeLoss()
        self.noise = GaussianNoise(model_pars['gaussian_noise_std'])

    def loss_fn(self, decoder_outputs, input_batches, encoder_outputs):
        smape_loss = self.SmapeLoss(decoder_outputs, input_batches)
        sparse_loss = self.SparseLoss(encoder_outputs)
        return smape_loss + sparse_loss, sparse_loss


    def train_batch(self, input_batches):
        if USE_CUDA:
            input_batches = input_batches.cuda()

        target_batches = Variable(input_batches)
        input_batches = Variable(input_batches)
        input_batches = self.transform(input_batches)

        # Zero gradients of both optimizers
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        # add noise
        input_with_noise = self.noise(input_batches)

        self.encoder.random_active()
        #decoder_outputs, decoder_hidden = self.SparseCombo(input_batches)
        encoder_outputs, encoder_hidden = self.encoder(input_with_noise, hidden=None)

        #sparsity penalty
        #decoder_inputs = L1Penalty.apply(encoder_outputs, self.l1weight)

        decoder_outputs, decoder_hidden = self.decoder(encoder_outputs, encoder_hidden)
        decoder_outputs = self.inv_transform(decoder_outputs)

        loss, sparse_loss = self.loss_fn(decoder_outputs, target_batches, encoder_outputs)

        loss.backward()

        # Clip gradient norms
        enc_clip = torch.nn.utils.clip_grad_norm(self.encoder.parameters(), self.clip)
        dec_clip = torch.nn.utils.clip_grad_norm(self.decoder.parameters(), self.clip)

        # Update parameters with optimizers
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.data[0], sparse_loss.data[0], enc_clip, dec_clip

    def train(self, train_bb):
        self.set_train(True)
        epoch_loss = 0
        epoch_sparse_loss = 0
        #train epoch

        tq = tqdm(range(1, self.train_batch_per_epoch+1), unit='batch')
        #for batch_cnt in range(self.train_batch_per_epoch):
        for batch_cnt in tq:
            tq.set_description('Batch %i/%i' % (batch_cnt, self.train_batch_per_epoch))
            #input_seq, _, _, target_seq = self.train_bb.build_batch()
            input_seq, _, _ = train_bb.build_batch()

            # Run the train function
            batch_loss, sparse_loss, enc_clip, dec_clip = self.train_batch(input_seq)
            epoch_loss += batch_loss
            epoch_sparse_loss += sparse_loss
            tq.set_postfix(total_loss=round(epoch_loss/batch_cnt, 3), sparse_loss=round(epoch_sparse_loss/batch_cnt, 3), enc_clip=round(enc_clip, 4), dec_clip=round(dec_clip, 4))
        epoch_loss /= self.train_batch_per_epoch
        return epoch_loss


'''
class ContractiveAE(Autoencoder):
    def __init__(self, model_pars):
        super().__init__(model_pars)

    def train_batch(self, input_batches):
        input_batches = Variable(input_batches)

        # Zero gradients of both optimizers
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        #target_batches = target_batches.float()

        # Run words through encoder
        if USE_CUDA:
            input_batches = input_batches.cuda()

        self.encoder.random_active()
        encoder_outputs, encoder_hidden = self.encoder(input_batches, hidden=None)

        #sparsity penalty
        #encoder_outputs = L1Penalty.apply(encoder_outputs, self.l1weight)

        decoder_output, decoder_hidden = self.decoder(encoder_outputs, encoder_hidden)

        loss = self.criterion(decoder_output, input_batches)

        loss.backward()

        # Clip gradient norms
        enc_clip = torch.nn.utils.clip_grad_norm(self.encoder.parameters(), self.clip)
        dec_clip = torch.nn.utils.clip_grad_norm(self.decoder.parameters(), self.clip)

        # Update parameters with optimizers
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.data[0], enc_clip, dec_clip

'''

