import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

class IntentClassifier(nn.Module):
    def __init__(self, input_dim, num_intent_labels, dropout_rate=0.):
        super(IntentClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_intent_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)


class SlotClassifier(nn.Module):
    def __init__(self, input_dim, num_slot_labels, dropout_rate=0.):
        super(SlotClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_slot_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)

class SlotTypeClassifier(nn.Module):
    def __init__(self, input_dim, num_slot_labels, dropout_rate=0.):
        super(SlotTypeClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_slot_labels)
        self.activate = nn.Tanh()

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(self.activate(x))

class BridgeIntentEntities(SlotClassifier): 
    pass

class NgramLSTM(nn.Module):
    def __init__(self, n_gram, input_size, dropout_rate):
        super(NgramLSTM, self).__init__()
        self.n_gram = n_gram

        self._num_layers = 1
        self.input_size = input_size
        self.hidden_size = input_size

        self.dropout = nn.Dropout(dropout_rate)
        self.rnn = nn.LSTM(self.input_size,
                           self.hidden_size,
                           batch_first=False,
                           num_layers=self._num_layers,
                           bidirectional=True)

    def forward(self, _x):
        # we need to create a new data input to learn the n-gram (k) feature using LSTM
        # with origin input (_x) = [emb_1, emb_2, emb_3 .. emb_{seq_length}]: batchsize x seq_length x emb_size
        n_gram = self.n_gram
        data_input = _x.unsqueeze(dim=0)
        data_org = _x

        batch_size = data_org.size(0)
        seq_length = data_org.size(1)
        hidden_size = self.hidden_size
        input_size = self.input_size

        #
        # 1. add padding k - 1 times =>  [k x batch_size x seq_length x emb_size]
        #    [zero_1, .. zero_{k-1}, emb_1, emb_2, emb_3 .. emb_{seq_length - k + 1}]: batchsize x seq_length x emb_size
        #    [zero_1, .. emb_1,      emb_2, emb_3 ..        emb_{seq_length - k + 2}]: batchsize x seq_length x emb_size
        #    ...
        #    [emb_1, emb_2, emb_3 ..                        emb_{seq_length}]: batchsize x seq_length x emb_size
        for i_gram in range(1, n_gram):
            mt_padd_i = F.pad(data_org.transpose(-1,-2), [i_gram, 0],
                              mode='constant', value=0).transpose(-1,-2)[:,:-i_gram,:]
            data_input = torch.cat((mt_padd_i.unsqueeze(dim=0), data_input), dim=0)

            #
        # reshape input into =>   [(batch_size x seq_length) x k x emb_size]
        # this mean that we cut the sentence into many sentence piece (k-gram) similar
        # n-gram in NLP, and combined all set of n-gram treat to LSTM as a batch of input
        zz = data_input.reshape([n_gram,
                                 batch_size * seq_length,
                                 hidden_size])

        # forward data using Bi-LSTM
        # we just get the cell state (num_layers * num_directions, batch, hidden_size)
        # because we need to get the long-memmory to extract the k-gram features of words
        # in this case, we use num_layers = 1, num_directions=2,
        # we sum all directions
        _bank_mt, (_h_n, c_n) = self.rnn(self.dropout(zz))
        out = torch.sum(_h_n+c_n, dim=0)

        # finally, we reshape original batch_size to return
        # (batch x seq x hidden_size)

        out = out.reshape(batch_size, -1, hidden_size)

        # rate_local_context = torch.sigmoid(_x)
        # out = rate_local_context*out + (1 - rate_local_context)*_x
        return out

class NgramMinPooling(nn.Module):
    def __init__(self, n_gram, input_size, phrase_drop=0.3):
        super(NgramMinPooling, self).__init__()
        self.n_gram = n_gram

        self.phrase_drop = phrase_drop
        self.dropout = nn.Dropout(0.3) 

    def forward(self, _x):
        # we need to create a new data input to learn the n-gram (k) feature using LSTM
        # with origin input (_x) = [emb_1, emb_2, emb_3 .. emb_{seq_length}]: batchsize x seq_length x emb_size
        n_gram = self.n_gram
        data_input = _x.unsqueeze(dim=0)
        data_org = _x

        batch_size = data_org.size(0)
        seq_length = data_org.size(1)
        hidden_size = data_org.size(-1)

        #
        # 1. add padding k - 1 times =>  [k x batch_size x seq_length x emb_size]
        #    [zero_1, .. zero_{k-1}, emb_1, emb_2, emb_3 .. emb_{seq_length - k + 1}]: batchsize x seq_length x emb_size
        #    [zero_1, .. emb_1,      emb_2, emb_3 ..        emb_{seq_length - k + 2}]: batchsize x seq_length x emb_size
        #    ...
        #    [emb_1, emb_2, emb_3 ..                        emb_{seq_length}]: batchsize x seq_length x emb_size
        for i_gram in range(1, n_gram):
            mt_padd_i = F.pad(data_org.transpose(-1,-2), [i_gram, 0],
                              mode='constant', value=0).transpose(-1,-2)[:,:-i_gram,:]
            data_input = torch.cat((mt_padd_i.unsqueeze(dim=0), data_input), dim=0)

            #
        rand_index = np.random.permutation(batch_size * seq_length)[:int(batch_size * seq_length * (1 - self.phrase_drop))]
        rand_index.sort()
        rand_index = torch.LongTensor(rand_index).to(device=data_input.device)

         # reshape input into =>   [(batch_size x seq_length) x k x emb_size]
        # this mean that we cut the sentence into many sentence piece (k-gram) similar
        # n-gram in NLP, and combined all set of n-gram treat to LSTM as a batch of input
        zz = data_input.view(n_gram, -1, hidden_size).index_select(1, rand_index)

        # forward data using min value
        out, _ = torch.min(self.dropout(zz), dim=0)

        # copy back phrase states override word states 
        out = _x.reshape(-1, hidden_size).index_copy(0, rand_index, out)

        # finally, we reshape original batch_size to return
        # (batch x seq x hidden_size)
        out = out.reshape(batch_size, -1, hidden_size)

        rate_local_context = torch.sigmoid(_x)
        out = rate_local_context*out + (1 - rate_local_context)*_x

        return out

class NgramMLP(nn.Module):
    def __init__(self, n_gram, input_size):
        super(NgramMLP, self).__init__()
        self.n_gram = n_gram

        self._num_layers = 1
        self.input_size = input_size
        self.hidden_size = input_size

        self.dropout = nn.Dropout(0.3)
        self.activate_layer = nn.Sequential(nn.Linear(n_gram*input_size, input_size), nn.Tanh())

    def forward(self, _x):
        # we need to create a new data input to learn the n-gram (k) feature using LSTM
        # with origin input (_x) = [emb_1, emb_2, emb_3 .. emb_{seq_length}]: batchsize x seq_length x emb_size
        n_gram = self.n_gram
        data_input = _x.unsqueeze(dim=0)
        data_org = _x

        batch_size = data_org.size(0)
        seq_length = data_org.size(1)
        hidden_size = self.hidden_size
        input_size = self.input_size

        #
        # 1. add padding k - 1 times =>  [k x batch_size x seq_length x emb_size]
        #    [zero_1, .. zero_{k-1}, emb_1, emb_2, emb_3 .. emb_{seq_length - k + 1}]: batchsize x seq_length x emb_size
        #    [zero_1, .. emb_1,      emb_2, emb_3 ..        emb_{seq_length - k + 2}]: batchsize x seq_length x emb_size
        #    ...
        #    [emb_1, emb_2, emb_3 ..                        emb_{seq_length}]: batchsize x seq_length x emb_size
        for i_gram in range(1, n_gram):
            mt_padd_i = F.pad(data_org.transpose(-1,-2), [i_gram, 0],
                              mode='constant', value=0).transpose(-1,-2)[:,:-i_gram,:]
            data_input = torch.cat((mt_padd_i.unsqueeze(dim=0), data_input), dim=0)

            #
        # reshape input into =>   [(batch_size x seq_length) x k x emb_size]
        # this mean that we cut the sentence into many sentence piece (k-gram) similar
        # n-gram in NLP, and combined all set of n-gram treat to LSTM as a batch of input
        zz = data_input.reshape([n_gram,
                                 batch_size * seq_length,
                                 hidden_size]).transpose(0, 1).reshape(-1, n_gram*hidden_size)

        # forward data using Multi layer perceptron
        out = self.activate_layer(self.dropout(zz))

        # finally, we reshape original batch_size to return
        # (batch x seq x hidden_size)

        out = out.reshape(batch_size, -1, hidden_size)

        rate_local_context = torch.sigmoid(_x)
        out = rate_local_context*out + (1 - rate_local_context)*_x
        return out