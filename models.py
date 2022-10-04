__author__ = 'martin.ringsquandl'

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, Parameter, BatchNorm1d as BN
from torch_geometric.nn import GATConv, APPNP, GINConv, JumpingKnowledge, global_mean_pool, GCNConv
from torch.autograd import Variable
import math
from sklearn.neural_network import MLPRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, ElasticNet


class NodeMLP(torch.nn.Module):
    def __init__(self, dropout, input_dim, hidden_dim=64, num_outputs=4):
        super(NodeMLP, self).__init__()
        self.dropout = dropout
        self.layer1 = Linear(input_dim, hidden_dim)
        self.layer2 = Linear(hidden_dim, hidden_dim)
        self.layer3 = Linear(hidden_dim, num_outputs)

    def forward(self, data):
        x = data.x
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layer1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.layer3(x)
        return x


class GCNNet(torch.nn.Module):
    def __init__(self, dropout, input_dim, hidden_dim=64, num_outputs=1, num_layers=3, aggr='add'):
        super(GCNNet, self).__init__()
        self.dropout = dropout
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv1.aggr = aggr
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers - 2):
            conv = GCNConv(hidden_dim, hidden_dim)
            conv.aggr = aggr
            self.convs.append(conv)
        self.convout = GCNConv(hidden_dim, num_outputs)
        self.convout.aggr = aggr

    def forward(self, data):
        x, edge_index = data.x, data.edge_index.type(torch.LongTensor)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        for conv in self.convs:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index)
            x = F.relu(x)

        x = self.convout(x, edge_index)

        return x


class GATNet(torch.nn.Module):
    def __init__(self, dropout, input_dim, hidden_dim=64, heads=2, num_outputs=1, num_layers=3, aggr='add'):
        super(GATNet, self).__init__()
        self.dropout = dropout
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads)
        self.conv1.aggr = aggr
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers - 2):
            conv = GATConv(heads * hidden_dim, hidden_dim, heads=heads)
            conv.aggr = aggr
            self.convs.append(conv)

        self.convout = GATConv(heads * hidden_dim, num_outputs)
        self.convout.aggr = aggr

    def forward(self, data):
        x, edge_index = data.x, data.edge_index.type(torch.LongTensor)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        for conv in self.convs:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index)
            x = F.relu(x)

        x = self.convout(x, edge_index)

        return x


class APPNPNet(torch.nn.Module):
    def __init__(self, dropout, input_dim, K, alpha, hidden_dim=64, num_outputs=1):
        super(APPNPNet, self).__init__()
        self.lin1 = Linear(input_dim, hidden_dim)
        self.lin2 = Linear(hidden_dim, num_outputs)
        self.prop1 = APPNP(K=K, alpha=alpha)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index.type(torch.LongTensor)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.prop1(x, edge_index)
        x = self.lin2(x)
        return x


class GIN0WithJK(torch.nn.Module):
    def __init__(self, input_dim, num_layers, dropout, hidden_dim, num_outputs, mode='cat'):
        super(GIN0WithJK, self).__init__()
        self.dropout = dropout
        self.conv1 = GINConv(
            Sequential(
                Linear(input_dim, hidden_dim),
                ReLU(),
                Linear(hidden_dim, hidden_dim),
                ReLU(),
                BN(hidden_dim),
            ), train_eps=False)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(hidden_dim, hidden_dim),
                        ReLU(),
                        Linear(hidden_dim, hidden_dim),
                        ReLU(),
                        BN(hidden_dim),
                    ), train_eps=False))
        self.jump = JumpingKnowledge(mode)
        if mode == 'cat':
            self.lin1 = Linear(num_layers * hidden_dim, hidden_dim)
        else:
            self.lin1 = Linear(hidden_dim, hidden_dim)
        self.lin2 = Linear(hidden_dim, num_outputs)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.jump.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        xs = [x]
        for conv in self.convs:
            x = conv(x, edge_index)
            xs += [x]
        x = self.jump(xs)
        # x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return x

    def __repr__(self):
        return self.__class__.__name__


class GATWithJK(torch.nn.Module):
    def __init__(self, input_dim, num_layers, dropout, hidden_dim, num_outputs, heads, mode='lstm',
                 loads=True, aggr='add'):
        super(GATWithJK, self).__init__()
        self.dropout = dropout
        self.conv1 = GATConv(input_dim, hidden_dim, heads)
        self.conv1.aggr = aggr

        self.loads = loads

        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            tmp_conv = GATConv(heads * hidden_dim, hidden_dim, heads)
            tmp_conv.aggr = aggr
            self.convs.append(tmp_conv)
        self.jump = JumpingKnowledge(mode, channels=heads*hidden_dim, num_layers=2)
        if mode == 'cat':
            if self.loads:
                self.lin1 = Linear(2*(heads * num_layers * hidden_dim), hidden_dim)
            else:
                self.lin1 = Linear(heads * num_layers * hidden_dim, hidden_dim)
        else:
            if self.loads:
                self.lin1 = Linear(2*(heads * hidden_dim), hidden_dim)
            else:
                self.lin1 = Linear(heads * hidden_dim, hidden_dim)

        self.lin2 = Linear(input_dim + hidden_dim, hidden_dim)
        self.lin_out = Linear(hidden_dim, num_outputs)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.jump.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.task_1.reset_parameters()
        self.lin_out.reset_parameters()

    def forward(self, data):
        x_in, edge_index, batch = data.x, data.edge_index, data.batch
        load_index = data.loads
        x = self.conv1(x_in, edge_index)
        xs = [x]
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            xs += [x]
        x = self.jump(xs)
        # x = global_mean_pool(x, batch)
        if self.loads:
            load_vector = torch.index_select(x, 0, load_index).sum(dim=0)
            load_vector = load_vector.repeat([x.shape[0], 1])
            x = torch.cat([x, load_vector], axis=1)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.lin2(torch.cat([x, x_in], axis=1))
        x = F.relu(x)
        x = self.lin_out(x)

        return x

    def __repr__(self):
        return self.__class__.__name__


class PositionalEncoder(torch.nn.Module):
    def __init__(self, d_model, max_seq_len=20):
        super().__init__()
        self.d_model = d_model

        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(1)
        x = x + Variable(self.pe[:, :seq_len], requires_grad=False)
        return x


class PositionalEncoder2(torch.nn.Module):
    def __init__(self, d_model, max_seq_len=20):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embed = torch.nn.Embedding(max_seq_len, d_model)

    def forward(self, x):
        x = x + self.embed(torch.range(0, self.max_seq_len-1).type(torch.LongTensor))
        return x


class Norm(torch.nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = Parameter(torch.ones(self.size))
        self.bias = Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class FeedForward(torch.nn.Module):
    def __init__(self, d_model, d_ff=4, dropout=0.1):
        super().__init__()
        self.linear_1 = Linear(d_model, d_model*d_ff)
        self.dropout = dropout
        self.linear_2 = Linear(d_model*d_ff, d_model)

    def forward(self, x):
        x = F.dropout(F.relu(self.linear_1(x)), p=self.dropout)
        x = self.linear_2(x)
        return x


def attention(q, k, v, d_k, mask=None):
    att_w = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        att_w = att_w.masked_fill(mask == 0, -1e9)
    output = F.softmax(att_w, dim=-1)

    output = torch.matmul(output, v)
    return output, att_w


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = Linear(d_model, d_model)
        self.v_linear = Linear(d_model, d_model)
        self.k_linear = Linear(d_model, d_model)
        self.dropout = dropout
        self.out = Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        if self.h == 1:
            # perform linear operation and split into h heads
            k = self.k_linear(k).view(bs, -1, self.d_k)
            q = self.q_linear(q).view(bs, -1, self.d_k)
            v = self.v_linear(v).view(bs, -1, self.d_k)
        else:
            k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
            v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
            # transpose to get dimensions bs * h * sl * d_model
            k = k.transpose(1, 2)
            q = q.transpose(1, 2)
            v = v.transpose(1, 2)

        # calculate attention using function we will define next
        scores, att_w = attention(q, k, v, self.d_k, mask)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous() \
            .view(bs, -1, self.d_model)

        output = self.out(concat)

        return output, att_w


class EncoderLayer(torch.nn.Module):
    def __init__(self, d_model, num_layers, heads, num_outputs, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.pe = PositionalEncoder2(d_model)
        self.attn = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model)
        self.dropout = dropout
        self.num_layers = num_layers
        self.regression = Linear(d_model, num_outputs)

    def forward(self, x, mask=None):
        # add positional encoding
        # x = self.pe(x)
        # batch normalization
        x2 = self.norm_1(x)
        att, scores = self.attn(x2, x2, x2, mask)
        x = x + F.dropout(att, p=self.dropout)
        x2 = self.norm_2(x)
        x = x + F.dropout(self.ff(x2), p=self.dropout)
        # x = torch.max(x, dim=1)[0]
        # last layer as output (in principle arbitrary)
        x = x[:, self.num_layers-1, :] + F.dropout(self.ff(x2[:, self.num_layers-1, :]), p=self.dropout)
        # x = x[:, 19, :]
        # x = self.regression(torch.flatten(x, start_dim=1))
        x = self.regression(x)
        return x, scores

    def get_attention(self, x, mask=None):
        x2 = self.norm_1(x)
        _, scores = self.attn(x2, x2, x2, mask)
        return scores


class TransfEnc(torch.nn.Module):
    def __init__(self, input_dim, num_layers, dropout, hidden_dim, num_outputs, heads,
                 loads=False, aggr='add'):
        super(TransfEnc, self).__init__()
        self.dropout = dropout
        self.conv1 = GATConv(input_dim, hidden_dim)
        self.conv1.aggr = aggr
        self.loads = loads
        self.num_outputs = num_outputs
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            tmp_conv = GATConv(hidden_dim, hidden_dim)
            tmp_conv.aggr = aggr
            self.convs.append(tmp_conv)

        self.encoder = EncoderLayer(d_model=hidden_dim,
                                    num_layers=num_layers,
                                    heads=heads,
                                    num_outputs=num_outputs,
                                    dropout=dropout)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.encoder.reset_parameters()

    def forward(self, data):
        x_in, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x_in, edge_index)
        xs = torch.unsqueeze(x, 1)
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            xs = torch.cat([xs, torch.unsqueeze(x, 1)], dim=1)
        x = self.encoder(xs)
        return x

    def get_att(self, data):
        x_in, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x_in, edge_index)
        xs = torch.unsqueeze(x, 1)
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            xs = torch.cat([xs, torch.unsqueeze(x, 1)], dim=1)

        attentions = self.encoder.get_attention(xs)
        return attentions

    def __repr__(self):
        return self.__class__.__name__


class FlatMultioutMLP(torch.nn.Module):
    def __init__(self, hidden_layers, dropout, input_dim, output_ids):
        super(FlatMultioutMLP, self).__init__()
        self.dropout = dropout
        self.layers = torch.nn.ModuleList()
        self.layers.append(Linear(input_dim, hidden_layers[0]))
        for i in range(1, len(hidden_layers)):
            self.layers.append(Linear(hidden_layers[i-1], hidden_layers[i]))

        self.layer_out = Linear(hidden_layers[-1], len(output_ids))
        # self.layer_out = Linear(input_dim, len(output_ids))

    def forward(self, x):
        for layer in self.layers:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = layer(x)
            x = F.relu(x)

        x = self.layer_out(x)
        return x

    def out_activation(self, x):
        x02 = torch.tanh(x) + 1 # x in range(0,2)
        scale = (self.max-self.min)/2.
        return x02 * scale + self.min

    def __repr__(self):
        return self.__class__.__name__


class FlatSingleoutMLP():
    def __init__(self, hidden_layers):
        self.dummy = MLPRegressor(hidden_layer_sizes=hidden_layers)    # ElasticNet(alpha=0)    # LinearRegression()

    def __repr__(self):
        return self.__class__.__name__


class FlatSingleoutMLPPytorch(torch.nn.Module):
    def __init__(self, input_dim):
        super(FlatSingleoutMLPPytorch, self).__init__()
        self.layer_out = Linear(input_dim, 1, bias=True)
        torch.nn.init.xavier_normal(self.layer_out.weight.data)
        torch.nn.init.normal(self.layer_out.bias.data)

    def forward(self, x):
        x = self.layer_out(x)
        return x

    def reset_parameters(self):
        torch.nn.init.normal_(self.layer_out)
        # self.layer_out.reset_parameters()


class DummyReg():
    def __init__(self, strategy):
        self.dummy = DummyRegressor(strategy=strategy)

    def __repr__(self):
        return self.__class__.__name__


class LinearReg():
    def __init__(self):
        self.dummy = LinearRegression()  #   # ElasticNet(alpha=0)

    def __repr__(self):
        return self.__class__.__name__
