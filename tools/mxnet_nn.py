import os
import struct
import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn

class MANN(gluon.HybridBlock):
    def __init__(self,
                 control_neurons=[285, 286, 287, 345, 346, 347, 393, 394, 395, 341, 342, 343, 84, 85, 86, 87, 88, 89, 90],
                 x_dim=480, h_dim=512, y_dim=363, xb_dim=19, hb_dim=32, yb_dim=8,
                 **kwargs):
        super(MANN, self).__init__(**kwargs)
        self.control_neurons = self.params.get_constant('control_neurons', control_neurons)
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.y_dim = y_dim
        self._xb_dim = xb_dim
        self._hb_dim = hb_dim
        self._yb_dim = yb_dim
        self.x_mean = self.params.get('x_mean', shape=(1, self.x_dim))
        self.x_std = self.params.get('x_std', shape=(1, self.x_dim))
        self.y_mean = self.params.get('y_mean', shape=(1, self.y_dim))
        self.y_std = self.params.get('y_std', shape=(1, self.y_dim))
        self.expert_w0 = self.params.get('expert_w0', shape=(1, self._yb_dim, self.h_dim, self.x_dim))
        self.expert_b0 = self.params.get('expert_b0', shape=(1, self._yb_dim, self.h_dim, 1))
        self.expert_w1 = self.params.get('expert_w1', shape=(1, self._yb_dim, self.h_dim, self.h_dim))
        self.expert_b1 = self.params.get('expert_b1', shape=(1, self._yb_dim, self.h_dim, 1))
        self.expert_w2 = self.params.get('expert_w2', shape=(1, self._yb_dim, self.y_dim, self.h_dim))
        self.expert_b2 = self.params.get('expert_b2', shape=(1, self._yb_dim, self.y_dim, 1))
        self.elu = nn.ELU(alpha=1.0)

        # MLP gating
        self.gating = nn.HybridSequential('gating')
        self.gating.add(nn.Dense(self._hb_dim, in_units=self._xb_dim))
        self.gating.add(nn.ELU(alpha=1.0))
        self.gating.add(nn.Dense(self._hb_dim, in_units=self._hb_dim))
        self.gating.add(nn.ELU(alpha=1.0))
        self.gating.add(nn.Dense(self._yb_dim, in_units=self._hb_dim))

    def hybrid_forward(self, F, x, x_mean, x_std, y_mean, y_std, expert_w0, expert_b0, expert_w1, expert_b1, expert_w2, expert_b2, control_neurons):
        # normalize X
        y = (x - x_mean) / x_std
        # gating
        bx = F.take(y, control_neurons, axis=1)
        # print(bx)
        by = F.softmax(self.gating(bx))

        # generate weights from expert weights
        # print('w0', expert_w0, by)
        by_expand = by.reshape((0, 0, 1, 1))
        w0 = F.broadcast_mul(expert_w0, by_expand).sum(axis=(0, 1), keepdims=False)
        b0 = F.broadcast_mul(expert_b0, by_expand).sum(axis=(0, 1), keepdims=False).reshape((-1))
        w1 = F.broadcast_mul(expert_w1, by_expand).sum(axis=(0, 1), keepdims=False)
        b1 = F.broadcast_mul(expert_b1, by_expand).sum(axis=(0, 1), keepdims=False).reshape((-1))
        w2 = F.broadcast_mul(expert_w2, by_expand).sum(axis=(0, 1), keepdims=False)
        b2 = F.broadcast_mul(expert_b2, by_expand).sum(axis=(0, 1), keepdims=False).reshape((-1))

        # prediction
        # y = self.elu(F.dot(y, w0.transpose()) + b0)
        y = self.elu(F.FullyConnected(y, weight=w0, bias=b0, num_hidden=self.h_dim))
        y = self.elu(F.FullyConnected(y, weight=w1, bias=b1, num_hidden=self.h_dim))
        y = F.FullyConnected(y, weight=w2, bias=b2, num_hidden=self.y_dim)
        y = F.broadcast_add(F.broadcast_mul(y, y_std), y_mean)
        return y

def load_wolf_mann(root, net):
    root = os.path.abspath(os.path.expanduser(root))
    params = net.collect_params()
    params.get('x_mean').set_data(load_params(root, 'Xmean.bin', (1, net.x_dim)))
    params.get('x_std').set_data(load_params(root, 'Xstd.bin', (1, net.x_dim)))
    params.get('y_mean').set_data(load_params(root, 'Ymean.bin', (1, net.y_dim)))
    params.get('y_std').set_data(load_params(root, 'Ystd.bin', (1, net.y_dim)))
    params.get('dense0_weight').set_data(load_params(root, 'wc0_w.bin', (net._hb_dim, net._xb_dim)))
    params.get('dense0_bias').set_data(load_params(root, 'wc0_b.bin', (net._hb_dim)))
    params.get('dense1_weight').set_data(load_params(root, 'wc1_w.bin', (net._hb_dim, net._hb_dim)))
    params.get('dense1_bias').set_data(load_params(root, 'wc1_b.bin', (net._hb_dim)))
    params.get('dense2_weight').set_data(load_params(root, 'wc2_w.bin', (net._yb_dim, net._hb_dim)))
    params.get('dense2_bias').set_data(load_params(root, 'wc2_b.bin', (net._yb_dim)))

    expert_w0, expert_b0, expert_w1, expert_b1, expert_w2, expert_b2 = [], [], [], [], [], []
    for i in range(net._yb_dim):
        expert_w0.append(load_params(root, 'cp0_a{}.bin'.format(i), (net.h_dim, net.x_dim)))
        expert_b0.append(load_params(root, 'cp0_b{}.bin'.format(i), (net.h_dim, 1)))
        expert_w1.append(load_params(root, 'cp1_a{}.bin'.format(i), (net.h_dim, net.h_dim)))
        expert_b1.append(load_params(root, 'cp1_b{}.bin'.format(i), (net.h_dim, 1)))
        expert_w2.append(load_params(root, 'cp2_a{}.bin'.format(i), (net.y_dim, net.h_dim)))
        expert_b2.append(load_params(root, 'cp2_b{}.bin'.format(i), (net.y_dim, 1)))

    params.get('expert_w0').set_data(mx.nd.stack(*expert_w0).expand_dims(0))
    params.get('expert_b0').set_data(mx.nd.stack(*expert_b0).expand_dims(0))
    params.get('expert_w1').set_data(mx.nd.stack(*expert_w1).expand_dims(0))
    params.get('expert_b1').set_data(mx.nd.stack(*expert_b1).expand_dims(0))
    params.get('expert_w2').set_data(mx.nd.stack(*expert_w2).expand_dims(0))
    params.get('expert_b2').set_data(mx.nd.stack(*expert_b2).expand_dims(0))


def load_params(root, fname, shape):
    fname = os.path.join(root, fname)
    size = np.prod(shape)
    with open(fname, 'rb') as bin_file:
        arr = struct.unpack('f'*size, bin_file.read(4*size))
    arr = mx.nd.array(arr, dtype='float32').reshape(shape)
    return arr

def test_x():
    with open('x.txt', 'rt') as f:
        x = [float(xx.strip()) for xx in f.readlines()[0].strip('[]').split(',')]
    assert len(x) == 480, 'len: {}'.format(len(x))
    return x

if __name__ == '__main__':
    mann = MANN(prefix='')
    mann.initialize()
    mann.hybridize()
    # x = mx.nd.zeros(shape=(1, mann.x_dim))
    x = mx.nd.array(test_x(), dtype='float32').reshape((1, mann.x_dim))
    # y = mann(x)
    load_wolf_mann('../src/NN_Wolf_MANN', mann)
    yy = mann(x)
    print(yy.shape)
    # print(yy)
    # load_params('../src/NN_Wolf_MANN/Xstd.bin', (1, 480))
    mann.export('wolf_mann')