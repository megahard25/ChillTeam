import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
import copy


torch.manual_seed(1)

import math


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)

@torch.jit.script
def f1(x:torch.Tensor, c2:int)->torch.Tensor:
    ans = x[:, : c2 ]
    return ans

@torch.jit.script
def f2(x:torch.Tensor, c2:int)->torch.Tensor:
    ans = x[:,  c2 :]
    return ans


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
        


class InvertedResidualWithShift(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidualWithShift, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        assert expand_ratio > 1

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        assert self.use_res_connect

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )
        
     
        
    def forward(self, x, shift_buffer):
        c = x.size(1)
        #print(type(c))
        #print(c)
        #c = int(c) // 8
        c0 = int(copy.deepcopy(c)) // 8
        x1 = f1(x, c0)
        x2 = f2(x, c0)
        #x1, x2 = x[:, : c // 8], x[:, c // 8:]
        return x + self.conv(torch.cat((shift_buffer, x2), dim=1)), x1

class LSTM(nn.Module):
 
    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=1,
                    num_layers=2):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
 
        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)
 
        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)
 
    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))
 
    def forward(self, input):
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both 
        # have shape (num_layers, batch_size, hidden_dim).
        lstm_out, self.hidden = self.lstm(input.view(len(input), self.batch_size, -1))
        
        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        
        #y_pred = self.linear(lstm_out[-1].view(self.batch_size, -1))
        #return y_pred.view(-1)
        return lstm_out
 
#model = LSTM(lstm_input_size, h1, batch_size=num_train, output_dim=output_dim, num_layers=num_layers)

class MobileNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        # input_channel = make_divisible(input_channel * width_mult)  # first channel is always 32!
        self.last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        global_idx = 0
        shift_block_idx = [2, 4, 5, 7, 8, 9, 11, 12, 14, 15]
        for t, c, n, s in interverted_residual_setting:
            output_channel = make_divisible(c * width_mult) if t > 1 else c
            for i in range(n):
                if i == 0:
                    block = InvertedResidualWithShift if global_idx in shift_block_idx else InvertedResidual
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                    global_idx += 1
                else:
                    block = InvertedResidualWithShift if global_idx in shift_block_idx else InvertedResidual
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                    global_idx += 1
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))


        # ------------------ building LSTM layer -----------------------
        self.hidden_dim = self.last_channel
        self.n_layers = 1
        self.lstm_input_channel = self.last_channel
        #self.lstm_input_channel = 62720
        self.batch_size = 1 
        
        self.lstm_layer = LSTM(self.lstm_input_channel, self.hidden_dim, self.batch_size, output_dim=12, num_layers=self.n_layers)
        #self.features.append(self.lstm_layer)
        # --------------------------------------------------------------

        # make it nn.Sequential
        self.features = nn.ModuleList(self.features)

        # building classifier
        self.classifier = nn.Linear(self.last_channel, n_class)
        #self.classifier = nn.Linear(1280, n_class)
        self._initialize_weights()


    def forward(self, x, *shift_buffer):
        shift_buffer_idx = 0
        out_buffer = []
        for f in self.features:
            if isinstance(f, InvertedResidualWithShift):
                x, s = f(x, shift_buffer[shift_buffer_idx])
                shift_buffer_idx += 1
                out_buffer.append(s)
                #print(f"------------------> {x.size()}")
            else:
                #print(x.size())
                x = f(x)
        #print(f"--------------------------------->  {x.size()}")
        x = x.mean(3).mean(2)
        #print(f"--------------------------------->  {x.size()}")
        
        # ---------------- LSTM block ------------------
        #batch_size = 1
        #seq_len = 1
        
        #x = torch.reshape(x, (self.batch_size, self.seq_len, self.lstm_input_channel))
        #hidden_state = torch.zeros(self.n_layers, self.batch_size, self.hidden_dim)
        #cell_state = torch.zeros(self.n_layers, self.batch_size, self.hidden_dim)
        #hidden = (hidden_state, cell_state)

        #lstm = nn.LSTM(x.size()[1], x.size()[1])
        x = self.lstm_layer(x)
        # ----------------------------------------------
        x = self.classifier(x)
        return (x, *out_buffer)

    def _initialize_weights(self):
        for m in self.modules():
            #print(m)
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mobilenet_v2_140():
    return MobileNetV2(width_mult=1.4)


if __name__ == '__main__':
    net = MobileNetV2()
    x = torch.rand(1, 3, 224, 224)
    shift_buffer = [torch.zeros([1, 3, 56, 56]),
                    torch.zeros([1, 4, 28, 28]),
                    torch.zeros([1, 4, 28, 28]),
                    torch.zeros([1, 8, 14, 14]),
                    torch.zeros([1, 8, 14, 14]),
                    torch.zeros([1, 8, 14, 14]),
                    torch.zeros([1, 12, 14, 14]),
                    torch.zeros([1, 12, 14, 14]),
                    torch.zeros([1, 20, 7, 7]),
                    torch.zeros([1, 20, 7, 7])]
    with torch.no_grad():
        for _ in range(10):
            y, shift_buffer = net(x, *shift_buffer)
            print([s.shape for s in shift_buffer])


