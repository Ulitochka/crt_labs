import torch


num_seg = 16
flag_biLSTM = True
classnum = 7



class Net(torch.nn.Module):
    def __init__(self, sphereface):
        super(Net, self).__init__()
        self.sphereface = sphereface
        self.linear = torch.nn.Linear(512,2)
        self.tanh = torch.nn.Tanh()
        self.avgPool = torch.nn.AvgPool2d((num_seg,1), stride=1)
        self.LSTM = torch.nn.LSTM(512, 512, 1, batch_first = True, dropout=0.2, bidirectional=flag_biLSTM)  # Input dim, hidden dim, num_layer
        for name, param in self.LSTM.named_parameters():
            if 'bias' in name:
                torch.nn.init.constant(param, 0.0)
            elif 'weight' in name:
                torch.nn.init.orthogonal(param)
        
    def sequentialLSTM(self, input, hidden=None):
        input_lstm = input.view([-1, num_seg, input.shape[1]])
        batch_size = input_lstm.shape[0]
        feature_size = input_lstm.shape[2]

        self.LSTM.flatten_parameters()
            
        output_lstm, hidden = self.LSTM(input_lstm)
        if flag_biLSTM:
             output_lstm = output_lstm.contiguous().view(batch_size, output_lstm.size(1), 2, -1).sum(2).view(batch_size, output_lstm.size(1), -1) 

        # avarage the output of LSTM
        output_lstm = output_lstm.view(batch_size,1,num_seg,-1)
        out = self.avgPool(output_lstm)
        out = out.view(batch_size,-1)
        return out
    
    def forward(self, x):
        x = self.sphereface(x)
        x = self.sequentialLSTM(x)
        x = self.linear(x)
        x = self.tanh(x)
        return x