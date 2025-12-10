import torch
import torch.nn as nn

class LSTMNet(nn.Module):
    def __init__(self, config):
        input_size = config.lstm.group_size * config.num_chan_eeg  # Number of features
        hidden_size = config.lstm.num_hidden  # Number of features in the hidden state
        num_layers = config.lstm.num_layers  # Number of stacked LSTM layers
        output_size = config.num_chan_kin
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out
