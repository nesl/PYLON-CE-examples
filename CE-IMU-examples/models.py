import torch.nn as nn


class CRNN(nn.Module):
    def __init__(self, n_class, drop_out=0.2):
        super().__init__()
        self.n_class = n_class
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=6, out_channels=8, kernel_size=3, padding='same'),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.BatchNorm1d(num_features=8),
            nn.Dropout(drop_out),
        )
  
        self.lstm = nn.LSTM(input_size=40, hidden_size=16, bidirectional=True, batch_first=True)
        self.fc = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(drop_out),
            nn.Flatten(),
            nn.Linear(in_features=256, out_features=self.n_class),
            nn.Softmax(dim=1)
        )
        
  
    def forward(self, x):
        x = self.conv(x)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x
    

class CNN1D(nn.Module):
    def __init__(self, n_class, drop_out=0.1):
        super().__init__()
        self.n_class = n_class
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=6, out_channels=64, kernel_size=5, padding=2),# Use for torch>1.9/0: nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=3),
            nn.BatchNorm1d(num_features=64),
            nn.Dropout(drop_out),
  
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=3),
            nn.BatchNorm1d(num_features=64),
            nn.Dropout(drop_out),

            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=3),
            nn.BatchNorm1d(num_features=64),
            nn.Dropout(drop_out)
        )
  
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=192, out_features=84),
            nn.ReLU(),
            nn.Linear(84, self.n_class),
            nn.ReLU(),
            nn.Softmax(dim=1)
        )
        
  
    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x