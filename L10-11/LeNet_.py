class Model(nn.Module):
    def __init__(self, input_, hidden_, output_):
        super(Model, self).__init__()
        self.Conv1 = nn.Sequential(
            nn.Conv2d(in_channels=input_, out_channels=hidden_[0],
                      kernel_size=5, stride=1, padding=2),
            nn.Sigmoid()
        )
        self.AvePool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_[0], out_channels=hidden_[1],
                      kernel_size=5, stride=1, padding=0),
            nn.Sigmoid()
        )
        self.AvePool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.FC1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(400, 120),
            nn.Sigmoid()
        )
        self.FC2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.Sigmoid()
        )
        self.Linear = nn.Linear(84, output_)
        self.Softmax = nn.Softmax()

    def forword(self, x):
        x = self.Conv1(x)
        x = self.AvePool1(x)
        x = self.Conv2(x)
        x = self.AvePool2(x)
        x = self.FC1(x)
        x = self.FC2(x)
        x = self.Linear(x)
        x = self.Softmax(x)
        return x