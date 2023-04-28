class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.cov1 = nn.Sequential(
            nn.Linear(in_features=4, out_features=16),
            nn.ReLU(),
        )
        self.cov2 = nn.Sequential(
            nn.Linear(in_features=16, out_features=10),
            nn.ReLU(),
        )
        self.Linear = nn.Linear(in_features=10, out_features=3)

    def forward(self, x):
        x = self.cov1(x)
        x = self.cov2(x)
        x = self.Linear(x)
        x = F.softmax(x, dim=1)
        return x