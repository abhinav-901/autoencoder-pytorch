import torch.nn as nn


class CnnAutoencoder(nn.Module):
    def __init__(self):
        super(CnnAutoencoder, self).__init__()
        self.encoder = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1),
                                     nn.ReLU(True), nn.MaxPool2d(2, 2),
                                     nn.Conv2d(16, 4, 3, padding=1),
                                     nn.ReLU(True), nn.MaxPool2d(2, 2))
        self.decoder = nn.Sequential(nn.ConvTranspose2d(4, 16, 2, 2),
                                     nn.ReLU(True),
                                     nn.ConvTranspose2d(16, 1, 2, 2),
                                     nn.Tanh())
        self.criterian = nn.MSELoss()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class ImprvdCnnAutoencoder(nn.Module):
    def __init__(self):
        super(ImprvdCnnAutoencoder, self).__init__()
        self.encoder = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1),
                                     nn.ReLU(True), nn.MaxPool2d(2, 2),
                                     nn.Conv2d(16, 4, 3, padding=1),
                                     nn.ReLU(True), nn.MaxPool2d(2, 2))
        self.decoder = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='nearest'), nn.ReLU(True),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='nearest'), nn.Tanh())

        self.criterian = nn.MSELoss()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


model = CnnAutoencoder()
print(model)
