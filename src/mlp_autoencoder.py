import torch.nn as nn
import constants


class MLPAutoencoder(nn.Module):
    def __init__(self):
        super(MLPAutoencoder, self).__init__()

        self.encoder = nn.Sequential(nn.Linear(28 * 28, 128), nn.ReLU(True),
                                     nn.Dropout(constants.DROP_RATE),
                                     nn.Linear(128, 64), nn.ReLU(True),
                                     nn.Dropout(constants.DROP_RATE),
                                     nn.Linear(64, 10))

        self.decoder = nn.Sequential(nn.Linear(10, 64), nn.ReLU(True),
                                     nn.Dropout(constants.DROP_RATE),
                                     nn.Linear(64, 128), nn.ReLU(True),
                                     nn.Dropout(constants.DROP_RATE),
                                     nn.Linear(128, 28 * 28), nn.Tanh())

        self.criterian = nn.MSELoss()

    def forward(self, x):
        x = self.encoder(x)
        out = self.decoder(x)
        return out


model = MLPAutoencoder()
print(model)
