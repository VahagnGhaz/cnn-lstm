import torch
import torch.nn as nn
from torchvision import models

class CNNLSTM(nn.Module):
    def __init__(self, num_classes, hidden_size, num_lstm_layers=2, use_pretrained=True):
        super(CNNLSTM, self).__init__()

        # Choose the CNN backbone
        if use_pretrained:
            # Use pretrained ResNet18
            self.backbone = models.resnet18(pretrained=True)
            # Replace the final fully connected layer of ResNet18
            self.backbone.fc = nn.Identity()
        else:
           self.backbone = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Dropout(0.25),  # Added dropout after the first MaxPool
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25),  # Added dropout after the first MaxPool
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25),  # Added dropout after the first MaxPool
                nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),

                nn.AdaptiveAvgPool2d((1, 1))
            )

        # LSTM
        self.lstm = nn.LSTM(512, hidden_size, num_lstm_layers, batch_first=True)

        # Final classifier
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: batch, num_frames, channels, height, width
        batch, num_frames, c, h, w = x.shape

        # Process each frame with the CNN
        x = x.view(batch * num_frames, c, h, w)
        x = self.backbone(x)

        # Reshape the output for the LSTM
        x = x.view(batch, num_frames, -1)

        # LSTM forward
        x, (h_n, c_n) = self.lstm(x)

        # Classifier
        x = self.fc(h_n[-1, ...])

        return x