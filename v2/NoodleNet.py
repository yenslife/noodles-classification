import torch.nn as nn

# 輸入的圖片是 300 * 200 之類的
# 記得把輸入的圖片 resize 224
class NoodleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)   # (B, 128, 14, 14)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # (B, 128, 1, 1)

        self.fc = nn.Sequential(
            nn.Flatten(), # (B, 128)
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 3),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.avgpool(x)
        return self.fc(x)

if __name__ == "__main__":
    model = NoodleNet()
    print(model)
