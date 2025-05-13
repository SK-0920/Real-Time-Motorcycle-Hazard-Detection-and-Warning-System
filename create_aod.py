import torch
import torch.nn as nn
import os

class AODNet(nn.Module):
    def __init__(self):
        super(AODNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=5, padding=2, bias=True)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=5, padding=2, bias=True)
        self.conv3 = nn.Conv2d(6, 3, kernel_size=5, padding=2, bias=True)
        self.conv4 = nn.Conv2d(6, 3, kernel_size=5, padding=2, bias=True)
        self.conv5 = nn.Conv2d(9, 3, kernel_size=5, padding=2, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        concat1 = torch.cat((x1, x2), 1)
        x3 = self.relu(self.conv3(concat1))
        concat2 = torch.cat((x2, x3), 1)
        x4 = self.relu(self.conv4(concat2))
        concat3 = torch.cat((x1, x2, x4), 1)
        k = self.relu(self.conv5(concat3))
        output = k * x - k + 1.0
        return torch.clamp(output, 0, 1)

# Create directory if it doesn't exist
pretrained_dir = "/Users/santhoshkumarg/Downloads/Projects/zTest_image_convertion/pretrained"
os.makedirs(pretrained_dir, exist_ok=True)

# Save random weights
model = AODNet()
output_path = os.path.join(pretrained_dir, "aod_net.pth")
torch.save(model.state_dict(), output_path)
print(f"Saved random-weight AOD-Net to {output_path}")