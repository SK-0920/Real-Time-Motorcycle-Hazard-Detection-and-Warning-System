import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
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

class RESIDEDataset(Dataset):
    def __init__(self, hazy_dir, clear_dir, transform=None):
        self.hazy_dir = hazy_dir
        self.clear_dir = clear_dir
        self.transform = transform
        self.hazy_images = sorted([f for f in os.listdir(hazy_dir) if f.endswith('.jpg')])
        self.clear_images = {f.split('.')[0]: f for f in os.listdir(clear_dir) if f.endswith('.jpg')}

    def __len__(self):
        return len(self.hazy_images)

    def __getitem__(self, idx):
        hazy_path = os.path.join(self.hazy_dir, self.hazy_images[idx])
        base_id = self.hazy_images[idx].split('_')[0]
        clear_filename = self.clear_images.get(base_id)
        if not clear_filename:
            raise FileNotFoundError(f"No clear image found for base ID {base_id}")
        clear_path = os.path.join(self.clear_dir, clear_filename)
        hazy_img = Image.open(hazy_path).convert('RGB')
        clear_img = Image.open(clear_path).convert('RGB')
        if self.transform:
            hazy_img = self.transform(hazy_img)
            clear_img = self.transform(clear_img)
        return hazy_img, clear_img

# Setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
dataset = RESIDEDataset(
    hazy_dir="/Users/santhoshkumarg/Downloads/Projects/zTest_image_convertion/data/RESIDE/OTS/hazy",
    clear_dir="/Users/santhoshkumarg/Downloads/Projects/zTest_image_convertion/data/RESIDE/OTS/clear",
    transform=transform
)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

model = AODNet().to(device)
criterion = nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train
for epoch in range(10):  # Or 5 for faster test
    for hazy, clear in dataloader:
        hazy, clear = hazy.to(device), clear.to(device)
        optimizer.zero_grad()
        output = model(hazy)
        loss = criterion(output, clear)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Save
torch.save(model.state_dict(), "/Users/santhoshkumarg/Downloads/Projects/zTest_image_convertion/pretrained/aod_net_trained.pth")
print("Saved trained AOD-Net to /Users/santhoshkumarg/Downloads/Projects/zTest_image_convertion/pretrained/aod_net_trained.pth")