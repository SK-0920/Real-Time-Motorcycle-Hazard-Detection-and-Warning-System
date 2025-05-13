import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
from PIL import Image
from pathlib import Path

# Custom dataset class
class WeatherDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Define label mapping for conditions: [fog, high beam, night, rain]
        self.label_map = {
            "fog": [1, 0, 0, 0],
            "high beam": [0, 1, 0, 0],
            "night": [0, 0, 1, 0],
            "rain": [0, 0, 0, 1],
            "rain+night": [0, 0, 1, 1]
        }
        
        # Load images and labels from subfolders
        self.images = []
        self.labels = []
        for condition in self.label_map.keys():
            folder_path = os.path.join(image_dir, condition)
            for img_name in os.listdir(folder_path):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    self.images.append(os.path.join(folder_path, img_name))
                    self.labels.append(self.label_map[condition])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label

# Define the model
class ConditionClassifier(nn.Module):
    def __init__(self):
        super(ConditionClassifier, self).__init__()
        self.model = models.mobilenet_v2(pretrained=True)
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.model.last_channel, 4)  # 4 conditions: fog, high beam, night, rain
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        return self.sigmoid(x)

# Training function
def train_model(model, dataloader, epochs=10, device="cpu"):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()  # Binary cross-entropy for multi-label

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    return model

# Main execution
if __name__ == "__main__":
    # Paths
    image_dir = "/Users/santhoshkumarg/Downloads/Projects/zTest_image_convertion/dataset/"
    model_dir = "/Users/santhoshkumarg/Downloads/Projects/zTest_image_convertion/models/"
    model_path = os.path.join(model_dir, "condition_classifier.pth")

    # Create model directory
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    # Dataset and DataLoader
    dataset = WeatherDataset(image_dir)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Initialize model
    model = ConditionClassifier()

    # Train the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    trained_model = train_model(model, dataloader, epochs=15, device=device)  # 15 epochs for better convergence

    # Save the trained model
    torch.save(trained_model.state_dict(), model_path)
    print(f"Model saved to {model_path}")