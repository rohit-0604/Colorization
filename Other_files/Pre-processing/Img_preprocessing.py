import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# Consider using a smaller target shape for efficiency in deep learning models
target_shape = (256, 256)  # Example target shape for efficiency
transform_color = transforms.Compose([
    transforms.Resize(target_shape),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def load_and_transform_images(folder, transform):
    images = []
    for filename in os.listdir(folder):
        try:
            path = os.path.join(folder, filename)
            img = Image.open(path).convert("RGB")  # Convert to RGB
            img = transform(img)
            images.append(img)
            print(f"Transformed image shape: {img.shape}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    return torch.stack(images)

# Paths to your image folders
color_folder = r"path_to_color_images"
grey_folder = r"path_to_grey_images"

color_images = load_and_transform_images(color_folder, transform_color)
grey_images = load_and_transform_images(grey_folder, transform_color)

# Show the shapes of the tensors
print("Grey Tensor Shape:", grey_images.shape)
print("Color Tensor Shape:", color_images.shape)

def show_grayscale_images(grey_tensor, num_images=3):
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    for i in range(num_images):
        grey_img = grey_tensor[i][0].numpy() * 0.5 + 0.5  # Extract one channel for display
        grey_img = np.clip(grey_img, 0, 1)
        axes[i].imshow(grey_img, cmap='gray')
        axes[i].axis('off')
    plt.show()

def show_color_images(color_tensor, num_images=3):
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    for i in range(num_images):
        color_img = color_tensor[i].permute(1, 2, 0).numpy() * 0.5 + 0.5
        color_img = np.clip(color_img, 0, 1)
        axes[i].imshow(color_img)
        axes[i].axis('off')
    plt.show()

show_grayscale_images(grey_images)
show_color_images(color_images)

class CustomDataset(Dataset):
    def __init__(self, grey_data, color_data):
        self.grey_data = grey_data
        self.color_data = color_data

    def __len__(self):
        return len(self.grey_data)

    def __getitem__(self, idx):
        return self.grey_data[idx], self.color_data[idx]

dataset = CustomDataset(grey_images, color_images)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for grey_batch, color_batch in dataloader:
    show_grayscale_images(grey_batch)
    show_color_images(color_batch)
    break  # Break after showing the first batch

class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        # Layer 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.norm1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        # Layer 2
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.norm2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        #Layer 3
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.norm3 = nn.BatchNorm2d(256)
        self.pool3 = nn.AvgPool2d(2)
        # Additional layers can be added similarly... if needed

        # Dropout layer
        self.dropout = nn.Dropout(0.3) # to reduce overfitting and more enhanced feature extractor

    def forward(self, x):
        # Pass through Layer 1
        x = self.pool1(self.relu1(self.norm1(self.conv1(x))))
        # Pass through Layer 2
        x = self.pool2(self.relu2(self.norm2(self.conv2(x))))
        # Pass through Layer 2
        x = self.pool3(self.relu3(self.norm3(self.conv3(x))))
        # Continue for additional layers... if there are more layers added

        # Apply dropout
        x = self.dropout(x)
        return x

import torch.nn.functional as F

class AxialAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, axis='height'):
        super(AxialAttentionBlock, self).__init__()
        self.axis = axis
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale = (out_channels // 8) ** -0.5

        self.query = nn.Linear(in_channels, out_channels)
        self.key = nn.Linear(in_channels, out_channels)
        self.value = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        # Reshape and permute x based on the axis
        if self.axis == 'height':
            x = x.permute(0, 2, 1, 3)  # Permute to put 'height' axis first
        elif self.axis == 'width':
            x = x.permute(0, 3, 1, 2)  # Permute to put 'width' axis first

        # Apply linear transformations
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Calculate attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_scores = F.softmax(attn_scores, dim=-1)

        # Apply attention to the value
        output = torch.matmul(attn_scores, v)

        # Reshape and permute back to the original shape
        if self.axis == 'height':
            output = output.permute(0, 2, 1, 3)  # Permute back
        elif self.axis == 'width':
            output = output.permute(0, 2, 3, 1)  # Permute back

        return output
    
class TransformerEncoderLayer(nn.Module):
    def __init__(self, channels):
        super(TransformerEncoderLayer, self).__init__()
        self.height_attention = AxialAttentionBlock(channels, channels, axis='height')
        self.width_attention = AxialAttentionBlock(channels, channels, axis='width')
        self.feed_forward = nn.Sequential(
            # Define feed-forward network
            nn.Linear(channels, channels * 4),
            nn.ReLU(),
            nn.Linear(channels * 4, channels)
        )
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)

    def forward(self, x):
        # Apply axial attention and feed-forward network with residual connections
        x = self.norm1(x + self.height_attention(x))
        x = self.norm2(x + self.width_attention(x))
        x = self.feed_forward(x)
        return x
    
class AxialTransformer(nn.Module):
    def __init__(self, num_layers, channels):
        super(AxialTransformer, self).__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(channels) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class ColorizationModel(nn.Module):
    def __init__(self):
        super(ColorizationModel, self).__init__()
        self.feature_extractor = CNNFeatureExtractor()
        self.transformer = AxialTransformer(num_layers=6, channels=256)
        
        # Upsampling and refinement layers
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.final_conv = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.transformer(x)

        # Final processing to produce a colorized image
        x = F.relu(self.upconv1(x))
        x = F.relu(self.conv1(x))
        x = F.relu(self.upconv2(x))
        x = F.relu(self.conv2(x))
        x = self.sigmoid(self.final_conv(x))
        return x

# Instantiate the ColorizationModel
model = ColorizationModel()
print(model)
model.eval()

class SimpleLoss(nn.Module):
    def __init__(self):
        super(SimpleLoss, self).__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, output, target):
        return self.l1_loss(output, target)
    
# Optimizer

import torch.optim as optim

model = ColorizationModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs=100
def train(model, dataloader, optimizer, num_epochs):
    loss_fn = SimpleLoss()  # Using the L1 loss

    for epoch in range(num_epochs):
        running_loss = 0.0

        for grey_images, color_images in dataloader:
            optimizer.zero_grad()

            # Forward pass
            output = model(grey_images)

            # Compute loss
            loss = loss_fn(output, color_images)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Print average loss for the epoch
        avg_loss = running_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')

