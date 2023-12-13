import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pydicom
from torchvision import models, transforms
import numpy as np

def load_dicom(path):           #.dcm loading
    dicom = pydicom.dcmread(path)
    image = dicom.pixel_array
    image = image / image.max()  #normalize to [0,1]
    image = image.astype(np.float32)  #convert to float32
    return image

class MRIDataset(Dataset):    #custom dataset for .dcm      
    def __init__(self, dicom_paths, labels, transforms=None):
        self.dicom_paths = dicom_paths
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.dicom_paths)

    def __getitem__(self, idx):
        image = load_dicom(self.dicom_paths[idx])
        image = (image * 255).astype(np.uint8)  #convert to uint8
        image = np.stack((image,)*3, axis=-1)  #repeat gray image across 3 channels
        image = transforms.ToPILImage()(image)
       
        if self.transforms:
            image = self.transforms(image)

        label = self.labels[idx]
        return image, label

ROOT_DIR =  r"C:\Users\Thinkpad\Downloads\main-alzheimers-project\MRI_Data"
NORMAL_DIR = os.path.join(ROOT_DIR, 'Normal')
ALZ_DIR = os.path.join(ROOT_DIR, 'Alzheimers')

normal_paths = [os.path.join(NORMAL_DIR, fname) for fname in os.listdir(NORMAL_DIR)]
alz_paths = [os.path.join(ALZ_DIR, fname) for fname in os.listdir(ALZ_DIR)]

all_paths = normal_paths + alz_paths
labels = [0] * len(normal_paths) + [1] * len(alz_paths)

transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
])

dataset = MRIDataset(all_paths, labels, transforms=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)  #using ResNet18
model.fc = nn.Linear(model.fc.in_features, 2)  #binary classification
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10 
for epoch in range(num_epochs):    #train loop
    model.train()  
    train_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
    
    train_loss = train_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}")
    
    model.eval()    #set to evaluation
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss = val_loss / len(val_loader.dataset)
    val_acc = correct / total
    print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

save_path = 'alzheimers_model_weights.pth'
torch.save(model.state_dict(), save_path)
print(f"Model weights saved to {save_path}")

# model.load_state_dict(torch.load(save_path))