import os
import pandas as pd
import cv2
import numpy as np
import torch
import model
import torch.nn as nn
import torchvision.models as models
from torchsummary import summary
import numpy as np
from PIL import Image
import model
import argparse
from utils.preprocess import preprocess
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import random
from sklearn.model_selection import train_test_split
import pickle
pytorchModel = model.KitModel("../weights.pth")

class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = preprocess(self.images[idx])
        label = self.labels[idx]
        
        return image, label
    

for param in pytorchModel.parameters():
    param.requires_grad = False

# Unfreeze the last layer
for param in pytorchModel.fc8_new_1.parameters():
    param.requires_grad = True
figure_name = ["3Dreconstruction","illustrations","flowchart","heatmap","LineGraph","microscopy","photography",'Radiology',"scatterplot",'signals_waves','table']

def get_data(figure_path):
    img_path = []
    images = []
    labels = []
    for figure_type in figure_path:
        dir = os.path.join("../img",figure_type)
        file_paths = []
        
        num_subdir = 0
        for p in os.listdir(dir):
            if(os.path.isdir(os.path.join(dir,p))):
                num_subdir += 1
        #print(num_subdir)
        full_paths = []
        for root , dirs , files in os.walk(dir):
            for img in files:
                full_paths.append(os.path.join(root,img))
                #print(os.path.join(root,img_path))
        #print(len(full_paths))
        file_paths = random.sample(full_paths,500)
        for file in file_paths:
            #print(image_path)
            img_path.append(file)
            labels.append(figure_name.index(figure_type))
    # Random shuffle
    #print(labels)
    zipped_list = list(zip(img_path, labels))
# Step 2: Shuffle the zipped list
    random.shuffle(zipped_list)
    # Step 3: Unzip the lists
    img_path, labels = zip(*zipped_list)
    img_path = list(img_path)
    labels = list(labels)
    print(img_path[:10])
    print(labels[:10])
    # Convert tuples back to lists (if needed)
    shuffled_images = np.array(img_path)
    shuffled_labels = np.array(labels)
    train_images, test_images, train_labels, test_labels = train_test_split(
    shuffled_images, shuffled_labels, test_size=0.2, random_state=42)
    print(train_images)
    print(train_labels)
    return shuffled_images, shuffled_labels, train_images, test_images, train_labels, test_labels


def train(images,labels,model):
    train_dataset = CustomDataset(images, labels)
    #test_dataset = CustomDataset(images, labels)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    #test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    #print(f"labels range: {torch.min(labels),torch.max(labels)}")
    # Training loop
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            #assert labels[0] >= 0 and labels[0] < len(figure_name)
            labels = torch.tensor(np.repeat(labels[np.newaxis, :], 10, axis=0),dtype=torch.long)
            images, labels = images.squeeze().to(device), labels.squeeze().to(device)
            #print(images.shape)
            #print(labels.shape)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")

def single_pass(model,img_path="./image.png"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = preprocess(img_path).to(device)
    outputs = model(img)
    probs = torch.sum(outputs,axis=0)
    pred = torch.argmax(probs)
    print(figure_name[pred])

def test(images,labels,model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    test_dataset = CustomDataset(images, labels)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    acc = 0
    accc_by_type = [0] * len(figure_name)
    length_by_type = [0] * len(figure_name)
    correct = []
    preds = []
    for images, labels in test_loader:
        outputs = model(images.squeeze().to(device))
        probs = torch.sum(outputs,axis=0)
        pred = torch.argmax(probs)
        #print(pred)
        #print(labels)
        acc += torch.sum(pred.cpu() == labels)
        accc_by_type[labels[0]] += torch.sum(pred.cpu() == labels)
        length_by_type[labels[0]] += 1
        correct.append(torch.sum(pred.cpu() == labels))
        preds.append(pred.cpu())
    for i in range(len(figure_name)):
        accc_by_type[i] = accc_by_type[i] / length_by_type[i]
    print(f'total length: {len(test_loader)}')
    print(f'accuracy: {acc/len(test_loader)}')
    print(f'accuracy by type: {accc_by_type}')
    return correct, preds

def visualize(correct, test_images):
    true_indices = [index for index, value in enumerate(correct) if value]
    

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Training Loop")

    # Add arguments
    pytorchModel = torch.load("../ckpts/classifier.pt")
    print("Model loaded")
    # Parse arguments
    args = parser.parse_args()
    _, _, train_images, test_images, train_labels, test_labels = get_data(figure_name)
    single_pass(pytorchModel)
    # correct_index, pred  = test(test_images,test_labels,pytorchModel)
    # visualize(correct_index,test_images)
    # with open("../test/test_images","wb") as f:
    #     pickle.dump(test_images,f)
    # with open("../test/correct_index","wb") as f:
    #     pickle.dump(correct_index,f)
    # with open("../test/labels","wb") as f:
    #     pickle.dump(test_labels,f)
    # with open("../test/preds","wb") as f:
    #     pickle.dump(pred,f)
    #train(train_images,train_labels,pytorchModel)
    #torch.save(pytorchModel,"../ckpts/classifier.pt")

if __name__ == "__main__":
    main()

#mapping: 

