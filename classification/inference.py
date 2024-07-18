import os
import pandas as pd
import cv2
import numpy as np
import torch
import model
import torch.nn as nn
import torchvision.models as models
import numpy as np
from PIL import Image
import model
import argparse
from preprocess import preprocess, preprocess_url
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import random
import json
import requests
from tqdm import tqdm

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

figure_name = ["3Dreconstruction","illustrations","flowchart","heatmap","LineGraph","microscopy","photography",'Radiology',"scatterplot",'signals_waves','tables','barcharts','boxcharts']



def read_json_img_path(path):
    with open(path, 'r', encoding='utf-8') as file:
        file_content = file.read()
    data = json.loads(file_content)
    paths = [d['imgLarge'] for d in data]
    return paths

def read_json_processed(path):
    with open(path, 'r') as f:
        return json.load(f)

def batched_inference(model,path_list,out_path,batch=10,mode="local"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    print(device)
    results = []
    for i in tqdm(range(0,len(path_list),batch)):
        batched_paths = path_list[i:i+batch]
        per_sample_segment = 10
        if mode == "local":
            batched_imgs = [preprocess(i) for i in batched_paths]
        elif mode == "url":
            batched_imgs = [preprocess_url(i) for i in batched_paths]
        else:
            print("Not supported")
        batched_input = torch.cat(batched_imgs,dim=0).to(device)
        #print(batched_input.shape)
        out = model(batched_input)
        print(out.shape)
        out = out.view(batch, -1, *out.shape[1:])
        
        # Compute the mean every 10 elements along the first dimension
        reshaped_out = out.view(-1, 10, *out.shape[2:])
        print(reshaped_out.shape)
        mean_out = reshaped_out.mean(dim=1)
        
        # Take the max along the first dimension
        max_mean_out = mean_out.argmax(dim=1)
        predicted_classes = [figure_name[i] for i in max_mean_out]
        print(predicted_classes)
        for j in range(batch):
            results.append({
                "url": batched_paths[j],
                "predicted_classes": predicted_classes[j]
            })
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {out_path}")

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Training Loop")

    # Add arguments
    pytorchModel = torch.load("./classifier_acl_aug_resume.pt")
    print("Model loaded")
    # Parse arguments
    args = parser.parse_args()
    img_paths = read_json_processed("./image_chunks/chunk_1.json")
    print(img_paths[:10])
    test_paths = ["C:\\Users\\charl\\Research\\figureClassfication\\img\\Radiology\\x-ray\\0.png",
                  "C:\\Users\\charl\\Research\\figureClassfication\\img\\Radiology\\x-ray\\1.png",
                  "C:\\Users\\charl\\Research\\figureClassfication\\img\\Radiology\\x-ray\\2.png",
                  "C:\\Users\\charl\\Research\\figureClassfication\\img\\Radiology\\x-ray\\3.png",
                  "C:\\Users\\charl\\Research\\figureClassfication\\img\\Radiology\\x-ray\\4.png",
                  "C:\\Users\\charl\\Research\\figureClassfication\\img\\Radiology\\x-ray\\5.png",
                  "C:\\Users\\charl\\Research\\figureClassfication\\img\\Radiology\\x-ray\\6.png",
                  "C:\\Users\\charl\\Research\\figureClassfication\\img\\Radiology\\x-ray\\7.png",
                  "C:\\Users\\charl\\Research\\figureClassfication\\img\\Radiology\\x-ray\\8.png",
                  "C:\\Users\\charl\\Research\\figureClassfication\\img\\Radiology\\x-ray\\9.png"]
    batched_inference(pytorchModel,img_paths[:60],batch=30,mode='url',out_path="./pred/pmc/chunk_1_pred_test.json")
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

