import torch, detectron2
import sys
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
# import some common libraries
import shutil
import numpy as np
import os, json, cv2, random
import requests
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from tqdm import tqdm
from detectron2.structures import BoxMode
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
import utils.model
import argparse
from utils.preprocess import preprocess, preprocess_url
from detectron2.utils.visualizer import ColorMode
setup_logger()
path = os.getcwd()
figure_name = ["3Dreconstruction","illustrations","flowchart","heatmap","LineGraph","microscopy","photography",'Radiology',"scatterplot",'signals_waves','tables','barcharts','boxcharts']
def load_segmentation_model(path):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
    segmodel = build_model(cfg)  # returns a torch.nn.Modu
    DetectionCheckpointer(segmodel).load(path)
    segmodel.eval()
    segmodel.to('cuda')
    return segmodel

def load_classification_model(path):
    classifyModel = torch.load(path)
    classifyModel.to('cuda')
    classifyModel.eval()
    return classifyModel

def read_coord(file_path):
    # Initialize an empty list to store the coordinates
    coordinates = []

    # Open the file and read line by line
    with open(file_path, 'r') as file:
        for line in file:
            try:
                # Strip any leading/trailing whitespace and split the line into components
                coords = line.strip().split()
                # Convert each component to a float (or int if appropriate) and store as a tuple
                coordinates.append(tuple(map(float, coords)))
            except ValueError as e:
                return []
                
    return coordinates

def read_json_processed(path):
    with open(path, 'r') as f:
        return json.load(f)

def get_val_dicts(img_dir):
    cnt = 0
    dataset_dicts = []
    base_dir = os.path.join(path,img_dir)
    for img in os.listdir(img_dir):
        img_dir = os.path.join(base_dir,img)
        txt_file = os.path.join(path,f"text/text/{img}"+".txt")
        #print(txt_file)
        #read_coord(txt_file)
        #print(img_dir)
        height, width = cv2.imread(img_dir).shape[:2]
                #print(f'{height, width}')
        record = {}
        objs = []
        record["file_name"] = img_dir
        record["image_id"] = cnt-1
        record["height"] = height
        record["width"] = width
        record["annotations"] = objs
        dataset_dicts.append(record)

def get_single_image_url(url):
    dataset_dicts = []
    responseImg = requests.get(url)
    if responseImg.status_code == 200:
        img_array = np.frombuffer(responseImg.content, np.uint8)
    # Decode the image using OpenCV
        im = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        transposed_array = np.transpose(im, (2, 0, 1))
        #print(transposed_array.shape)
        dataset_dicts = []
        record = {}
        record['image'] = torch.from_numpy(transposed_array).to('cuda')
        record['image_orig'] = im
        dataset_dicts.append(record)
    return dataset_dicts

def batched_inference(classifyModel,path_list,batch=10,mode="local"):
    predicted_classes = []
    for i in range(0,len(path_list),batch):
        curr_batch = min(batch,len(path_list)-i)
        batched_paths = path_list[i:i+curr_batch]
        if mode == "local":
            batched_imgs = [preprocess(i) for i in batched_paths]
        elif mode == "url":
            batched_imgs = [preprocess_url(i) for i in batched_paths]
        else:
            print("Not supported")
        batched_input = torch.cat(batched_imgs,dim=0).to('cuda')
        #print(batched_input.shape)
        out = classifyModel(batched_input)
        #print(out.shape)
        out = out.view(batch, -1, *out.shape[1:])
        
        # Compute the mean every 10 elements along the first dimension
        reshaped_out = out.view(-1, 10, *out.shape[2:])
        #print(reshaped_out.shape)
        mean_out = reshaped_out.mean(dim=1)
        
        # Take the max along the first dimension
        max_mean_out = mean_out.argmax(dim=1)
        predicted_classes.extend([figure_name[i] for i in max_mean_out])
    #print(predicted_classes)
    return predicted_classes


def seg_and_pred_url(segmodel,classifyModel,url):
    info = {}
    info['url'] = url
    data = get_single_image_url(url)
    seg_out = segmodel(data)
    instances = seg_out[0]['instances']
    boxes = instances.pred_boxes.tensor
    tmp_dir = "./tmp"
    os.makedirs(tmp_dir,exist_ok=True)
    paths = []
    coords = []
    for i, box in enumerate(boxes):
        if(instances.scores[i] > 0.7):
            x1, y1, x2, y2 = box.int().tolist()
            coords.append([x1, y1, x2, y2])
            segmented_image = data[0]['image_orig'][y1:y2, x1:x2]
            segmented_path = os.path.join(tmp_dir,f"./segmented_{i}.png")
            cv2.imwrite(segmented_path, segmented_image)
            #print("Written!")
            paths.append(segmented_path)
    #print(f"Segmented {len(paths)} images, classifying...")
    predicted_classes = batched_inference(classifyModel,paths,batch=10)
    info['segments'] = coords
    info['predicted_classes'] = predicted_classes
    return info
    #shutil.rmtree(tmp_dir)



#info = seg_and_pred_url("https://openi.nlm.nih.gov/imgs/512/94/4042575/PMC4042575_nutrients-06-01913-g002.png")

# dataset_dicts = get_balloon_dicts("./img")
# for d in random.sample(dataset_dicts, 3):
#     img = cv2.imread(d["file_name"])
#     visualizer = Visualizer(img[:, :, ::-1], metadata=balloon_metadata, scale=0.5)
#     out = visualizer.draw_dataset_dict(d)
#     print(type(out.get_image()[:, :, ::-1]))
#     output_path = f"./annotated_image.jpg"
#     cv2.imwrite(output_path, out.get_image()[:, :, ::-1])
#     print(f"Image saved to {output_path}")

def seg_and_pred(args):
    segmodel = load_segmentation_model(args.segment_model_path)
    classifymodel = load_classification_model(args.classify_model_path)
    os.makedirs(args.output_dir,exist_ok=True)
    for chunk in sorted(os.listdir(args.input_dir), key=lambda x: int(x.split("_")[1].split(".")[0])):
        print(chunk)
        chunk_path = os.path.join(args.input_dir,chunk)
        urls = read_json_processed(chunk_path)
        info_list = []
        for i in tqdm(urls):
            info = seg_and_pred_url(segmodel,classifymodel,i)
            info_list.append(info)
        out_chunk_path = os.path.join(args.output_dir,chunk)
        with open(out_chunk_path, 'w') as json_file:
            json.dump(info_list, json_file, indent=2) 

def single_pass(args,url):
    segmodel = load_segmentation_model(args.segment_model_path)
    classifymodel = load_classification_model(args.classify_model_path)
    if os.path.exists("./tmp"):
        shutil.rmtree("./tmp")
    seg_and_pred_url(segmodel,classifymodel,url)

def main():
    parser = argparse.ArgumentParser(description='Segment and classify pubmed data')
    parser.add_argument('--segment_model_path', type=str, required=True, help='Path of segmentation model')
    parser.add_argument('--classify_model_path', type=str, required=True, help='Path of classification model')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directories, containing one or multiple json files')
    parser.add_argument('--output_dir', type=str, required=True, help='Input directories, containing one or multiple json files')

    args = parser.parse_args()
    seg_and_pred(args)
    #single_pass(args,"https://openi.nlm.nih.gov/imgs/512/84/4042565/PMC4042565_nutrients-06-02077-g004.png")

    
    
if __name__ == "__main__":
    main()

