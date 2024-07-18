import torch, detectron2
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
import model
import argparse
from preprocess import preprocess, preprocess_url
from detectron2.utils.visualizer import ColorMode
setup_logger()
path = os.getcwd()
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("subfigure_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 4  # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 5000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
cfg.OUTPUT_DIR = "./out"
segmodel = build_model(cfg)  # returns a torch.nn.Module
DetectionCheckpointer(segmodel).load("../out/model_final.pth")
segmodel.eval()
#print(cfg)
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
predictor = DefaultPredictor(cfg)
classifyModel = torch.load("../classifier_acl_aug_resume.pt")
classifyModel.to('cuda')
classifyModel.eval()
figure_name = ["3Dreconstruction","illustrations","flowchart","heatmap","LineGraph","microscopy","photography",'Radiology',"scatterplot",'signals_waves','tables','barcharts','boxcharts']
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
        print(transposed_array.shape)
        dataset_dicts = []
        record = {}
        record['image'] = torch.from_numpy(transposed_array)
        record['image_orig'] = im
        dataset_dicts.append(record)
    return dataset_dicts

def batched_inference(path_list,batch=10,mode="local"):
    predicted_classes = []
    for i in tqdm(range(0,len(path_list),batch)):
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
        print(reshaped_out.shape)
        mean_out = reshaped_out.mean(dim=1)
        
        # Take the max along the first dimension
        max_mean_out = mean_out.argmax(dim=1)
        predicted_classes.extend([figure_name[i] for i in max_mean_out])
    print(predicted_classes)
    return predicted_classes


def seg_and_pred_url(url):
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
        if(instances.scores[i] > 0.5):
            x1, y1, x2, y2 = box.int().tolist()
            coords.append([x1, y1, x2, y2])
            segmented_image = data[0]['image_orig'][y1:y2, x1:x2]
            segmented_path = os.path.join(tmp_dir,f"./segmented_{i}.png")
            cv2.imwrite(segmented_path, segmented_image)
            print("Written!")
            paths.append(segmented_path)
    print(f"Segmented {len(paths)} images, classifying...")
    predicted_classes = batched_inference(paths,batch=10)
    info['segments'] = coords
    info['predicted_classes'] = predicted_classes
    return info
    #shutil.rmtree(tmp_dir)



#info = seg_and_pred_url("https://openi.nlm.nih.gov/imgs/512/94/4042575/PMC4042575_nutrients-06-01913-g002.png")
urls = [i['url'] for i in read_json_processed("../pred/pmc/chunk_1_pred.json")]
info_list = []
for i in urls[:20]:
    info = seg_and_pred_url(i)
    info_list.append(info)

with open('data.json', 'w') as json_file:
    json.dump(info_list, json_file, indent=4)  # indent=4 for pretty formatting
# dataset_dicts = get_balloon_dicts("./img")
# for d in random.sample(dataset_dicts, 3):
#     img = cv2.imread(d["file_name"])
#     visualizer = Visualizer(img[:, :, ::-1], metadata=balloon_metadata, scale=0.5)
#     out = visualizer.draw_dataset_dict(d)
#     print(type(out.get_image()[:, :, ::-1]))
#     output_path = f"./annotated_image.jpg"
#     cv2.imwrite(output_path, out.get_image()[:, :, ::-1])
#     print(f"Image saved to {output_path}")

