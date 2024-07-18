import torch, detectron2
from detectron2.engine import DefaultTrainer
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
print("detectron2:", detectron2.__version__)
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from tqdm import tqdm

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
# Now 'coordinates' contains a list of tuples, each with 4 float values


from detectron2.structures import BoxMode
path = os.getcwd()
print(path)
def get_balloon_dicts(img_dir):
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
        if os.path.exists(txt_file):
            coord = read_coord(txt_file)
            if coord != []:
                #print(coord)
                cnt += 1
                for coordinate in coord:
                    x1, y1, x2, y2 = coordinate  # Assuming each coord is a tuple of 4 values
                    poly = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                    poly = [p for x in poly for p in x]  # Flatten the list of tuples
                    #print(poly)
                    obj = {
                        "bbox": [x1, y1, x2, y2],
                        "bbox_mode": BoxMode.XYWH_ABS,
                        "segmentation": [poly],
                        "category_id": 0,  # Replace with actual category_id if needed
                    }
                    objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

def get_validation(img_dir):
    cnt = 0
    dataset_dicts = []
    base_dir = os.path.join(path,img_dir)
    for img in os.listdir(img_dir):
        image_dir = os.path.join(base_dir,img)
        dataset_dicts.append(image_dir)
    return dataset_dicts

get_balloon_dicts("./img")

for d in ["train"]:
    DatasetCatalog.register("subfigure_" + d, lambda d=d: get_balloon_dicts("./single_test/out"))
balloon_metadata = MetadataCatalog.get("subfigure_train")

# dataset_dicts = get_balloon_dicts("./img")
# for d in random.sample(dataset_dicts, 3):
#     img = cv2.imread(d["file_name"])
#     visualizer = Visualizer(img[:, :, ::-1], metadata=balloon_metadata, scale=0.5)
#     out = visualizer.draw_dataset_dict(d)
#     print(type(out.get_image()[:, :, ::-1]))
#     output_path = f"./annotated_image.jpg"
#     cv2.imwrite(output_path, out.get_image()[:, :, ::-1])
#     print(f"Image saved to {output_path}")
print("Numpy version:", np.__version__)
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("subfigure_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 32   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
cfg.OUTPUT_DIR = "./out"
#print(cfg)
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
predictor = DefaultPredictor(cfg)
#trainer = DefaultTrainer(cfg)
#trainer.resume_or_load(resume=False)
#trainer.train()

from detectron2.utils.visualizer import ColorMode
dataset_dicts = get_validation("./single_test/scatters")
for d in tqdm(dataset_dicts):
    im = cv2.imread(d)
    print(d)
    name = d.split("/")[-1]
    name_dir = name.rstrip(".png")
    print(name)
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    print(type(outputs["instances"]))
    print(type(outputs["instances"].pred_boxes.tensor))
    print(outputs["instances"].pred_boxes.tensor)
    boxes = outputs["instances"].pred_boxes.tensor
    os.makedirs(f"./single_test_out/{name_dir}",exist_ok=True)
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.int().tolist()
        segmented_image = im[y1:y2, x1:x2]

        # Save the segmented image
        cv2.imwrite(f"./single_test_out/{name_dir}/segmented_{i}.png", segmented_image)

        # Convert BGR (OpenCV format) to RGB (matplotlib format)

    v = Visualizer(im[:, :, ::-1],
                   metadata=balloon_metadata,
                   scale=0.5,
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    #cv2.imwrite(f"./img_test_out/{name}", out.get_image()[:, :, ::-1])