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
    for img in os.listdir(img_dir):
        img_dir = os.path.join(path,f"./img/{img}")
        txt_file = os.path.join(path,f"text/{img}"+".txt")
        #print(txt_file)
        #read_coord(txt_file)
        if os.path.exists(txt_file):
            coord = read_coord(txt_file)
            if coord != []:
                #print(coord)
                cnt += 1
                record = {}
                height, width = cv2.imread(img_dir).shape[:2]
                #print(f'{height, width}')
                record["file_name"] = img_dir
                record["image_id"] = cnt-1
                record["height"] = height
                record["width"] = width
                objs = []
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
    # for idx, v in enumerate(imgs_anns.values()):
    #     record = {}

    #     filename = os.path.join(img_dir, v["filename"])
    #     height, width = cv2.imread(filename).shape[:2]

    #     record["file_name"] = filename
    #     record["image_id"] = idx
    #     record["height"] = height
    #     record["width"] = width

    #     annos = v["regions"]
    #     objs = []
    #     for _, anno in annos.items():
    #         assert not anno["region_attributes"]
    #         anno = anno["shape_attributes"]
    #         px = anno["all_points_x"]
    #         py = anno["all_points_y"]
    #         poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
    #         poly = [p for x in poly for p in x]

    #         obj = {
    #             "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
    #             "bbox_mode": BoxMode.XYXY_ABS,
    #             "segmentation": [poly],
    #             "category_id": 0,
    #         }
    #         objs.append(obj)
    #     record["annotations"] = objs
    #     dataset_dicts.append(record)
    return dataset_dicts

for d in ["train"]:
    DatasetCatalog.register("subfigure_" + d, lambda d=d: get_balloon_dicts("./img"))
balloon_metadata = MetadataCatalog.get("subfigure_train")

class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_writers(cls, cfg):
        """
        Build writers that log metrics. By default, it includes a TensorboardXWriter.
        This function returns a list of writers without TensorboardXWriter.
        """
        from detectron2.utils.events import CommonMetricPrinter, JSONWriter

        # Writers to use: CommonMetricPrinter and JSONWriter
        writers = [
            CommonMetricPrinter(cfg.OUTPUT_DIR),
            JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
        ]
        return writers

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
cfg.SOLVER.IMS_PER_BATCH = 4  # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 5000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
cfg.OUTPUT_DIR = "./out"
#print(cfg)
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)

    # Start training
trainer.resume_or_load(resume=False)
trainer.train()
#trainer = DefaultTrainer(cfg)
#trainer.resume_or_load(resume=False)
#trainer.train()