
from darkflow.net.build import TFNet

options = {"model": "cfg/yolo_custom.cfg", 
           "load": "bin/yolov2.weights",
           "batch": 8,
           "epoch": 100,
           "gpu": 1.0,
           "train": True,
           "annotation": "./new_data/annotations/",
           "dataset": "./new_data/images/"}


tfnet = TFNet(options)

tfnet.train()

tfnet.savepb()

## si queremos hacerlo por consola
# python flow --model cfg/yolo.cfg --load bin/yolov2.weights --train --annotation new_data\annots --dataset new_data\images --epoch 1



