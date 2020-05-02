import cv2
from darkflow.net.build import TFNet
import matplotlib.pyplot as plt

%config InlineBackend.figure_format = 'svg'


options = {
 'model': 'cfg/yolo.cfg',
 'load': 10,
 'threshold': 0.3,
 'backup':'ckpt/'
}

tfnet2 = TFNet(options)

tfnet2.load_from_ckpt()


original_img = cv2.imread('new_data/images/00001.jpg')
original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
results = tfnet2.return_predict(original_img)
print(results)