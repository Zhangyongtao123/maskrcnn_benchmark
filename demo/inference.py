import cv2

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

config_file = "../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

coco_demo = COCODemo(
    cfg,
    min_image_size=600,
    confidence_threshold=0.7,
)
# load image and then run prediction
image = cv2.imread("789.jpg")

predictions = coco_demo.run_on_opencv_image(image)
cv2.imwrite("test.jpg", predictions)