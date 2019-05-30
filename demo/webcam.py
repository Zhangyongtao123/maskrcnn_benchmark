# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import cv2
import os
from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

import time


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Webcam Demo")
    parser.add_argument(
        "--config-file",
        default="../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.7,
        help="Minimum score for the prediction to be shown",
    )
    parser.add_argument(
        "--min-image-size",
        type=int,
        default=224,
        help="Smallest size of the image to feed to the model. "
            "Model was trained with 800, which gives best results",
    )
    parser.add_argument(
        "--show-mask-heatmaps",
        dest="show_mask_heatmaps",
        help="Show a heatmap probability for the top masks-per-dim masks",
        action="store_true",
    )
    parser.add_argument(
        "--masks-per-dim",
        type=int,
        default=2,
        help="Number of heatmaps per dimension to show",
    )
    parser.add_argument(
        "opts",
        help="Modify model config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # load config from file and command-line arguments
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # prepare object that handles inference plus adds predictions on top of image
    coco_demo = COCODemo(
        cfg,
        confidence_threshold=args.confidence_threshold,
        show_mask_heatmaps=args.show_mask_heatmaps,
        masks_per_dim=args.masks_per_dim,
        min_image_size=args.min_image_size,
    )

    # added for video
    video_path = './1.MP4'
    cam = cv2.VideoCapture(video_path)
    fps = 30
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    # video_writer = cv2.VideoWriter(filename='./result.avi', fourcc=fourcc, fps=fps, frameSize=(640,480))
    video_writer = cv2.VideoWriter(filename='./result2.avi', fourcc=cv2.VideoWriter_fourcc(*'MJPG'), fps=fps,
                                   frameSize=(853, 480))
    while True:
        start_time = time.time()
        ret_val, img = cam.read()
        if img is None:
            break

        composite = coco_demo.run_on_opencv_image(img)
        # cv2.imwrite('a.jpg', composite)
        print("Time: {:.2f} s / img".format(time.time() - start_time))

        # cv2.imshow("COCO detections", composite)
        # if cv2.waitKey(1) == 27:
        #     break  # esc to quit

        # added by zhangyongtao

        composite = cv2.resize(composite, (853, 480))
        video_writer.write(composite)

        # # ./result1.avi  这个相对路径是对于py程序而言，所以avi视频文件被保存在py程序所在的文件夹
        # for i in range(1, 1841):
        #     p = i
        #     # print(str(p)+'.png'+'233333')
        #     if os.path.exists('E:/data/Caltech_jpg/set07/set07_V000_' + str(p) + '.jpg'):  # 判断图片是否存在
        #         img = cv2.imread(filename='E:/data/Caltech_jpg/set07/set07_V000_' + str(p) + '.jpg')
        #         # cv2.waitKey(10)
        #         img = cv2.resize(img, (640, 480))
        #         video_writer.write(img)
        #         print(str(p) + '.jpg' + ' done!')

    # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
