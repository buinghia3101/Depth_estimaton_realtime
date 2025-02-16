from Depth_estimate import Depth_Estimation
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


if __name__=="__main__":

    ROOT='archive/depth_selection'
    # img=cv2.imread("depth_selection\\test_depth_completion_anonymous\\image\\0000000000.png")
    # model_detection=YOLO('yolov8s.pt')
    model_depth=Depth_Estimation()
    # depth_map,_=model_depth.inference(img)
    # cv2.imwrite("results.png",depth_map)
    os.makedirs("depth",exist_ok=True)
    for roots,dirs,files in os.walk(ROOT):
        if len(dirs)==0 and os.path.basename(roots)=="image" :
            
            for file in files:
                file_name=os.path.join(roots,file)
                img=cv2.imread(file_name)
                depth_map,_=model_depth.inference(img)
                cv2.imwrite(os.path.join("depth",file),depth_map)
                print("[INFO] save done {}".format(os.path.join("depth",file)))
                