import cv2
import torch
import time
import numpy as np


class Depth_Estimation():
    def __init__(self,model_type="DPT_Hybrid") -> None:
        self.model= torch.hub.load("intel-isl/MiDaS", model_type)
        # Move model to GPU if available
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

        # Load transforms to resize and normalize the image
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform
        
        self.Q= np.array(([1.0, 0.0, 0.0, -160.0],
              [0.0, 1.0, 0.0, -120.0],
              [0.0, 0.0, 0.0, 350.0],
              [0.0, 0.0, 1.0/90.0, 0.0]),dtype=np.float32)
        
    def preprocess(self,img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
         # Apply input transforms
        input_batch = self.transform(img).to(self.device)
        return input_batch
    


    def inference(self,img):
        input_batch=self.preprocess(img)
            # Prediction and resize to original resolution
        with torch.no_grad():

            prediction = self.model(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

            depth_map = prediction.cpu().numpy()

            depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

            #Reproject points into 3D
            points_3D = cv2.reprojectImageTo3D(depth_map, self.Q, handleMissingValues=False)


            


            #Get rid of points with value 0 (i.e no depth)
            mask_map = depth_map > 0.45

            #Mask colors and points. 
            output_points = points_3D[mask_map]
            # output_colors = img[mask_map]

            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


            depth_map = (depth_map*255).astype(np.uint8)
            # depth_map = cv2.applyColorMap(depth_map , cv2.COLORMAP_MAGMA)
            return depth_map,output_points
        



        





