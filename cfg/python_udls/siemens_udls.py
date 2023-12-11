#!/usr/bin/env python3
import cascade_context
from derecho.cascade.udl import UserDefinedLogic
from derecho.cascade.member_client import ServiceClientAPI
import numpy as np
import json
import platform
import sys
from pathlib import Path
import os, io
from PIL import Image
import torch
from torchvision import transforms
import math


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] / 'yolov5'
CWD =  FILE.parents[0]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov5') not in sys.path:
    sys.path.append(str(ROOT / 'yolov5'))  # add yolov5 ROOT to PATH
if str(ROOT / 'yolov5' / 'models') not in sys.path:
    sys.path.append(str(ROOT /  'yolov5' / 'models'))  
if str(ROOT / 'yolov5' / 'utils') not in sys.path:
    sys.path.append(str(ROOT / 'yolov5' / 'utils'))  
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


from models.common import DetectMultiBackend
from utils.general import (LOGGER, Profile, 
                           non_max_suppression)
from utils.torch_utils import select_device, smart_inference_mode





class CrackDetectUDL(UserDefinedLogic):
     '''
     CrackDetectUDL is a UDL that utilize YOLO model to detect cracks in an image
     '''
     
     def load_model(self):
          '''
          Load YOLOv5s model
          '''
          self.model = DetectMultiBackend(self.weights, 
                                          device=self.device, 
                                          dnn=True, data=self.data, fp16=True)

     def __init__(self,conf_str):
          '''
          Constructor
          '''
          super(CrackDetectUDL,self).__init__(conf_str)
          self.capi = ServiceClientAPI()
          device_name = 'cuda:0'
          self.device = torch.device(device_name)
          self.transform = transforms.Compose([transforms.ToTensor()])
          self.model = None
          self.categories = None
          self.weights= './python_udl/yolov5/yolov5s.pt'
          self.data = './python_udl/yolov5/data/coco128.yaml'
          self.load_model()
          print(f"CrackDetectUDL Constructed and YOLO model loaded to GPU {device_name}")


     def ocdpo_handler(self,**kwargs):
          '''
          The off-critical data path handler, gets executed when the UDL is triggered at server nodes
          '''
          key = kwargs["key"]
          message_id = int(key[key.rfind("/") + 1:])
          blob = kwargs["blob"]
          image = Image.open(io.BytesIO(blob))
          # image pre-proccessing 
          input_image = self.transform(image).unsqueeze(0)
          input_image = input_image.to(self.device)
          input_image /= 255
          # run inference
          pred, train_out  = self.model(input_image, augment=False, visualize=False)
          # post-processing: TODO: draw bounding box?
          pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.25, classes=False, agnostic=False, max_det=1000)
          print(f"CrackDetectUDL ocdpo_handler: message_id={message_id}, result={len(pred)}")
          print(type(pred))
          print(pred)
          # save result
          new_key = "/crack_result/" + str(message_id)
          bytes_list = [t.cpu().numpy().tobytes().decode('utf-8') for t in pred]
          combined_bytes = ' '.join(bytes_list).encode('utf-8')  
          self.capi.put(new_key, combined_bytes)
          # stacked_pred = np.stack([t.cpu().numpy() for t in pred])
          # cascade_context.emit(new_key, stacked_pred)
          

     def __del__(self):
          '''
          Destructor
          '''
          print(f"CrackDetectUDL destructor")
          pass


class HoleDetectUDL(UserDefinedLogic):
     '''
     HoleDetectUDL is a UDL that utilize YOLO model to detect holes in an image
     '''
     
     def load_model(self):
          '''
          Load YOLOv5s model
          '''
          self.model = DetectMultiBackend(self.weights, 
                                          device=self.device, 
                                          dnn=True, data=self.data, fp16=True)

     def __init__(self,conf_str):
          '''
          Constructor
          '''
          super(CrackDetectUDL,self).__init__(conf_str)
          device_name = 'cuda:0'
          self.device = torch.device(device_name)
          self.transform = transforms.Compose([transforms.ToTensor()])
          self.model = None
          self.categories = None
          self.weights= './python_udl/yolov5/yolov5s.pt'
          self.data = './python_udl/yolov5/data/coco128.yaml'
          self.load_model()
          print(f"HoleDetectUDL Constructed and YOLO model loaded to GPU {device_name}")


     def ocdpo_handler(self,**kwargs):
          '''
          The off-critical data path handler, gets executed when the UDL is triggered at server nodes
          '''
          key = kwargs["key"]
          message_id = int(key[key.rfind("/") + 1:])
          blob = kwargs["blob"]
          image = Image.open(io.BytesIO(blob))
          input_image = self.transform(image).unsqueeze(0)
          input_image = input_image.to(self.device)
          input_image /= 255
          pred, train_out  = self.model(input_image, augment=False, visualize=False)
          pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.25, classes=False, agnostic=False, max_det=1000)
          print(f"HoleDetectUDL ocdpo_handler: message_id={message_id}, result={pred}")

     def __del__(self):
          '''
          Destructor
          '''
          print(f"HoleDetectUDL destructor")
          pass


class AggregateUDL(UserDefinedLogic):
     '''
     AggregateUDL is a UDL that aggregate the classification results across cameras for each objects
     '''

     def __init__(self,conf_str):
          '''
          Constructor
          '''
          super(AggregateUDL,self).__init__(conf_str)
          self.results = {} # map: {obj_id->[img1_result,img2_result, ... ], ... }
          print(f"AggregateUDL Constructed")

     def ocdpo_handler(self,**kwargs):
          '''
          The off-critical data path handler, gets executed when the UDL is triggered at server nodes
          '''
          key = kwargs["key"]
          blob = kwargs["blob"]
          pass

     def __del__(self):
          '''
          Destructor
          '''
          print(f"AggregateUDL destructor")
          pass
