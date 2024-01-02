#!/usr/bin/env python3
import cascade_context
from derecho.cascade.member_client import ServiceClientAPI
from derecho.cascade.member_client import TimestampLogger
from derecho.cascade.udl import UserDefinedLogic
import numpy as np
import json
import platform
import re
import sys
from pathlib import Path
import os, io, time
from PIL import Image
import torch
from torchvision import transforms
import math
from setup import *


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
          print(f"CrackDetectUDL Constructed and YOLO model loaded to GPU")

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
          self.my_id = self.capi.get_my_id()
          if (self.my_id != 0):
               self.load_model()
          self.tl = TimestampLogger()
          self.last_img_collected_num = 0
          


     def ocdpo_handler(self,**kwargs):
          '''
          The off-critical data path handler, gets executed when the UDL is triggered at server nodes
          '''
          key = kwargs["key"]
          obj_id = int(key.split('-')[0])
          res_key = key.split('-')[1]
          round_id = int(res_key.split('_')[0])
          camera_id = int(res_key.split('_')[1])
          extra_log_id = round_id * TOTAL_CAMERA + camera_id
          blob = kwargs["blob"]
          self.tl.log(BEGIN_CRACK_PRE_TIMESTAMP,self.my_id,obj_id,extra_log_id)
          image = Image.open(io.BytesIO(blob))
          # image pre-proccessing 
          input_image = self.transform(image).unsqueeze(0)
          input_image = input_image.to(self.device)
          input_image /= 255
          # run inference
          self.tl.log(BEGIN_CRACK_DETECT_TIMESTAMP,self.my_id,obj_id,extra_log_id)
          pred, train_out  = self.model(input_image, augment=False, visualize=False)
          # post-processing: TODO: draw bounding box?
          pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.25, classes=False, agnostic=False, max_det=1000)
          self.tl.log(FINISH_CRACK_DETECT_TIMESTAMP,self.my_id,obj_id,extra_log_id)
          # print(f"CrackDetectUDL ocdpo_handler: message_id={key}, result={len(pred)}")
          # save result
          stacked_pred = np.stack([t.cpu().numpy() for t in pred])
          new_key = key + "_crack"
          cascade_context.emit(new_key, stacked_pred)
          # new_key = "/crack_result/" + str(key)
          # bytes_list = [t.cpu().numpy().tobytes().decode('utf-8') for t in pred]
          # combined_bytes = ' '.join(bytes_list).encode('utf-8')  
          # self.capi.put(new_key, combined_bytes)
          if(obj_id == TOTAL_NUM_OBJ -1 and FLUSH_RESULT):
               self.tl.flush("crack_timestamps.dat",False)
               print("flushed crack_timestamp.dat")
               # self.last_img_collected_num += 1
               # if (self.last_img_collected_num == TOTAL_CAMERA * TOTAL_ROUND):
               #      self.tl.flush("crack_timestamps.dat",False)
          
          

     def __del__(self):
          '''
          Destructor
          '''
          print(f"CrackDetectUDL destructor")


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
          print(f"HoleDetectUDL Constructed and YOLO model loaded to GPU")

     def __init__(self,conf_str):
          '''
          Constructor
          '''
          super(HoleDetectUDL,self).__init__(conf_str)
          device_name = 'cuda:1'
          self.capi = ServiceClientAPI()
          self.device = torch.device(device_name)
          self.transform = transforms.Compose([transforms.ToTensor()])
          self.model = None
          self.categories = None
          self.weights= './python_udl/yolov5/yolov5s.pt'
          self.data = './python_udl/yolov5/data/coco128.yaml'
          self.my_id = self.capi.get_my_id()
          if (self.my_id != 0):
               self.load_model()
          self.tl = TimestampLogger()

     def ocdpo_handler(self,**kwargs):
          '''
          The off-critical data path handler, gets executed when the UDL is triggered at server nodes
          '''
          key = kwargs["key"]
          obj_id = int(key.split('-')[0])
          res_key = key.split('-')[1]
          round_id = int(res_key.split('_')[0])
          camera_id = int(res_key.split('_')[1])
          extra_log_id = round_id * TOTAL_CAMERA + camera_id
          blob = kwargs["blob"]
          self.tl.log(BEGIN_HOLE_PRE_TIMESTAMP,self.my_id,obj_id,extra_log_id)
          image = Image.open(io.BytesIO(blob))
          input_image = self.transform(image).unsqueeze(0)
          input_image = input_image.to(self.device)
          input_image /= 255
          self.tl.log(BEGIN_HOLE_DETECT_TIMESTAMP,self.my_id,obj_id,extra_log_id)
          pred, train_out  = self.model(input_image, augment=False, visualize=False)
          pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.25, classes=False, agnostic=False, max_det=1000)
          # print(f"HoleDetectUDL ocdpo_handler: message_id={key}, result={pred}")
          self.tl.log(FINISH_HOLE_DETECT_TIMESTAMP,self.my_id,obj_id,extra_log_id)
          stacked_pred = np.stack([t.cpu().numpy() for t in pred])
          new_key = key + "_hole"
          cascade_context.emit(new_key, stacked_pred)
          if(obj_id == TOTAL_NUM_OBJ-1 and FLUSH_RESULT):
               self.tl.flush("hole_timestamps.dat",False)
               print("flushed hole_timestamp.dat")

     def __del__(self):
          '''
          Destructor
          '''
          print(f"HoleDetectUDL destructor")
          self.tl.flush("hole_timestamps.dat",False)


class AggregateUDL(UserDefinedLogic):
     '''
     AggregateUDL is a UDL that aggregate the classification results across cameras for each objects
     '''

     def __init__(self,conf_str):
          '''
          Constructor
          '''
          super(AggregateUDL,self).__init__(conf_str)
          self.conf = json.loads(conf_str)
          self.img_count_per_obj = int(self.conf["img_count_per_obj"])
          self.results = {} # map: {obj_id->{"hole": {(round_id, camera_id):result, ...}, "crack": [img1_hole_result, img2_hole_result, ...]}, ... }
          self.tl = TimestampLogger()
          print(f"AggregateUDL Constructed, img_count_per_obj set to {self.img_count_per_obj}")

     def check_collect_all(self, obj_id, task_name):
          if obj_id not in self.results:
               return False
          if "crack" not in self.results[obj_id] or len(self.results[obj_id]["crack"]) < self.img_count_per_obj:
               return False
          if "hole" not in self.results[obj_id] or len(self.results[obj_id]["hole"]) < self.img_count_per_obj:
               return False
          return True

     def ocdpo_handler(self,**kwargs):
          '''
          The off-critical data path handler, gets executed when the UDL is triggered at server nodes
          '''
          key = kwargs["key"]
          print(f"aggr key: {key}")
          blob = kwargs["blob"]
          obj_id = key.split('-')[0]
          res_key = key.split('-')[1]
          round_id = res_key.split('_')[0]
          camera_id = res_key.split('_')[1]
          task_name = res_key.split('_')[2]
          # print(f"obj_id:{obj_id}, round_id:{round_id}, camera_id:{camera_id}, task_name:{task_name}")
          if obj_id not in self.results:
               self.results[obj_id] = {}
          if task_name not in self.results[obj_id]:
               self.results[obj_id][task_name] = {}
          img_info = round_id + "-" + camera_id
          self.results[obj_id][task_name][img_info] = blob
          if self.check_collect_all(obj_id, task_name):
               print(f"------- COLLECTED_ALL: object_id:{obj_id} -----")
          

     def __del__(self):
          '''
          Destructor
          '''
          print(f"AggregateUDL destructor")
          pass
