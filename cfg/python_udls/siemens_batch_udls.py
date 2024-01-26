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
import pickle
import torch
from torchvision import transforms
import math
from logging_flags import *



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



"""       ------------     METHOD1. of batching      ------------      """


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
          print(f"batch CrackDetectUDL Constructed and YOLO model loaded to GPU")

     def __init__(self,conf_str):
          '''
          Constructor
          '''
          super(CrackDetectUDL,self).__init__(conf_str)
          self.capi = ServiceClientAPI()
          self.my_id = self.capi.get_my_id()
          if self.my_id < 3:
               device_name = 'cuda:0'
          else:
               device_name = 'cuda:1'
          self.device = torch.device(device_name)
          self.transform = transforms.Compose([transforms.ToTensor()])
          self.model = None
          self.categories = None
          self.weights= './python_udls/yolov5/yolov5s.pt'
          self.data = './python_udls/yolov5/data/coco128.yaml'
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
          blob = kwargs["blob"]
          res_key = key.split('-')[1]
          round_id = (res_key.split('_')[0])[1:] 
          # camera_id = (res_key.split('_')[1])[1:]
          extra_log_id = int(round_id)*1000 #+ int(camera_id)
          self.tl.log(BEGIN_CRACK_PRE_TIMESTAMP,self.my_id,obj_id,extra_log_id)
          # image pre-proccessing 
          image_array = pickle.loads(blob)
          images = [self.transform(image) for image in image_array]
          input_batch = torch.stack(images)
          input_batch = input_batch.to(self.device)
          # run inference
          self.tl.log(BEGIN_CRACK_DETECT_TIMESTAMP,self.my_id,obj_id,extra_log_id)
          pred, train_out  = self.model(input_batch, augment=False, visualize=False)
          # post-processing: TODO: draw bounding box?
          pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.25, classes=False, agnostic=False, max_det=1000)
          self.tl.log(FINISH_CRACK_DETECT_TIMESTAMP,self.my_id,obj_id,extra_log_id)
          # save result
          # serialize preds
          serialized_tensors = [pickle.dumps(tensor.cpu()) for tensor in pred]
          concatenated_bytes = b''.join(len(t).to_bytes(4, 'big') + t for t in serialized_tensors)
          new_key = "/partial_result/" + key + "_crack"
          self.capi.put(new_key, concatenated_bytes)
          self.tl.log(SEND_CRACK_RESULT_TO_NEXT_TIMESTAMP,self.my_id,obj_id,extra_log_id)

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
          self.capi = ServiceClientAPI()
          self.my_id = self.capi.get_my_id()
          if self.my_id < 3:
               device_name = 'cuda:0'
          else:
               device_name = 'cuda:1'
          self.device = torch.device(device_name)
          self.transform = transforms.Compose([transforms.ToTensor()])
          self.model = None
          self.categories = None
          self.weights= './python_udls/yolov5/yolov5s.pt'
          self.data = './python_udls/yolov5/data/coco128.yaml'
          if (self.my_id != 0):
               self.load_model()
          self.tl = TimestampLogger()

     def ocdpo_handler(self,**kwargs):
          '''
          The off-critical data path handler, gets executed when the UDL is triggered at server nodes
          '''
          key = kwargs["key"]
          obj_id = int(key.split('-')[0])
          blob = kwargs["blob"]
          res_key = key.split('-')[1]
          round_id = (res_key.split('_')[0])[1:] 
          # camera_id = (res_key.split('_')[1])[1:]
          extra_log_id = int(round_id)*1000 # + int(camera_id)
          self.tl.log(BEGIN_HOLE_PRE_TIMESTAMP,self.my_id,obj_id,extra_log_id)
          image_array = pickle.loads(blob)
          images = [self.transform(image) for image in image_array]
          input_batch = torch.stack(images)
          input_batch = input_batch.to(self.device)
          # run inference
          self.tl.log(BEGIN_HOLE_DETECT_TIMESTAMP,self.my_id,obj_id,extra_log_id)
          pred, train_out  = self.model(input_batch, augment=False, visualize=False)
          pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.25, classes=False, agnostic=False, max_det=1000)
          self.tl.log(FINISH_HOLE_DETECT_TIMESTAMP,self.my_id,obj_id,extra_log_id)
          emit_pred = np.array([t.cpu().numpy() for t in pred], dtype=object)
          serialized_tensors = [pickle.dumps(tensor.cpu()) for tensor in pred]
          concatenated_bytes = b''.join(len(t).to_bytes(4, 'big') + t for t in serialized_tensors)
          new_key = "/partial_result/" + key + "_hole"
          self.capi.put(new_key, concatenated_bytes)
          self.tl.log(SENT_HOLE_RESULT_TO_NEXT_TIMESTAMP,self.my_id,obj_id,extra_log_id)

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
          self.camera_per_obj = int(self.conf["camera_per_obj"])
          self.round_per_obj = int(self.conf["round_per_obj"])
          self.results = {} # map: {obj_id->{"hole": {(round_id, camera_id):result, ...}, "crack": [img1_hole_result, img2_hole_result, ...]}, ... }
          self.tl = TimestampLogger()
          self.capi = ServiceClientAPI()
          self.my_id = self.capi.get_my_id()
          print(f"AggregateUDL Constructed, camera_per_obj set to {self.camera_per_obj},round_per_obj set to {self.round_per_obj} ")

     def check_collect_all(self, obj_id):
          if obj_id not in self.results:
               return False
          if "crack" not in self.results[obj_id] or len(self.results[obj_id]["crack"]) < self.round_per_obj * self.camera_per_obj:
               return False
          if "hole" not in self.results[obj_id] or len(self.results[obj_id]["hole"]) < self.round_per_obj * self.camera_per_obj:
               return False
          return True

     def process_aggr_results(self, obj_id):
          '''
          Example use case of aggregate results
          Return True, if object has defect (i.e., is identified with crack or hole by a camera in a round)
          Return False, if object has no defect (i.e., no crack or hole identified for this object by any camera in any round)
          '''
          for crack_result in self.results[obj_id]["crack"].values():
               if len(crack_result) > 0:
                    print(f"Object{obj_id} crack_result has identified target object")
                    return True 
          for hole_result in self.results[obj_id]["hole"].values():
               if len(hole_result) > 0:
                    print(f"Object{obj_id} hole_result has identified target object")
                    return True  
          print(f"Object {obj_id} has no defect")
          return False

     def deserialize_blob(self, concatenated_bytes):
          deserialized_tensors = []
          i = 0
          while i < len(concatenated_bytes):
               # Read the length
               length = int.from_bytes(concatenated_bytes[i:i+4], 'big')
               i += 4
               tensor_bytes = concatenated_bytes[i:i+length]
               i += length
               tensor = pickle.loads(tensor_bytes)
               deserialized_tensors.append(tensor)
          return deserialized_tensors

     def ocdpo_handler(self,**kwargs):
          '''
          The off-critical data path handler, gets executed when the UDL is triggered at server nodes
          '''
          key = kwargs["key"] # in the format "[objID]-r[roundID]_c[cameraID]_[taskName]"(e.g. "0-r0_c0_crack")
          blob = kwargs["blob"]
          obj_id = int(key.split('-')[0])
          res_key = key.split('-')[1]
          round_id = (res_key.split('_')[0])[1:] 
          # camera_id = (res_key.split('_')[1])[1:]
          extra_log_id = int(round_id)*1000 # + int(camera_id)
          self.tl.log(BEGIN_AGGR_TIMESTAMP,self.my_id,obj_id,extra_log_id)
          task_name = res_key.split('_')[-1]
          if obj_id not in self.results:
               self.results[obj_id] = {}
          if task_name not in self.results[obj_id]:
               self.results[obj_id][task_name] = {} 
          # self.results[obj_id][task_name][img_info] = blob
          deserialized_results = self.deserialize_blob(blob)
          for i in range(len(deserialized_results)):
               camera_id = i   # assume the images batch is according to the camera order. TODO: edit this
               img_info = (round_id,camera_id)
               self.results[obj_id][task_name][img_info] = deserialized_results[i]
          if self.check_collect_all(obj_id):
               print(f"------- COLLECTED_ALL: object_id:{obj_id} -----")
               has_defect = self.process_aggr_results(obj_id)
               # Store a simple array [True/False] to represent if the product has defect. 
               # Could be encoded to a more informative object to store to this object key
               cascade_context.emit(str(obj_id), np.array(has_defect))
               self.tl.log(FINISH_AGGR_TIMESTAMP,self.my_id,obj_id,extra_log_id)
               # Flush the logging file
               if(int(obj_id) % LOGGING_POINT == 0 and FLUSH_RESULT):
                    self.tl.flush("server"+str(self.my_id)+"_timestamps.dat",False)
                    print("flushed server"+str(self.my_id)+"_timestamp.dat")
          else:
               self.tl.log(FINISH_AGGR_TIMESTAMP,self.my_id,obj_id,extra_log_id)

     def __del__(self):
          '''
          Destructor
          '''
          print(f"AggregateUDL destructor")
          pass