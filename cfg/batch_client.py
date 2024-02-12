#!/usr/bin/env python3
from derecho.cascade.external_client import ServiceClientAPI
from derecho.cascade.external_client import TimestampLogger
import io
import numpy as np
import os
import sys,time,math,json
from PIL import Image
from python_udls.logging_flags import *
import torch
from torchvision import transforms
import pickle


IMAGE_DIRECTORY = './siemensimgs'

TOTAL_NUM_OBJ = 500
TOTAL_CAMERA = 8
TOTAL_ROUND = 3

OBJECT_POOLS_LIST = [
['/img_input', "VolatileCascadeStoreWithStringKey",0, None],
['/partial_result', "VolatileCascadeStoreWithStringKey",0, "/[0-9]+-"],
['/aggregate_result', "VolatileCascadeStoreWithStringKey", 0, None]
]

SEPERATE_UDL = True
if SEPERATE_UDL:
     OBJECT_POOLS_LIST = [
     ['/imgCrake', "VolatileCascadeStoreWithStringKey",0, None],
     ['/imgHole', "VolatileCascadeStoreWithStringKey",0, None],
     ['/partial_result', "VolatileCascadeStoreWithStringKey",0, "/[0-9]+-"],
     ['/aggregate_result', "VolatileCascadeStoreWithStringKey", 0, None]
     ]

def get_image_pathnames(directory, img_suffixes=['.jpg','.png', '.jpeg']):
     '''
     get the path names of all images under a directory
     '''
     pathnames = []
     for filename in os.listdir(directory):
          if (filename[-4:] in img_suffixes) or (filename[-5:] in img_suffixes):
               pathnames.append(os.path.join(directory, filename))
          if len(pathnames) == TOTAL_NUM_OBJ * TOTAL_CAMERA * TOTAL_ROUND:
               break
     return pathnames


def get_image(image_path):
     '''
     get image from image_path, and convert it to byte array
     '''
     i_image = Image.open(image_path)
     i_image = i_image.resize((640,640))
     if i_image.mode != "RGB":
          i_image = i_image.convert(mode="RGB")
     # byteIO = io.BytesIO()
     # i_image.save(byteIO, format='PNG')
     # byteArr = byteIO.getvalue()
     # byteArr = np.array(i_image).tobytes()
     # img_tensor = transform(i_image)
     return i_image

if __name__ == '__main__':
     # 0 - create Cascade client
     capi = ServiceClientAPI()
     my_id = capi.get_my_id()
     tl = TimestampLogger()
     
     # 1 - create object pool to accept and keep the result
     for object_pool_info in OBJECT_POOLS_LIST:
          if object_pool_info[3]:
               affinity_set_regex = object_pool_info[3]
               res = capi.create_object_pool(object_pool_info[0], object_pool_info[1],object_pool_info[2], affinity_set_regex=affinity_set_regex)
          else:
               res = capi.create_object_pool(object_pool_info[0], object_pool_info[1],object_pool_info[2])
          if res:
               res.get_result()
     # capi.create_object_pool('/aggregate_result', "PersistentCascadeStoreWithStringKey",0)

     # 2 - send the request to Cascade servers
     image_pathnames = get_image_pathnames(IMAGE_DIRECTORY)
     images = []
     for i in range(TOTAL_NUM_OBJ*TOTAL_CAMERA*TOTAL_ROUND):
          if i >= len(image_pathnames):
               break
          images.append(get_image(image_pathnames[i]))

     image_id = 0
     last_obj_time = int(time.perf_counter() * 1000)
     for obj_id in range(TOTAL_NUM_OBJ):
          last_round_time = int(time.perf_counter() * 1000)
          for round_id in range(TOTAL_ROUND):
               camera_ids = np.arange(TOTAL_CAMERA)
               np.random.shuffle(camera_ids)
               # Logging
               for camera_id in camera_ids:
                    extra_log_id = round_id * 1000 + camera_id
                    tl.log(CAMERA_SEND_TIME,capi.get_my_id(),obj_id,extra_log_id)
               while(int(time.perf_counter() * 1000) - last_round_time < 200):
                    time.sleep(0.0001)
               if SEPERATE_UDL:
                    last_round_time = time.perf_counter() * 1000
               # get batched images from camera
               input_images = []
               for camera_id in camera_ids:
                    img_value = images[image_id % len(image_pathnames)]
                    input_images.append(img_value)
                    image_id += 1
               # convert to byte array
               images_bytes = pickle.dumps(input_images)
               key = str(obj_id) + "-r" + str(round_id)
               extra_log_id = round_id * 1000
               tl.log(EXTERNAL_CLIENT_SEND_TIME,capi.get_my_id(),obj_id,extra_log_id)
               if SEPERATE_UDL:
                    tl.log(EXTERNAL_CLIENT_BEGIN_SEND_CRACK_TIME,capi.get_my_id(),obj_id,extra_log_id)
                    res = capi.put(f"/imgCrake/{key}",images_bytes,trigger=False,message_id=image_id)
                    tl.log(EXTERNAL_CLIENT_FINISH_SEND_CRACK_TIME,capi.get_my_id(),obj_id,extra_log_id)
                    if res:
                         res.get_result()
                    

                    tl.log(EXTERNAL_CLIENT_BEGIN_SEND_HOLE_TIME,capi.get_my_id(),obj_id,extra_log_id)
                    res = capi.put(f"/imgHole/{key}",images_bytes,trigger=False,message_id=image_id)
                    tl.log(EXTERNAL_CLIENT_FINISH_SEND_HOLE_TIME,capi.get_my_id(),obj_id,extra_log_id)

                    if res:
                         res.get_result()
               else:
                    capi.put(f"/img_input/{key}",images_bytes,trigger=False,message_id=image_id)
               tl.log(EXTERNAL_CLIENT_FINISH_SEND_TIME,capi.get_my_id(),obj_id,extra_log_id)
               if not SEPERATE_UDL:
                    last_round_time = time.perf_counter() * 1000
          while(int(time.perf_counter() * 1000) - last_obj_time < 2600):
               time.sleep(0.0001)
          last_obj_time = time.perf_counter() * 1000
          # time.sleep(2.6)
     tl.flush(f'client_timestamps.dat',False) 


