#!/usr/bin/env python3
from derecho.cascade.external_client import ServiceClientAPI
from derecho.cascade.external_client import TimestampLogger
import io
import numpy as np
import os
import sys,time,math,json
from PIL import Image


IMAGE_DIRECTORY = './siemensimgs'

TOTAL_NUM_OBJ = 1
TOTAL_CAMERA = 8
TOTAL_ROUND = 3

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
     byteIO = io.BytesIO()
     i_image.save(byteIO, format='PNG')
     byteArr = byteIO.getvalue()
     # byteArr = np.array(i_image).tobytes()
     return byteArr

if __name__ == '__main__':
     # 0 - create Cascade client
     capi = ServiceClientAPI()
     my_id = capi.get_my_id()
     
     # 1 - create object pool to accept and keep the result
     capi.create_object_pool('/img_input', "VolatileCascadeStoreWithStringKey",0)
     affinity_set_regex = "/[0-9]+-"
     capi.create_object_pool('/partial_result', "VolatileCascadeStoreWithStringKey",0, affinity_set_regex=affinity_set_regex)
     capi.create_object_pool('/aggregate_result', "VolatileCascadeStoreWithStringKey", 0)
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
                    extra_log_id = round_id * TOTAL_CAMERA + camera_id
               while(int(time.perf_counter() * 1000) - last_round_time < 200):
                    time.sleep(0.0001)
               last_round_time = time.perf_counter() * 1000

               for camera_id in camera_ids:
                    input_value = images[image_id % len(image_pathnames)]
                    key = str(obj_id) + "-r" + str(round_id) + "_c" + str(camera_id)
                    extra_log_id = round_id * TOTAL_CAMERA + camera_id
                    capi.put(f"/img_input/{key}",input_value,trigger=False,message_id=image_id)
                    image_id += 1
                    asyc_noise = np.random.exponential(scale=0.001)
                    time.sleep(asyc_noise)
          while(int(time.perf_counter() * 1000) - last_obj_time < 2600):
               time.sleep(0.0001)
          last_obj_time = time.perf_counter() * 1000               


