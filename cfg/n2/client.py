#!/usr/bin/env python3
from derecho.cascade.external_client import ServiceClientAPI
from derecho.cascade.external_client import TimestampLogger
import io
import numpy as np
import os
import sys,time,math,json
from PIL import Image


IMAGE_DIRECTORY = './siemensimgs'
TOTAL_NUM = 1

def get_image_pathnames(directory, img_suffixes=['.jpg','.png', '.jpeg']):
     '''
     get the path names of all images under a directory
     '''
     pathnames = []
     for filename in os.listdir(directory):
          if (filename[-4:] in img_suffixes) or (filename[-5:] in img_suffixes):
               pathnames.append(os.path.join(directory, filename))
          if len(pathnames) == TOTAL_NUM:
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
     res = capi.create_object_pool('/crack', "VolatileCascadeStoreWithStringKey",0)
     res = capi.create_object_pool('/crack_result', "PersistentCascadeStoreWithStringKey",0)
     
     # 2 - send the request to Cascade servers
     image_pathnames = get_image_pathnames(IMAGE_DIRECTORY)
     message_id = 0
     for message_id in range(TOTAL_NUM):
          input_img = image_pathnames[message_id % len(image_pathnames)]
          input_value = get_image(input_img)
          print(len(input_value))
          capi.put(f"/crack/{message_id}",input_value,trigger=False,message_id=message_id)

