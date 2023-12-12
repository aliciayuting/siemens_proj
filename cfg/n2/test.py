#!/usr/bin/env python3
from derecho.cascade.external_client import ServiceClientAPI
from derecho.cascade.external_client import TimestampLogger
import io
import numpy as np
import os
import sys,time,math,json
from PIL import Image



if __name__ == '__main__':
     # 0 - create Cascade client
     capi = ServiceClientAPI()
     my_id = capi.get_my_id()
     
     # 3 - send the request to Cascade servers
     res = capi.get(f"/crack/0",stable=True)
     if res:
          obj = res.get_result()
          print(obj)
