#!/usr/bin/env python3
from derecho.cascade.external_client import ServiceClientAPI
from derecho.cascade.external_client import TimestampLogger


OBJECT_POOLS_LIST = [
['/img_input', "VolatileCascadeStoreWithStringKey",0, None],
['/partial_result', "VolatileCascadeStoreWithStringKey",0, "/[0-9]+-"],
['/aggregate_result', "VolatileCascadeStoreWithStringKey", 0, None]
]

if __name__ == '__main__':
     # 0 - create Cascade client
     capi = ServiceClientAPI()
     my_id = capi.get_my_id()
     
     # 1 - create object pool to accept and keep the result
     for object_pool_info in OBJECT_POOLS_LIST:
          if object_pool_info[3]:
               affinity_set_regex = object_pool_info[3]
               res = capi.create_object_pool(object_pool_info[0], object_pool_info[1],object_pool_info[2], affinity_set_regex=affinity_set_regex)
          else:
               res = capi.create_object_pool(object_pool_info[0], object_pool_info[1],object_pool_info[2])
          if res:
               res.get_result()
          
