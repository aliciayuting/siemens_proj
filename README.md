# Example program that illustrate how to construct cascade to serve assemply line application.

## Run the program

#### 1.Start the server nodes

Under the server nodes' folder (n0, n1 in this example), start the server nodes by the command ```cascade_server```. (To start the server nodes from fresh, run ``` ./clear_log.sh ``` before start the servers)
One may note the initialization of the server node, and UDLs. For the two ML UDLs, they would load the models to GPU upon initialization.

#### 2.Setup the Object Pool

Under the client node's folder (n2 in this example), there are 2 steps to do.

First is to configure the object pool. This is done via ``` python setup.py ```, to run the python code written in setup.py. This command only needs to run once. 

Second is to start the camera with input, via ``` camera_connector_client.py ```. This function is to connect the camera inputs and put them to cascade servers (it serves similar purpose as vision_connector in Siemens setting). In this demo code, we simulate the image input, by loading the image from local folder siemensimg, and directly put to Cascade. To get fullflege solution, one can customize this file to have it connect to real cameras and send the received images upon receiving the images from the cameras. 


## Server and dataflow graph configuration

Under the directory of /siemens_proj/cfg, there are several configuration files of interest.

#### - dfgs.json.tmp

Defines the dataflow graph of the application. In this assembly line application, each input is going to be
processed by 3 UDLs: CrackDetectUDL, HoleDetectUDL, and AggregateUDL. The processing sequence and graph relation of their executions are defined in dfgs.json.tmp.

In the first vertex of the dataflow graph, it specifies the following: the input of the pipeline (individual image) is put to the object pool named "/img_input", which has 2 UDLs attached to this pathname (CrackDetectUDL, HoleDetectUDL). Once a server receives an object with the prefix of "/img_input", it triggers thoes two UDLs. The outputs of these two UDLs is going to be put to the object pool named "/partial_result".

In the second vertex of the dataflow graph, it specifices the following: the object pool of "/partial_result" is attached with the UDL named AggregateUDL. This UDL serves the purpose of aggregating the crack detection and hole detection results for the images with the same object ID. The result of this aggregation is stored to the object pool named "/aggregate_result".


#### -layout.json.tmp

Define the layout of the cascade's server nodes. In this layout, we use only two server nodes, where there is one shard per each subgroup, to run the service. We also created a example layout_2shards_example.json.tmp for your reference, if you want to experiment with more than 1 shard per subgroup.


## UDL (user defined logic function) code

Under python_udls directory, there are two main componenets.

#### -yolov5/

This is derived from YOLO repo from ultralitics, https://github.com/ultralytics/yolov5 . In the UDLs, we use the basic yolov5 model without any customization directly from the original repo.

#### -siemens_udls.py

This file defines all the UDLs we use in this example code. 

For CrackDetectUDL and HoleDetectUDL, these two UDLs load model object to GPU upon initialization, and run the model on object upon trigger. The lambda function got triggered is defined in the class function ocdp_handler(self,**kwargs). These handlers could be replaced with your customized model and process.

AggregateUDL is a UDL to aggregate the crack and hole detection result for each object. Because each object has total of 24 images (3 rounds as the object goes by the assembly line, where 8 cameras taking pictures from different angle in each round), this UDL aggregate the 48 model inference results that each image gets from two ML models (CrackDetection, HoleDetection). 

