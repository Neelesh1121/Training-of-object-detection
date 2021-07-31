# Training of Object Detection Model for Fruits Detection 
In this repository, I have implemented the TensorFlow Object Detection API for Training a custom object detection model using Tensorflow 2.5.0 version.
![12](https://user-images.githubusercontent.com/24295100/127749958-080ae570-ee59-4f2d-a912-72d4a7b19c02.jpg)

## Steps for Installation
To train models, we have to install the required library for object detection. I am mentioning the required steps pointwise here: 
* Create a virtual env and install the required Tensorflow version. 
* To create a new virtual env in conda: `conda create -n tensorflow pip python=3.9`
* Activate the virtual env : `conda activate tensorflow`
* Install tensorflow version 2 : `pip install --ignore-installed --upgrade tensorflow==2.5.0`
* Once your tensorflow installation is completed, you test it by running this command : `python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"`
* Now, we can install the Tensorflow Object detection API from two methods, one is cloning the repository by git clone command or you can simply download it as zip file and then on your local system you can unzip it. To install it from git command : 
`git clone https://github.com/tensorflow/models.git`
* Tensorflow Object Detection API uses Protobufs to configure model and training parameters. Before the framework can be used, the Protobuf libraries must be downloaded and compiled. We should run this command from /models/research/:  
`protoc object_detection/protos/*.proto --python_out=.`
* In Tensorflow 2.x version of object detection API "pycocotools' ' listed as a dependency we need to install this. Generally it gets installed when installing the object detection API but there could be multiple reasons of issues so we can install it separately by running these four commands:
```
git clone https://github.com/cocodataset/cocoapi.git
cd cocoa api/PythonAPI
make
cp -r pycocotools <PATH_TO_models>/models/research/
```

* Now the dependency and required libraries installation is completed. After completing upto these 8 points, we will install object detection API, this could be done by running the following commands from models/research directory :
```
cp object_detection/packages/tf2/setup.py .
# python -m pip install --use-feature=2020-resolver .
```

* You can also test the installation by running this command from same directory: 
`python object_detection/builders/model_builder_tf2_test.py`
![model_test](https://user-images.githubusercontent.com/24295100/127745744-bf77efd4-e9d9-4045-b869-4f02b9bc44bd.PNG)
 
 ## Training
 * create a labelmap.pbtxt file, in which we will map to objects from an id. It would be placed in annotations folder
 *
 * First we will create tfrecord file from training and testing data. You can place your annotated data in the image directory.
 ```
 # Create train data:
python generate_tfrecord.py -x [PATH_TO_IMAGES_FOLDER]/train -l [PATH_TO_ANNOTATIONS_FOLDER]/label_map.pbtxt -o [PATH_TO_ANNOTATIONS_FOLDER]/train.record

# Create test data:
python generate_tfrecord.py -x [PATH_TO_IMAGES_FOLDER]/test -l [PATH_TO_ANNOTATIONS_FOLDER]/label_map.pbtxt -o [PATH_TO_ANNOTATIONS_FOLDER]/test.record
```

* We are using a tensorflow pre-trained model so we have to create a training configuration file. From pipeline.config, we need to make changes in certain parameters according to our requirements :
```
num_classes: 3 # Set this to the number of different label classes
type: "ssd_mobilenet_v1_fpn_keras"
weight: 3.9999998989515007e-05 (L2 regularizer weight)
max_detections_per_class: 100( total number of detection per class)
max_total_detections: 100( total number of detections)
batch_size: 8 # Increase/Decrease this value depending on the available memory (Higher values require more memory and vice-versa)
epochs: 8000
learning_rate_base: 0.003999999910593033
total_steps: 200
warmup_learning_rate: 0.0013333000242710114
fine_tune_checkpoint: "pre-trained-models/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0" # Path to checkpoint of pre-trained model
fine_tune_checkpoint_type: "detection" # Set this to "detection" since we want to be training the detection model
label_map_path: "annotations/label_map.pbtxt" # Path to label map file
input_path: "annotations/train.record" # Path to training TFRecord file
use_moving_averages: false
input_path: "annotations/test.record" # Path to testing TFRecord
```

## Start Training 
* To start the training, need to run the model_main_tf2.py file with path of the model and pipeline.config
```
python model_main_tf2.py --model_dir=models/ssd_mobilenet_v1_fpn --pipeline_config_path=models/ssd_mobilenet_v1_fpn/pipeline.config
```

## Export the Trained Model
* Once training is got completed you can export the trained model which you can use in inference deployments
```
python exporter_main_v2.py --input_type image_tensor --pipeline_config_path models/ssd_mobilenet_v1_fpn/pipeline.config --trained_checkpoint_dir models/ssd_mobilenet_v1_fpn --output_directory exported-models/my_trained_model
```

## Test the Model
* To test the model, I have written ```inference_object_detection.py``` in which by changing the exported model path and you test images location you will get the detected output.
```
python3 inference_object_detection.py
```
