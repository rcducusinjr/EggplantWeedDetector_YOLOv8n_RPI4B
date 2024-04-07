# Requirements for RPI4B
The Raspberry Pi 4 Model B used in the study runs in:
- OS: Bookworm
- Debian Version: 12
- Kernel Version: 6.6
```
sudo apt-get update
sudo apt-get install build-essential
```

## Prerequisite
- Raspberry Pi 4 Model B
- Pi Camera Module 2 / Any Webcamera USB
Check if the camera is detected
````
libcamera-hello
libcamera-hello --list-cameras
````

## Create environment
Needed for YOLOv8 export and execution
````
conda create -n weed_detector python=3.11
conda activate weed_detector
pip install ultralytics==8.0.221
pip install tensorflow==2.13.1
pip install onnx==1.15.0 onnxruntime==1.16.3 onnxsim==0.4.33
pip install -U --force-reinstall flatbuffers==23.5.26
````

Need for PiCamera2 support from libcamera module
```
sudo apt install -y python3-libcamera python3-kms++
sudo apt install -y python3-pyqt5 python3-prctl libatlas-base-dev ffmpeg python3-pip
```

```
pip install picamera2
sudo cp -r /usr/lib/python3/dist-packages/libcamera ~/miniconda3/envs/yolov8_picam/lib/python3.11/site-packages/
sudo cp -r /usr/lib/python3/dist-packages/pykms ~/miniconda3/envs/yolov8_picam/lib/python3.11/site-packages/

cd ~/miniconda3/envs/yolov8_picam/lib
mv -vf libstdc++.so.6 libstdc++.so.6.old
ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6 ./libstdc++.so.6
```

## Run 

Set utf8 format for python if you are getting strange error with latin1 encoding
```
export PYTHONUTF8=1 
```


### Run exported models
````
-m == --model       # Path to model file
-i == --imgsz       # Image size from the camera to be feeded to the model
-d == --debug       # Opens a debugging window showing the detections
-pf == --print_fps  # Prints the frames per second of the feed in the terminal
-sc == --show_conf  # Shows the confidence value of the detection beside the label
````

Sample full command run
```
python main_rpi -m ./models/yolov8n_saved_model/custom_224_yolov8n_fiq.tflite -i 224 -d -pf -sc
```