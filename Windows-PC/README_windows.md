# Windows (For Training, Validating, Testing, and Exporting)

## Specifications
- CPU: Ryzen 7 5700x
- GPU: NVidia RTX 4070
- SSD: 1TB

## Prerequisite
- CUDA Version: 12.3
- CUDNN Version: 9.0.0
- Ultralytics: 8.1.9


## Create environment
Needed for YOLOv8 transfer learning and exporting
````
conda create -n weed_detector python=3.11
conda activate weed_detector
pip install ultralytics==8.1.9
pip install tensorflow==2.13.1
pip install onnx==1.15.0 onnxruntime==1.16.3 onnxsim==0.4.33
pip install -U --force-reinstall flatbuffers==23.5.26
````


## Export yolov8n to tflite format
```
python exporter.py
```

## Run 

### Run exported models
```
python main.py --model ./models/yolov8n_224/best_full_integer_quant.tflite --imgsz 224 --show_conf
```