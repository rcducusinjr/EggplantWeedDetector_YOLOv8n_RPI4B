'''
Export the trained model to tflite format.

Usage:
python exporter.py 
'''

from ultralytics import YOLO
import argparse

if __name__ == '__main__':
    # Arguments Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default="yolov8n.pt",
                        help="specify model or yaml. e.g. yolov8n_custom_224x224.pt")
    parser.add_argument('-i', '--imgsz', type=int, default=224)

    args = parser.parse_args()

    # Load a model
    model = YOLO(args.model)  # load a custom trained model

    # Export the model
    model.export(format='tflite', imgsz=args.imgsz, half=True, int8=True)