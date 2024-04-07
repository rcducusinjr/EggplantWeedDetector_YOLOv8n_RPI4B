'''
Usage:
python test.py \
    --model yolov8n.pt \
    --data datasets/Eggplant-and-Weeds-Detection/test/images \
    --save
'''

from ultralytics import YOLO
import argparse
 
if __name__ == '__main__':
    # Arguments Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default="yolov8n.pt")
    parser.add_argument('-d', '--data', default="datasets/Eggplant-and-Weeds-Detection-10/test/images")
    parser.add_argument('-s', '--save', action=argparse.BooleanOptionalAction)

    args = parser.parse_args()


    # Load a model
    model = YOLO(args.model)

    # Run batched inference on a list of images
    results = model.predict(
        source=args.data,
        save=args.save
    ) # return a list of Results objects

    # Process results list
    for result in results:
        boxes = result.boxes  # Boxes object for bbox outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs