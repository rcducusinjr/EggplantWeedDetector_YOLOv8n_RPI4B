'''
Before proceeding, check if you can run YOLOv8 using a GPU.
Run the 'cuda_checker.py' file. Make sure the output in the terminal is "cuda".

You can download and configure the Eggplant and Weeds Dataset using this link:
https://app.roboflow.com/uplb-ducusin-sp/eggplant-and-weeds-detection

Parameters used in this study to determine the best model for RPI4B:
Images Sizes: 224, 320, 640
Epochs: 50, 75, 100, 300
Batches: 4, 8, 16

Usage: (using a pretrained model)
python train.py \
    --model yolov8n.pt \
    --imgsz 320 \
    --data datasets/Eggplant-and-Weeds-Detection-10/data.yaml \
    --epochs 100 \
    --batch_size 8 \
    --name yolov8n_v7_b4_300e

Usage: (resume training of a model - only the model and resume options are available)
python train.py --model runs/detect/yolov8n_v7_b4_300e/weights/last.pt --resume
'''

from ultralytics import YOLO
import argparse
 
if __name__ == '__main__':
    # Arguments Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default="yolov8n.pt",
                        help="specify model or yaml. e.g. yolov8n_custom_224x224.pt")
    parser.add_argument('-i', '--imgsz', type=int, default=224)
    parser.add_argument('-d', '--data', type=str, default="datasets/Eggplant-and-Weeds-Detection-10/data.yaml")
    parser.add_argument('-e', '--epochs', type=int, default=100)
    parser.add_argument('-b', '--batch_size', type=int, default=8)
    parser.add_argument('-n', '--name', type=str, default="yolov8n_custom")
    parser.add_argument('-r', '--resume', action=argparse.BooleanOptionalAction)

    args = parser.parse_args()


    if not args.resume:
        # Load the model.
        model = YOLO(args.model).to('cuda')

        # Training.
        results = model.train(
                data=args.data,
                imgsz=args.imgsz,
                epochs=args.epochs,
                batch=args.batch_size,
                name=args.name
            )
    else:
        # Loads the partially trained model
        model = YOLO(args.model)  

        # Resume training
        results = model.train(resume=True)
    