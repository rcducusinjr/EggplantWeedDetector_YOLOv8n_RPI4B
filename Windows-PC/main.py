'''
Usage:
python main.py --model best_full_integer_quant.tflite --imgsz 224 --show_conf
'''

import time
import argparse

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import cv2

# Draw annotations/boxes to the image
def draw_boxes(frame, results, threshold, show_conf):
    class_list = [('Eggplant', 'Weeds')]
    color = [(0, 128, 255), (255, 128, 0)]         # blue, orange
    annotator = None
    for curr in results:
        annotator = Annotator(frame)

        boxes = curr.boxes
        for box in boxes:
            b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
            c = str(box.cls)[8]         # get class number


            if (show_conf):
                conf = box.conf         # get confidence value
                if conf >= threshold:
                    conf = str(box.conf)[8:12]  
                    annotator.box_label(box=b, label=f"{class_list[int(c)]} {conf}", color=color[int(c)], txt_color=(0, 0, 0))
            else:
                annotator.box_label(box=b, label=class_list[int(c)], color=color[int(c)], txt_color=(0, 0, 0))


    if annotator is not None:
        frame = annotator.result()

    return frame

if __name__ == '__main__':
    # Arguments Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default="runs/detect/yolov8n_2ndEdition_v9_b4_100e_224/weights/best_saved_model/best_full_integer_quant.tflite")
    parser.add_argument('-i', '--imgsz', default=224)
    parser.add_argument('-sc', '--show_conf', action=argparse.BooleanOptionalAction)

    args = parser.parse_args()


    # Load a model
    model = YOLO(args.model)  # load a custom model
    threshold = 0.5

    # Configuration of Webcamera
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    fps = 0
    frame_count = 0


    while True:
        start_time = time.time()
        
        ret, frame = cap.read()
        H, W, _ = frame.shape
        frame_count += 1

        result = model(frame, stream=True)

        results = model.predict(frame, imgsz=args.imgsz)

        if results is not None:
            frame = draw_boxes(frame, results, threshold, args.show_conf)
        
        end_time = round(time.time() - start_time, 3)
        fps = (0.9 * fps) + (0.1 * (1 / end_time))

        print("FPS: " + str(fps))
        
        cv2.imshow(f"Webcamera", frame)
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()