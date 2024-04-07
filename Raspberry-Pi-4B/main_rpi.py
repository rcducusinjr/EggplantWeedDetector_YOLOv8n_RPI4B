import cv2
import argparse
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

import picamera2 as PiCam
import time

# Draw annotations/boxes to the image
def draw_boxes(frame, class_list, results, print_conf):
    color = [(0, 128, 255), (255, 128, 0)]         # blue, orange
    annotator = None
    for curr in results:
        annotator = Annotator(frame)

        boxes = curr.boxes
        for box in boxes:
            b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
            c = str(box.cls)[8]         # get class number

            if (print_conf):
                conf = str(box.conf)[8:12]  # get confidence value
                annotator.box_label(box=b, label=f"{class_list[int(c)]} {conf}", color=color[int(c)], txt_color=(0, 0, 0))
            else:
                annotator.box_label(box=b, label=class_list[int(c)], color=color[int(c)], txt_color=(0, 0, 0))


    if annotator is not None:
        frame = annotator.result()

    return frame

def draw_annotation(img, label_names, results):
    annotator = None
    for r in results:
        annotator = Annotator(img)

        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
            c = box.cls
            annotator.box_label(b, label_names[int(c)])

    if annotator is not None:
        annotated_img = annotator.result()
    else:
        annotated_img = img.copy()

    return annotated_img


# Main
if __name__ == '__main__':
    # Arguments Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default="./models/yolov8n_2ndEdition_v9_b4_100e_224/best_full_integer_quant.tflite")
    parser.add_argument('-i', '--imgsz', default=224)
    parser.add_argument('-d', '--debug', action=argparse.BooleanOptionalAction)
    parser.add_argument('-pf', '--print_fps', action=argparse.BooleanOptionalAction)
    parser.add_argument('-sc', '--show_conf', action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    imgsz = args.imgsz


    # Pi Camera Configuration
    picam = PiCam.Picamera2()
    config = picam.create_still_configuration(
        main={"size": (640, 480)}, # scale down the image, but maintain the full field of view
        raw={'size': (imgsz, imgsz)},		# Setting this to the same as imgsz will make a slight impact to the FPS
        buffer_count=2,			# Decreasing this <2 will make the FPS go lower
        queue=True,				# Setting this to False will make the FPS go lower
        controls={'FrameRate': 10},
    )
    picam.align_configuration(config)
    picam.configure(config)

    picam.start()

    # To stabilize exposure, gains, etc. of the Pi Camera Module
    time.sleep(5)


    # Configuring the YOLO model
    model = YOLO(args.model)
    class_list = ['Eggplant', 'Weeds']
    fps = 0
    frame_count = 0
    print_conf = args.print_conf

    while True:
        start_time = time.time()
        frame = picam.capture_array()

        frame_count += 1

        results = model.predict(frame, imgsz=224)           # Inference of the YOLO model

        # Show the UI
        if args.debug:
            if results is not None:
                frame = draw_boxes(frame, class_list, results, print_conf)
            cv2.imshow("piCam", frame)
            if cv2.waitKey(1) == ord('q'):
                break


        end_time = round(time.time() - start_time, 3)
        fps = (0.9 * fps) + (0.1 * (1 / end_time))

        # Prints the FPS
        if args.print_fps:
            print("FPS: " + str(fps))

# Clean up
cv2.destroyAllWindows()
