#video.py
import tensorflow as tf
from utils import load_class_names, output_boxes, draw_outputs, resize_image
from yolov3 import YOLOv3Net
import cv2
import time
from google.colab.patches import cv2_imshow   # Google Colab edit
# Google Colab returns an error while using cv2.imshow(): Returns Qt application error
# Instead you have to use cv2_imshow

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

SAVE_VIDEO = True

'''
# Folder Paths
class_name = './data/coco.names'
cfgfile = 'cfg/yolov3.cfg'
weightfile = 'weights/yolov3_weights.tf'
video_path = "data/videos/test.mp4"
'''
# Google Colab edit
class_name = "/content/drive/MyDrive/MASTER/Master_Thesis/YOLO/PROJECTS/YOLOv3/data/coco.names"
cfgfile = "/content/drive/MyDrive/MASTER/Master_Thesis/YOLO/PROJECTS/YOLOv3/cfg/yolov3.cfg"
weightfile = "/content/drive/MyDrive/MASTER/Master_Thesis/YOLO/PROJECTS/YOLOv3/weights/yolov3_weights.tf"
video_path = "/content/drive/MyDrive/MASTER/Master_Thesis/YOLO/PROJECTS/YOLOv3/data/videos/test.mp4"

model_size = (416, 416,3)
num_classes = 80
max_output_size = 100
max_output_size_per_class= 20
iou_threshold = 0.5
confidence_threshold = 0.5

def main():
    model = YOLOv3Net(cfgfile,model_size,num_classes)
    model.load_weights(weightfile)
    class_names = load_class_names(class_name)

    # There is restriction from colab you can not use window function of cv2
    # because its use for local system
    # win_name = 'Yolov3 detection'
    # cv2.namedWindow(win_name)
    '''
    # Specify the vidoe input. 0 means input from cam 0.
    # For video, just change the 0 to video_path
    cap = cv2.VideoCapture(0)
    '''
    cap = cv2.VideoCapture(video_path)
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    
    if SAVE_VIDEO == True:
        output_vid = cv2.VideoWriter('test.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 5, frame_size)

    try:
        while True:
            start = time.time()
            ret, frame = cap.read()

            # Break if there is no captured video
            if not ret:   break 
                
            resized_frame = tf.expand_dims(frame, 0)
            resized_frame = resize_image(resized_frame, (model_size[0],model_size[1]))
            pred = model.predict(resized_frame)
            
            # Get detected boxes, scores and class names
            boxes, scores, classes, nums = output_boxes( \
                pred, model_size,
                max_output_size=max_output_size,
                max_output_size_per_class=max_output_size_per_class,
                iou_threshold=iou_threshold,
                confidence_threshold=confidence_threshold)
            
            img = draw_outputs(frame, boxes, scores, classes, nums, class_names)
            # cv2.imshow(win_name, img)
            # cv2_imshow(img)   # Google Colab edit

            if SAVE_VIDEO == True:
                output_vid.write(img)

            stop = time.time()
            seconds = stop - start
            # print("Time taken : {0} seconds".format(seconds))
            # Calculate frames per second
            fps = 1 / seconds
            print("Estimated frames per second : {0}".format(fps))
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == ord('Q'):
                break
    finally:
        cv2.destroyAllWindows()
        cap.release()
        if SAVE_VIDEO == True:
            output_vid.release()
        print('Detections have been performed successfully.')

if __name__ == '__main__':
    main()