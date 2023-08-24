import tensorflow as tf
from utils import load_class_names, output_boxes, draw_outputs, resize_image
import cv2
import numpy as np
from yolov3 import YOLOv3Net
from google.colab.patches import cv2_imshow   # Google Colab edit
# Google Colab returns an error while using cv2.imshow(): Returns Qt application error
# Instead you have to use cv2_imshow

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

'''
# Folder Paths
class_name = './data/coco.names'
cfgfile = 'cfg/yolov3.cfg'
weightfile = 'weights/yolov3_weights.tf'
img_path = "data/images/test.jpg"
'''
# Google Colab edit
class_name = "/content/drive/MyDrive/MASTER/Master_Thesis/YOLO/PROJECTS/YOLOv3/data/coco.names"
cfgfile = "/content/drive/MyDrive/MASTER/Master_Thesis/YOLO/PROJECTS/YOLOv3/cfg/yolov3.cfg"
weightfile = "/content/drive/MyDrive/MASTER/Master_Thesis/YOLO/PROJECTS/YOLOv3/weights/yolov3_weights.tf"
img_path = "/content/drive/MyDrive/MASTER/Master_Thesis/YOLO/PROJECTS/YOLOv3/data/images/test.jpg"

model_size = (416, 416, 3)
num_classes = 80
max_output_size = 40
max_output_size_per_class= 20
iou_threshold = 0.5
confidence_threshold = 0.5

def main():
    model = YOLOv3Net(cfgfile,model_size,num_classes)
    model.load_weights(weightfile)
    class_names = load_class_names(class_name)

    image = cv2.imread(img_path)
    image = np.array(image)
    image = tf.expand_dims(image, 0)

    resized_frame = resize_image(image, (model_size[0],model_size[1]))
    pred = model.predict(resized_frame)

    # Get detected boxes, scores and class names
    boxes, scores, classes, nums = output_boxes( \
        pred, model_size,
        max_output_size=max_output_size,
        max_output_size_per_class=max_output_size_per_class,
        iou_threshold=iou_threshold,
        confidence_threshold=confidence_threshold)
    
    print("Detection Boxes Coordinates")
    print(boxes)
    print("Score Array")
    print(scores)
    print("Detected Classes Array")
    print(classes)

    image = np.squeeze(image)
    img = draw_outputs(image, boxes, scores, classes, nums, class_names)
    win_name = 'Image detection'
    # cv2.imshow(win_name, img)
    cv2_imshow(img)   # Google Colab edit
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Save the image with detection boxes
    cv2.imwrite('test.jpg', img)

if __name__ == '__main__':
    main()