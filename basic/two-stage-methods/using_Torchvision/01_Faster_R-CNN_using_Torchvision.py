#-*- coding: utf-8 -*-

"""
* Run this code on VScode (it can be run with Jupyter notebook conntection)
* Run each code cell with pressing 'shift + Enter'   
* Like here -> https://blog.naver.com/cheeryun/221685740941
"""


#%% 임포트 패키지  
import os.path as osp
import os 

import numpy as np 
import cv2 
from PIL import Image  # 파이썬용 이미지를 처리하는 패키지 ; (ref) http://pythonstudy.xyz/python/article/406-%ED%8C%8C%EC%9D%B4%EC%8D%AC-%EC%9D%B4%EB%AF%B8%EC%A7%80-%EC%B2%98%EB%A6%AC-Pillow
from tqdm import tqdm 
import matplotlib.pyplot as plt
import torch 
import torchvision
import torchvision.transforms as transforms  # Transformations we can perform on our dataset



cwd =  osp.dirname(osp.realpath(__file__))  # 현재 파일의 디렉토리 경로 가져오기; 
                                                # (ref) https://stackoverflow.com/questions/5137497/find-current-directory-and-files-directory
data_path = osp.join(cwd, 'data')


# ================================================================= #
#                         1. Set class names                        #
# ================================================================= #
# %% 01. 클래스 레이블 이름 등록 
""" Class labels from official PyTorch documentation for the pretrained model
    Note that there are some N/A's for complete list check https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
"""


COCO_INSTANCE_CATEGORY_NAMES = [    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                                    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
                                    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                                    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
                                    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                                    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                                    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                                    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                                    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
                                    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                                    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
                                    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
                                ]


# ================================================================= #
#                   2. Get the pretrained model                     #
# ================================================================= #
# %% 02. 사전 학습 모델 불러오기 (using torchvision)

""" get the pretrained model from torchvision.models
    Note: pretrained=True will get the pretrained weights for the model.
"""

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()  # inference mode 



# ================================================================= #
#                         3. Detect Objects                         #
# ================================================================= #
# %% 03. object detection 함수 정의 
""" the function below to load the image and pass it through the model. 

    The output prediction consists of:
        * the predicted bounding boxes
        * class to which it belongs
        * confidence of prediction
"""

def get_prediction(img_path, threshold):
    """
    parameters:
      - img_path - path of the input image
      - threshold - threshold value for prediction score

    method:
      - Image is obtained from the image path
      - the image is converted to image tensor using PyTorch's Transforms
      - image is passed through the model to get the predictions
      - class, box coordinates are obtained, but only prediction score > threshold
        are chosen.
    """
    img = Image.open(img_path)
    transform = transforms.Compose([transforms.ToTensor()])
    img = transform(img)
    pred = model([img])
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]
    pred_score = list(pred[0]['scores'].detach().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x>threshold][-1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    return pred_boxes, pred_class


# %%
def detect_and_display(img_path, threshold=0.5, rect_th=3, text_size=3, text_th=3):
    """
    parameters:
      - img_path - path of the input image
      - threshold - threshold value for prediction score
      - rect_th - thickness of bounding box
      - text_size - size of the class label text
      - text_th - thickness of the text

    method:
      - prediction is obtained from get_prediction method
      - for each prediction, bounding box is drawn and text is written 
        with opencv
      - the final image is displayed
    """
    boxes, pred_cls = get_prediction(img_path, threshold)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for i in range(len(boxes)):
        cv2.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th)
        cv2.putText(img,pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)

    plt.figure(figsize=(20,30))
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()


# ================================================================= #
#                            4. Examples                            #
# ================================================================= #
# %% 04. 예시 출력 

input_list = os.listdir(data_path)

for idx, item in enumerate(input_list):
    img_path = osp.join(data_path, item)    
    detect_and_display(img_path, rect_th=2, text_th=1, text_size=1)






# ================================================================= #
#        Comparing the inference time of model in CPU & GPU         #
# ================================================================= #
# %% Inference time 비교 
import time 

def check_inference_time(image_path, gpu=False):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    
    img = Image.open(image_path)
    transform = transforms.Compose([transforms.ToTensor()])
    img = transform(img)

    if gpu:
        model.cuda()
        img = img.cuda()
    else:
        model.cpu()
        img = img.cpu()

    start_time = time.time()
    pred = model([img])
    end_time = time.time()
    return end_time-start_time


# %%
img_path = osp.join(data_path, 'traffic_scene.jpg')  

testCount=10
cpu_time = sum([check_inference_time(img_path, gpu=False) for _ in range(testCount)])/testCount
gpu_time = sum([check_inference_time(img_path, gpu=True) for _ in range(testCount)])/testCount

print(f'Average Time taken by the model with GPU = {gpu_time}s')
print(f'Average Timen take by the model with CPU = {cpu_time}s')


# %%
