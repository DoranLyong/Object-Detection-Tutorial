#-*- coding: utf-8 -*-

"""
* Run this code on VScode (it can be run with Jupyter notebook conntection)
* Run each code cell with pressing 'shift + Enter'   
* Like here -> https://blog.naver.com/cheeryun/221685740941
"""


"""
pytorch/vision/torchvision, github (ref) https://github.com/pytorch/vision/tree/master/torchvision/models/detection


Fine-tuning 셋업 연습 

1. Set device  
2. Load pre-trained Faster R-CNN  
3. Prepare labeling data  
4. Model Inference     
5. Model Training
6. Model Building Blocks
7. rpn (Region Proposal Network)
8. roi_heads (RoI Pooling)
"""

#%% 임포트 패키지 
import os.path as osp
import random 
from operator import itemgetter # 튜플 정렬하기; (ref) https://popawaw.tistory.com/17

import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
from PIL import Image  # 파이썬용 이미지를 처리하는 패키지 ; (ref) http://pythonstudy.xyz/python/article/406-%ED%8C%8C%EC%9D%B4%EC%8D%AC-%EC%9D%B4%EB%AF%B8%EC%A7%80-%EC%B2%98%EB%A6%AC-Pillow
import torch
import torchvision
import torchvision.transforms as transforms  # Transformations we can perform on our dataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn    # finetuning 이 가능한 모델 불러오기 ; 
                                                                    # (ref) https://pytorch.org/vision/stable/models.html


cwd =  osp.dirname(osp.realpath(__file__))  # 현재 파일의 디렉토리 경로 가져오기; 
                                                # (ref) https://stackoverflow.com/questions/5137497/find-current-directory-and-files-directory
data_path = osp.join(cwd, 'data')



# ================================================================= #
#                         1. Set device                             #
# ================================================================= #
# %% 01. 프로세스 장비 설정 