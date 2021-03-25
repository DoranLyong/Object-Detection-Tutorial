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