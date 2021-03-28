#-*- coding: utf-8 -*-

"""
* Run this code on VScode (it can be run with Jupyter notebook conntection)
* Run each code cell with pressing 'shift + Enter'   
* Like here -> https://blog.naver.com/cheeryun/221685740941
"""


"""
pytorch/vision/torchvision, github (ref) https://github.com/pytorch/vision/tree/master/torchvision/models/detection


COCO dataset을 활용한 Fine-tuning 연습 

* backbone network 
* Region-Proposal network (RPN)
* ROIPooling layer 
* ROI Heads 
"""

#%% 임포트 패키지 
import os.path as osp

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
gpu_no = 0  # gpu_number 
device = torch.device( f'cuda:{gpu_no}' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")



# ================================================================= #
#               2. Load pre-trained Faster R-CNN                    #
# ================================================================= #
# %% 02. ResNet-50 FPN backbone 을 활용한 사전 학습 모델 불러오기 

model = fasterrcnn_resnet50_fpn(pretrained=True)
model = model.to(device)



# ================================================================= #
#                     3. Prepare labeling data                      #
# ================================================================= #
# %% 03. 데이터 레이블링 

image_list = ['FudanPed00066.png', 'PennPed00011.png']

bbox_list  = [  [[248.0, 50.0, 329.0, 351.0]],
                [[92.0, 62.0, 236.0, 344.0], [242.0, 52.0, 301.0, 355.0]],
             ]

label_list = [  [1],
                [1, 1],
             ]            


img1 = Image.open(osp.join(data_path, image_list[0]))
img1_tensor = transforms.ToTensor()(img1)   # Convert a PIL Image or numpy.ndarray to tensor; 
                                            # (ref) https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.ToTensor
bboxes1 = torch.tensor(bbox_list[0])
label1  = torch.tensor(label_list[0])


img2 = Image.open(osp.join(data_path, image_list[1]))
img2_tensor = transforms.ToTensor()(img2)  
                                           
bboxes2 = torch.tensor(bbox_list[1])
label2  = torch.tensor(label_list[1])


print(f'img1_tensor size: {img1_tensor.size()}')
print(f'img2_tensor size: {img2_tensor.size()}')




# ================================================================= #
#                           4. Model Inference                      #
# ================================================================= #
# %% 04. 모델 출력 확인 
""" We need a list (not tensor) of images for the model inference.
        * PyTorch 모델의 입력은 자료형이 list 이어야 하는 군 
        * 그래서 '02_understanding_Faster_R-CNN.py' 에서 image_list.ImageList() 를 활용했지 

    Images size may be different. This means we need not resize to a constant size. 
    Faster RCNN PyTorch Implementation has its own image pre-process block.        
"""
input_img1 = img1_tensor.clone()
input_img2 = img2_tensor.clone()


# input list 
inputs = [  input_img1.to(device),
            input_img2.to(device)
        ]


model.eval() 
output = model(inputs)

print(output)



# ================================================================= #
#                           5. Model Training                       #
# ================================================================= #
# %% 05. 모델 학습 
""" Targets should be a list, and each target should have the following format:

    *   {   'boxes': bounding boxes tensor,
            'labels': label tensor, 
        }

    Object detection in Faster R-CNN is done in two stages.
    First, it classifies all regions of the image in just two classes; background or object.
    In the second stage, it predicts classes of the object and improves its bounding box predictions.
"""

input_img1 = img1_tensor.clone()

target1 = { 'boxes': bboxes1.clone().to(device),
            'labels': label1.clone().to(device),
          }


input_img2 = img2_tensor.clone()

target2 = { 'boxes': bboxes2.clone().to(device),
            'labels': label2.clone().to(device),
          }          


# input list 
inputs = [  input_img1.to(device),
            input_img2.to(device)
         ]

# target list 
targets = [ target1,
            target2, 
          ]


""" change to train mode 
"""
model.train()
output_train = model(inputs, targets)

print(output_train)


""" 여기서 핵심은 
    model.train() 의 입력 파라미터들은 모두 list 자료형일 것 
"""


# ================================================================= #
#                    6. Model Building Blocks                       #
# ================================================================= #
# %% 06. Faster R-CNN 구조의 각 블록 설명 

model.eval()
print(model)


""" We can see the model has the following building blocks of Faster R-CNN:

        * transform : This block pre-processes the input image.
        * backbone : This is equivalent to 'conv layers' in the above image.
        * rpn : This is equivalent to Region Proposal Network(RPN) in the above image.
        * roi_heads : This is equivalent to 'RoI Pooling'.
        * box_predictor : This is equivalent to 'classifier' in the above image.
"""


# %% transform 블록 
""" This block pre-processes the input like normalizing, resizing, etc.
"""

input_img1 = img1_tensor.clone()
input_img2 = img2_tensor.clone()


inputs = [  input_img1.to(device),
            input_img2.to(device)
         ]

trans_image_list, trans_target_list = model.transform(inputs)   # Faster R-CNN의 transform 블록의 출력    
 
print(f'Tensor shape: {trans_image_list.tensors.shape} \n')



""" Let's have a look at transforms parameters.
"""
print('transform ( GeneralizedRCNNTransform) parameters:')
print(f'min_size: {model.transform.min_size}')
print(f'max_size: {model.transform.max_size}')
print(f'image_mean: {model.transform.image_mean}') # 각 채널별 이미지 패널의 mean  
print(f'image_std: {model.transform.image_std}') # 각 채널별 이미지 패널의 std 


""" Transform params in Faster R-CNN Fine-tune model:

        * If we have smaller images for training, 
          we might be interested in changing the transform parameters

        * 변환 파라미터 변경 
"""
ft_min_size = 300
ft_max_size = 500

ft_mean = [0.485, 0.456, 0.406]
ft_std = [0.229, 0.224, 0.225]



# %% backbone (conv layers) 블록 
print(f"{trans_image_list.tensors.shape}")  # [2, 3, 800, 1088]

backbone_out = model.backbone(trans_image_list.tensors)  # backbone에 이미지 리스트 입력 


for key, value in backbone_out.items():  # backbone 블록의 각 계층별 출력 
    print(f'{key}: {value.size()}')  



# %%
""" The output of the backbone is OrderedDict[Tensor] of five tuples
"""
print(f'Number of output channel of the backbone: {model.backbone.out_channels}')


# %% Backbone of Faster R-CNN Fine-tune model
import torchvision.models as models

""" Faster R-CNN 의 backbone 바꿔서 끼우기 
"""

alexnet = models.alexnet(pretrained=True)  # 사전 학습 모델 다운 
print(alexnet)

ft_backbone = alexnet.features

# number of out-channel in alexnet features is 256
ft_backbone.out_channels = 256
print(ft_backbone)




# ================================================================= #
#                   7. rpn (Region Proposal Network)                #
# ================================================================= #
# %% 07. rpn (RPN)
""" It takes features from the backbone and predicts the objectness 
    (the region and whether it is object or background) and coordinates of the region.

        * 감지된 부분이 object 인가 background 인가? 
        * localization은 어떻게 되는 가? 
"""

print(model.rpn)


print(f'Anchor sizes: {model.rpn.anchor_generator.sizes}')
print(f'Aspect ratios: {model.rpn.anchor_generator.aspect_ratios}')



# %% Anchor of Faster R-CNN Fine-tune model
from torchvision.models.detection.rpn import AnchorGenerator

""" Since the AlexNet has a single label output, 
    the anchor size should be a single tuple.
"""

ft_anchor_generator = AnchorGenerator(  sizes=((32, 64, 128, 256),), 
                                        aspect_ratios=((0.5, 1.0, 2.0),)
                                     )



# ================================================================= #
#                      8. roi_heads (RoI Pooling)                   #
# ================================================================= #
# %% 08. RoI Pooling 

print(model.roi_heads)


print('Box RoI Pool Parameters:')
print(f'featmap_names: {model.roi_heads.box_roi_pool.featmap_names}')
print(f'output_size: {model.roi_heads.box_roi_pool.output_size}')
print(f'sampling_ratio: {model.roi_heads.box_roi_pool.sampling_ratio}')


# %%
""" RoI Pooler of Faster RCNN Fine-tune model
"""
print(backbone_out.keys())
print(type(ft_backbone(torch.rand((2, 3, 300, 300)))))


# %%
from torchvision.ops import MultiScaleRoIAlign

ft_roi_pooler = MultiScaleRoIAlign(featmap_names=['0'], output_size=4, sampling_ratio=1)




# ================================================================= #
#                Faster R-CNN with AlexNet Backbone                 #
# ================================================================= #
# %% AlexNet 을 backbone 으로 만들어서 사용하기 
from torchvision.models.detection import FasterRCNN

"""Let number of classes 4 (including background)
"""

ft_model = FasterRCNN(backbone=ft_backbone,
                      num_classes=2, 
                      min_size=ft_min_size, 
                      max_size=ft_max_size, 
                      image_mean=ft_mean, 
                      image_std=ft_std, 
                      rpn_anchor_generator=ft_anchor_generator, 
                      box_roi_pool=ft_roi_pooler
                      )

ft_model = ft_model.to(device)





# %% Model Inference
input_img1 = img1_tensor.clone()
input_img2 = img2_tensor.clone()


# input list 
inputs = [  input_img1.to(device),
            input_img2.to(device)
        ]

ft_model.eval()
output = ft_model(inputs)

print(output)


# %% Model Training
input_img1 = img1_tensor.clone()

target1 = { 'boxes': bboxes1.clone().to(device),
            'labels': label1.clone().to(device),
          }


input_img2 = img2_tensor.clone()

target2 = { 'boxes': bboxes2.clone().to(device),
            'labels': label2.clone().to(device),
          }          


# input list 
inputs = [  input_img1.to(device),
            input_img2.to(device)
         ]

# target list 
targets = [ target1,
            target2, 
          ]


# change to train mode
ft_model.train()
ft_model(inputs, targets)


