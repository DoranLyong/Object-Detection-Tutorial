#-*- coding: utf-8 -*-

"""
* Run this code on VScode (it can be run with Jupyter notebook conntection)
* Run each code cell with pressing 'shift + Enter'   
* Like here -> https://blog.naver.com/cheeryun/221685740941
"""


"""
pytorch/vision/torchvision, github (ref) https://github.com/pytorch/vision/tree/master/torchvision/models/detection


Faster R-CNN의 구성 요소에 대해 알아보자. 

* backbone network 
* Region-Proposal network (RPN)
* ROIPooling layer 
* ROI Heads 
"""



#%% 임포트 패키지  
import os.path as osp
import os 
import collections   # 데이터 처리를 위한 유용한 객체들 포함; (ref) https://wikidocs.net/84392


import torch 
import torchvision
import torchvision.transforms as transforms  # Transformations we can perform on our dataset
import torchvision.models as models # computer vision tasks를 위한 모델 포함; (ref) https://pytorch.org/vision/stable/models.html
from torchvision.models.detection import faster_rcnn   # finetuning 이 가능한 모델 불러오기 ; (ref) https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html


cwd =  osp.dirname(osp.realpath(__file__))  # 현재 파일의 디렉토리 경로 가져오기; 
                                                # (ref) https://stackoverflow.com/questions/5137497/find-current-directory-and-files-directory
data_path = osp.join(cwd, 'data')


# ================================================================= #
#                               Setup                               #
# ================================================================= #
# %% Setup
"""
랜덤 발생 기준을 시드로 고정함. 
그러면 shuffle=True 이어도, 언제나 동일한 방식으로 섞여서 동일한 데이터셋을 얻을 수 있음. 
"""
SEED = 42 # set seed 
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED) # for multi-gpu



# ================================================================= #
#                           1. The backbone                         #
# ================================================================= #
# %% 01. Backbone 네트워크 구성 

""" Let's use a simple model called the `AlexNet` instead of VGG or Resnet or FPN.
    torchvision lib already has built this model, it even has the pretrained-weights too.
"""
alexnet = models.alexnet(pretrained=False)

print(alexnet)  # 모델 확인 


# %% backbone 가져오기 
""" As we see below, the model is broken down into `features`, `avgpool` and `classifier`.
    We shall notice that the `features` is full of Conv-layers and we are interested in only this module.
    So let's go ahead and just keep this `features` module and store it inside `backbone`.

    Backbone 네트워크 := feature map 을 추출하는 부분. 
        * 따라서, alexnet의 head 부분은 버리고 
        * body 부분만을 가져온다 
"""

backbone = alexnet.features   # backbone 추출 

print(backbone)


# %% backbone에 대한 입출력 확인 
""" We also need to note the output-channels that the `features` will return.  
    We can get the final channels from the 10th line from the Alexnet features-'(10)':

        * Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))` which is `256`(<- 필터 개수).
"""

image = torch.randn(1, 3, 800, 800)  # 예시 입력 ; input of size (800,800) and pass it through the `feature` module.
feature_map = backbone(image)

print(f"Output shape from backbone is \n\n {feature_map.shape}")



""" We notice that the feature-map size has (24, 24) cells, which essentially means we can place 
    as many anchors with as many aspect-ratios as possible across each of these cells. 

        * 셀의 개수 := 24x24 
        * 각 셀에 가능한 한 많은 aspect-ratio를 가진 앵커를 배치 할 수 있음 
"""


# ================================================================= #
#                    2. Region Proposal Network                     #
# ================================================================= #
# %% 02. RPN 구축 
from torchvision.models.detection import rpn # 다른 backbone을 추가하도록 모델 수정하기;  (ref) https://tutorials.pytorch.kr/intermediate/torchvision_tutorial.html

rpn_layer = rpn.RPNHead(in_channels = 256, num_anchors = 9)
print(f"The RPN layer: \n\n {rpn_layer}")



""" The RPNHead is the convolutional model we talked about, in the previous block.
    Torchvision already has built the `RPN` for us. You see, it is a very simple 
    convolutional block parameterized with `in_channels` and `num_anchors`.

    Notice that `in_channels` must be the same as final channels from the `backbone` i.e 256.
    What about the `num_anchors`?
    Well, it depends on how many anchors we want to place over the feature map. 

    For example, if we want to place anchors of sizes-
    [ (128, 64), (128, 128), (128, 256), 
     (256, 128), (256, 256), (256,512), 
    (512, 256), (512, 512), (512, 1024)]
    
    we will substitute 9 as the parameter for `num_anchors`. (총 9종의 anchor box design)
"""


""" After printing the `RPNHead()`, we notice a `cls_logits` which is a Conv-Layer with 9 output channels.

        * each of these output channels correspond to each anchor box (:= 9채널 출력 = 앵커 박스 개수 9개).

    What about the `36` channels in `bbox_pred`? 
    We can split these 36 channels into 9 groups, such that first four channels depict box locations of anchor (128, 64).
    The next set of 4 channels correspond to box location for anchor with size-(128, 128) and so on. 

        * 1st 4-channel : (128, 64) anchor box의 위치를 나타냄;  (x1, y1, x2, y2)? 
        * 2nd 4-channel : (128, 128) anchor box의 위치를 나타냄;
        * 3rd 4-channel : (128, 256) anchor box의 위치를 나타냄;
        * ... so on. 
        * 즉, 36-channel := 박스 위치 4-channel  x   anchor box 9개 
"""



# %% RPN 출력 확인 
""" Forward the output of backbone to the `RPNHead()` and look at the output size.

    [backbone] -> feature_map -> [RPNHead] -> ? 
"""
print(feature_map.shape)  # [1, 256, 24, 24] shape 
print(feature_map.unsqueeze(0).shape)   # [1, 1, 256, 24, 24] shape 
                                        # 왜 축을 하나 더 추가했지? 


object_score, bbox_locs = rpn_layer(feature_map.unsqueeze(0))


print(f"Object score: {torch.stack(object_score).squeeze(0).shape}") # [1, 9, 24, 24] ; 24x24 feature_map 에서 9 종의 앵커 박스에 위치는 객체 분류 
print(f"Bbox locs: {torch.stack(bbox_locs).squeeze(0).shape}")  # [1, 36, 24, 24] ; 24x24 feature_map 에서 9 종의 앵커 박스 위치 := 9 x 4 
                                                                # torch.stack() ; (ref) https://wikidocs.net/52846


# %% 앵커 생성 
""" At this point, we have just created the RPN but we haven't set (or associated) the anchors with the channels yet.
    So we need to create these anchors.

    Fortunately, `torchvision` also has a function which generates the anchor boxes with respective aspect ratios.
"""
anchor_generator = rpn.AnchorGenerator(sizes=((128, 256, 512),), aspect_ratios=(0.5, 1, 2)) # 앵커 생성 객체 초기화 


"""The above line generates the following anchors.
    [ (128, 64), (128, 128), (128, 256), 
    (256, 128), (256, 256), (256,512), 
    (512, 256), (512, 512), (512, 1024)]

    높이 := 128 일 때, aspect_ratio 를 0.5, 1, 2 비율로 맞추면 => (128, 64,), (128, 128), (128, 256)
    so on... 
"""


# %% RPN + anchor generator 결합 
""" Now that we have created the `rpn_layer` and the anchors i.e `anchor_generator`, 
    lets fuse these two into a single block.

    Torchvision provides us a class `RegionProposalLayer()` whose inputs will be the `rpn-layer`, the `anchors` and a few parameters.

    There are a few technical details about the parameters such as the `nms_threshold`, `foreground_iou_threshold`, 
    `background_iou_threshold`, etc which will not be covered here.
    These parameters' source can be found in the Faster-RCNN research paper itself.
"""

rpn_pre_nms = dict(training=2000, testing=1000)
rpn_post_nms = dict(training=2000, testing=1000)

region_proposer = rpn.RegionProposalNetwork(    anchor_generator,   # 생성된 앵커 
                                                rpn_layer,          # RPN 계층  
                                                0.5, 0.5, 512, 0.5, 
                                                rpn_pre_nms, 
                                                rpn_post_nms, 0.7 
                                            )
region_proposer.training = False

print(f"The Region-Proposal-Network is: \n\n {region_proposer}")


""" As you see, the whole `Region Proposal Network` consists of the `anchor_generator` i.e the anchors, 
    and the `RPNHead` which is the rpn-layer.
"""


# %% Let's see how many objects (or locations) the RPN can detect out of the `(9*24*24) = 5184` possible locations
from torchvision.models.detection import image_list

