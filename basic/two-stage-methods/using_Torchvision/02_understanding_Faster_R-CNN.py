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
"""
앵커 개수 := 9 
feature map area := (24, 24) 를 다루는 RPN 있다면, 이놈은 몇 개의 물체를 다룰 수 있을 까? 
"""

from torchvision.models.detection import image_list     # 이미지 리스트를 단일 텐서 형태로 저장함 
                                                        #(ref) https://github.com/pytorch/vision/blob/master/torchvision/models/detection/image_list.py


image = torch.randn(1, 3, 800, 800)  # 예시 입력 ; input of size (800,800) and pass it through the `feature` module.
inputs = image_list.ImageList(tensors = image,  image_sizes=[(800,800)])



# %%
"""The `region_proposer` takes the input as a list of images and the backbone output.
"""
rpn_bbox, _ = region_proposer(inputs, dict(feats = feature_map))
num_boxes = rpn_bbox[0].shape[0]
print(f"The number of boxes with objects detected by RPN: \n\n {rpn_bbox[0].shape}")    # [915, 4]
                                                                                        # 즉, bbox 개수 := 915 종                                                                                 

""" Since the `inputs` is a list of images, the output i.e `bbox` will also be a list of boxes. 
    Hence, we need to index the list to access the boxes.
"""



# ================================================================= #
#                          3. RoI Pooling                           #
# ================================================================= #
# %% 03. RoI Pooling 레이어의 역할에 대해 알아보자 
"""
RPN 을 통해 예측된 bbox 들(=regions of feature map)은 사이즈들이 모두 다르다. 
즉, 균일한 Fully-connected layer, Convolution layer, 또는 pooling layer 를 적용할 수 없다 (bbox 마다 출력 사이즈가 달라지니까). 

따라서, 모든 arbitrary sized 를 가진 feature map 을 단 하나의 fixed size 로 바꿔야한다. 
그 역할을 하는 것이 RoI Pooling Layer 이다. 

    * 'scaling-down' or 'scaling-up' 
    * Use 'torch.nn.AdaptiveMaxPooling2d()'
"""

fixed_size_pooler = torch.nn.AdaptiveMaxPool2d((7,10))  # target output size of 7x10
                                                        # (ref) https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html
                                                        # (ref) https://discuss.pytorch.org/t/what-is-adaptiveavgpool2d/26897/2
                                                        # 공대생의 차고 (ref) https://underflow101.tistory.com/41

input1 = torch.randn(1, 64, 77, 63) # B, C, H, W
input2 = torch.randn(3, 44, 67)
input3 = torch.randn(1, 33, 10, 10)

out1 = fixed_size_pooler(input1)
out2 = fixed_size_pooler(input2)
out3 = fixed_size_pooler(input3)

print(f"Output from ROI-Pooling with different-sized inputs: \n{out1.shape} \n{out2.shape} \n{out3.shape}")

""" 출력: 
    torch.Size([1, 64, 7, 10]) 
    torch.Size([3, 7, 10]) 
    torch.Size([1, 33, 7, 10])

    Batch 와 Channel 의 크기는 다르지만 
    feature map의 넓이(area)는 모두 7x10 으로 같아졌다.. 
"""

# %% Torchvision 을 활용한 RoI Pooling 
from torchvision.ops import MultiScaleRoIAlign  
                                                
""" `torchvision` has a function which does the task of ROI Pooling for us.
        * It goes by the name `MultiScaleROIAlign()`.

    Note that this function does the same operation as the `nn.AdaptiveMaxPooling()` but it has some extra features.
    Hence, we tend to use this class instead of the `nn.AdaptiveMaxPooling2d()`

    It takes the argument `feature_map-names` as the input, the final `output_size` which we want to end up with
    and a `sampling_ratio` which tells how to sample the points in a feature map.

        * featmap_names  - the names of the feature maps that will be used for the pooling.
        * output_size    – the fixed output size for the pooled region
        * sampling_ratio - sampling ratio for ROIAlign

(ref) https://pytorch.org/vision/stable/ops.html#torchvision.ops.MultiScaleRoIAlign
(ref) https://github.com/pytorch/vision/blob/master/torchvision/ops/poolers.py        
"""

roi_pooler = MultiScaleRoIAlign(featmap_names=['0'], output_size=(7, 10), sampling_ratio=2)  # RoI Pooling 계층 초기화 
                                                                                             # 사용할 feature map 의 이름 := '0'


# %%
""" As the 'ROI Pooling' pools the variable sized feature map regions into fixed sized regions, 
        * we need the backbone features (got from the backbone) and the adjusted-boxes (got from the RPN-layer).
        * Also, it needs the original image size to calculate the scale.
        * (하나의 고정된 사이즈의 pooled map 을 얻기 위해 필요한 것 들)

    We already have the output from backbone i.e the `feature_map` out of the backbone.
    We also have the RPN output i.e `rpn_bbox` but these outputs are completely random. 
    Hence, we will create the rpn boxes again. 

    Can you guess what will be the size of the output from `roi_pooler`?
        * area size := 7 x 10 
"""
fmaps = collections.OrderedDict()   # 순서가 있는 dict 자료형
                                    # (ref) https://www.daleseo.com/python-collections-ordered-dict/

fmaps["0"] = feature_map    # backbone 에서 나온 feature map 을 저장 
                            # roi_pooler를 초기화 할 때 사용한 것 과 동일한 이름으로 key를 설정. 


""" Let's assume that we have the 915 boxes from the RPN 
    (the same number of boxes we got previously i.e `rpn_bbox[0].shape`)
"""

rpn_bbox_rand = torch.rand(num_boxes, 4) * 400; # let's create some random boxes as output from RPN. 
                                                # 임의로 한변의 길이가 최대 400 이내인 bbox 를 생성. 
                                                # 예시) rpn_bbox_rand[0] := tensor([304.2635, 114.6079, 282.7722, 337.3543])
                                                # torch.rand() ; (ref) https://pytorch.org/docs/stable/generated/torch.rand.html

rpn_bbox_rand[:, 2:] += rpn_bbox_rand[:, :2] # the format should be (x_top, y_top, width, height) 
                                             # hence, we add the (x_top, y_top) to (width, height) coordinates
                                             # 즉, rpn_bbox_rand[0] := tensor([304.2635, 114.6079, 282.7722 + 304.2635, 337.3543 + 114.6079])

""" 이 부분 왜 이렇게 더하는지 이해가 안 되네? :s 
        * [x_top, y_top, width + x_top, height + y_top]
        * 아하 ! (x_top, y_top, x_bottom, y_bottom)  형태가 되는구나 
        * bbox의 좌 윗끝 우 아래끝을 나타내기 위함. 
"""

# %%

""" MultiScaleRoIAlign 객체의 forward 메소드 사용법: 

        * x (OrderedDict[Tensor]): feature maps for each level. 
        * boxes (List[Tensor[N, 4]]): boxes to be used to perform the pooling operation, in (x1, y1, x2, y2) format
        * image_shapes (List[Tuple[height, width]]): the sizes of each image before they have been fed to a CNN to obtain feature maps.

(ref) https://github.com/pytorch/vision/blob/master/torchvision/ops/poolers.py                                
"""

roi_pooling_op = roi_pooler(x=fmaps, boxes=[rpn_bbox_rand], image_shapes=[(800, 800)])  # 입력 이미지 텐서의 사이즈는 (1, 3, 800, 800) 로 가정 


print(f"Output from ROI-Pooling layer: {roi_pooling_op.shape} ")    # torch.Size([915, 256, 7, 10])

""" RPN을 통해 예측된 915 개의 bbox 들을 모두 
    RoI Pooling 을 통해 (7, 10) 의 고정된 사이즈로 만듬 
"""



# ================================================================= #
#                           4. RoI Heads                            #
# ================================================================= #
# %% 04. RoI Pooling 의 결과를 활용해 multi tasks 수행하기 
""" Since we now want to connect the output from 'ROI-Pooling' with 'Fully Connected layers', 

        * so we should flatten-out the roi-pooling-output first 
        * and then feed this output to the Linear-layers.


Now we can think of moving to predicting which object each of this 915 regions contain 
and also further refine their box locations with the help of Linear layers        
"""


""" Tasks: 
    1. Class prediction - (1, num_boxes, num_classes)
    2. Bounding-box regression - (1, num_boxes, num_classes * 4)
"""



#%% (1) MLP - FC 레이어 
from torchvision.models.detection import faster_rcnn    # (ref) https://pytorch.org/vision/stable/_modules/torchvision/models/detection/faster_rcnn.html
                                                        # (ref) https://github.com/pytorch/vision/blob/master/torchvision/models/detection/faster_rcnn.py
                                                        # models for addressing different tasks; (ref) https://pytorch.org/vision/stable/models.html

""" `torchvision` provides us an in-built linear layer called `TwoMLPHead()`. 

    It has two arguments: 
        1. the 'in_channels' which is the flattened version of `256*7*10` 
            * roi_pooling 의 결과가 (256, 7, 10) 이니까 

        2. `representation_size` which is kept as 1024.
            * 출력하고 싶은 MLP의 노드 개수 := 1024
"""

mlp_head = faster_rcnn.TwoMLPHead(in_channels = 256*7*10, representation_size=1024) # MLP 계층 초기화 
                                                                                    # (ref) https://github.com/pytorch/vision/blob/master/torchvision/models/detection/faster_rcnn.py
                                                                                    # (ref) https://www.programmersought.com/article/91836285020/

print(f"MLPHead is \n{mlp_head}")                           


# %% MLP 에 입력 
mlp_op = mlp_head(roi_pooling_op)  # [RoI Pooling] -> feature maps -> [MLP]

print(f"Output from the mlp-head is: \n\n{mlp_op.shape}")   # torch.Size([915, 1024])
                                                            # (batch_size, feature_size)

""" Note that we did not flatten the 'roi_pooling_op'. 
    The reason is that the `TwoMLPHead` does the flattening for us.

    Flatten 기능은 'TwoMLPHead' 메소드가 알아서 해줌 
"""


# %% (2) the FastRCNNPredictor.
""" You must be wondering by now have we reached the output layer?
    Well, we are almost there! Just one more layer.

    We call this last layer, the 'FastRCNNPredictor'. Again, it is a simple network with one speciality.
    
    It spits out two outputs: 

        1. one for the classification 
        2. the other for the bounding-box regression.
"""

final_layer = faster_rcnn.FastRCNNPredictor(in_channels=1024, num_classes=17)



print(f"Final layer of Faster-RCNN is : \n\n {final_layer} ")


""" As you see, there is a `cls_score` whose output nodes are equal to num_classes
    and `bbox_pred` whose output nodes are equal to 4 times the num_classes.

    Basically, the `68` nodes are broken into 17 such groups such that each set contains 4 nodes
    depicting the offset for x1, y1, x2, y2 location corresponding to that box.

        * 68 = 17 x 4 
"""


# %% Now let's pass in the output of `mlp_head` to the `final_layer`.
final_scores, final_bboxes = final_layer(mlp_op)


print(f"Output from final-layer of Faster-RCNN is: \n{final_scores.shape} \n{final_bboxes.shape}")





# ================================================================= #
#                               정리                                #
# ================================================================= #
# %% 위의 전 과정을 단순하게 구현하기 

""" Can we combine all the topics covered and all the functions called in a very few lines of code? 
    Yes ! 
"""
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import FasterRCNN


""" Get the backbone of any pretrained network, we'll use Alexnet
"""
alexnet = models.alexnet(pretrained=True)
new_backbone = alexnet.features
print(new_backbone)

new_backbone.out_channels = 256
print(new_backbone)
#%%

""" Configure the anchors. We shall have 12 different anchors.
"""
new_anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256),), 
                                      aspect_ratios=((0.5, 1.0, 2.0),))


""" Configure the output size of RoI-Pooling layer. 
    We shall end up with (num_boxes, num_features, 4, 4) after the ROIPooling layer
"""
new_roi_pooler = MultiScaleRoIAlign(featmap_names=['0'], output_size=4, sampling_ratio=1)


""" let's use dummy variables for mean, std, min_size and max_size
"""
min_size = 300
max_size = 500
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

""" Instantiate the Faster-rcnn model with the variables declared above.
"""
frcnn_model = FasterRCNN(   backbone=new_backbone,
                            num_classes=17, 
                            min_size=min_size, 
                            max_size=max_size, 
                            image_mean=mean, 
                            image_std=std, 
                            rpn_anchor_generator=new_anchor_generator, 
                            box_roi_pool=new_roi_pooler
                        )

# %%
"""As you see below, the backbone, rpn and roi_heads are joined to form one big network.
"""
print(frcnn_model)

# %%
