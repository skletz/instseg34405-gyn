""" Instrument Segmentation Model.

    Implementation is based on Mask R-CNN and ResNet.
    ******************************************************
          _    _      ()_()
         | |  | |    |(o o)
      ___| | _| | ooO--`o'--Ooo
     / __| |/ / |/ _ \ __|_  /
     \__ \   <| |  __/ |_ / /
     |___/_|\_\_|\___|\__/___|
    ******************************************************
    @author skletz
    @version 1.0 09/01/19
"""

import torch
from torchvision.models.utils import load_state_dict_from_url
from torchvision.models.detection.mask_rcnn import MaskRCNN

from model.torchvision_mrcnn.nets.mrcnn import (MaskRCNN, MaskRCNNPredictor, FastRCNNPredictor)
from model.torchvision_mrcnn.nets.fpn import (BackboneWithFPN)
from model.torchvision_mrcnn.nets.resnet import (ResNet, Bottleneck)
from torchvision.ops import misc as misc_nn_ops

from urllib.parse import urlparse
import logging


class MaskRCNNResNet:
    params_to_update = []

    def __init__(self, num_classes=7,
                 pre_trained=True,
                 uri='https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth',
                 **kwargs):

        pre_defined_kwargs = {
            'rpn_pre_nms_top_n_train': 2000,
            'rpn_post_nms_top_n_train': 2000,
            'rpn_pre_nms_top_n_test': 1000,
            'rpn_post_nms_top_n_test': 1000,
        }

        for key in kwargs:
            logging.debug("Using parameters {}={}.".format(key, kwargs[key]))

        for key in pre_defined_kwargs:
            if key not in kwargs:
                kwargs[key] = pre_defined_kwargs[key]
                logging.debug("Extend parameters with default settings {}={}.".format(key, kwargs[key]))

        super(MaskRCNNResNet, self).__init__()

        self.num_classes = num_classes
        self.pre_trained = pre_trained

        backbone = self.resnet50_fpn_backbone(num_classes=self.num_classes)

        init_before_load = True
        if not self.pre_trained:
            self.maskrcnn = MaskRCNN(backbone, self.num_classes, **kwargs)
            init_before_load = False
        else:
            default_class_num = 91
            self.maskrcnn = MaskRCNN(backbone, num_classes=default_class_num, **kwargs)
            if urlparse(uri).scheme in ('http', 'https',):
                state_dict = load_state_dict_from_url(uri, progress=True)
                init_before_load = False
            else:
                try:
                    checkpoint = torch.load(uri)
                except RuntimeError as e:
                    logging.warning(e)
                    checkpoint = torch.load(uri, map_location='cpu')
                    logging.warning("Model loaded with map_location='cpu'")
                state_dict = checkpoint['model_state_dict']

            if not init_before_load:
                self.maskrcnn.load_state_dict(state_dict, strict=False)

        # get number of input features for the classifier
        in_features = self.maskrcnn.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        self.maskrcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        # now get the number of input features for the mask classifier
        in_features_mask = self.maskrcnn.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # and replace the mask predictor with a new one
        self.maskrcnn.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                                   hidden_layer,
                                                                   num_classes)

        self.backbone = self.maskrcnn.backbone

        if init_before_load:
            self.maskrcnn.load_state_dict(state_dict, strict=True)

        for name, param in self.maskrcnn.named_parameters():
            if param.requires_grad == True:
                self.params_to_update.append(param)
            else:
                logging.debug("Frozen: %s" % name)

        pass

    def __call__(self, *args, **kwargs):
        output = self.maskrcnn(*args)
        return output

    def to(self, device):
        self.maskrcnn.to(device)

    def eval(self):
        self.maskrcnn.eval()

    def train(self):
        self.maskrcnn.train()

    def parameters(self):
        return self.maskrcnn.parameters()

    def state_dict(self):
        return self.maskrcnn.state_dict()

    def load_state_dict(self, state_dict):
        self.maskrcnn.load_state_dict(state_dict)

    @staticmethod
    def resnet50_fpn_backbone(num_classes=2):
        """
        Source-code copied and modified from file backbone_utils.py obtained from
        https://github.com/pytorch/vision/blob/master/torchvision/models/detection/backbone_utils.py

        :param num_classes:
        :return:
        """
        # ResNet 50
        layers = [3, 4, 6, 3]

        backbone = ResNet(block=Bottleneck, layers=layers, num_classes=num_classes,
                          norm_layer=misc_nn_ops.FrozenBatchNorm2d)

        num_ftrs = backbone.fc.in_features
        backbone.fc = torch.nn.Linear(num_ftrs, num_classes)

        for name, parameter in backbone.named_parameters():
            if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)

        return_layers = {'layer1': 0, 'layer2': 1, 'layer3': 2, 'layer4': 3}

        in_channels_stage2 = 256
        in_channels_list = [
            in_channels_stage2,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]
        out_channels = 256
        return BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels)
