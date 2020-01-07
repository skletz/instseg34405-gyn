import numpy as np
import cv2
import torch

from torchvision.transforms import functional
from utils.annotation import AnnotationUtil
from model.mrcnn_resnet import MaskRCNNResNet
from model.data.target_objects import INSTRUMENT_MAPPING, LABEL_MAPPING
from model.data.target_objects import Instrument

FONT = cv2.FONT_HERSHEY_COMPLEX_SMALL
FONT_SCALE = 0.6


class InstSegPredictorDemo:
    num_classes = 7
    min_size = 360
    max_size = 540
    box_detections_per_img = 10
    score_threshold = 0.8

    def __init__(self, device=torch.device('cpu'), det_model_uri=None):
        self.device = device
        self.det_model_uri = det_model_uri

        self.det_model_uri = det_model_uri

        self.det_model = MaskRCNNResNet(num_classes=self.num_classes, pre_trained=True,
                                        uri=self.det_model_uri,
                                        min_size=self.min_size,
                                        max_size=self.max_size,
                                        box_detections_per_img=self.box_detections_per_img,
                                        box_score_thresh=self.score_threshold, box_nms_thresh=0.15)
        self.device = device
        self.det_model.to(self.device)
        self.det_model.eval()

    def set_image_dimension(self, image_width, image_height):
        self.max_size = image_width
        self.min_size = image_height

    def predict_instruments(self, image):
        """
        Applies the model to a single input image
        :param image: Tensor[C,W,H]
        :return: (dict, int) the dict is represented as:
        {'labels': numpy.array[N],
        'scores' numpy.array[N],
        'boxes' numpy.array[N,4],
        'masks' numpy.array[N,W,H]}, where N represents the number of instances
        """

        image = functional.to_tensor(image)
        image = image.to(self.device)
        input_images = image.detach().clone()
        if len(list(image.shape)) == 3:
            input_images = input_images.unsqueeze(0)

        with torch.no_grad():
            predictions = self.det_model(input_images)

        outputs = []
        outputs_nr_instances = []
        for prediction in predictions:
            masks = prediction['masks'].squeeze(1)
            output = {
                'labels': prediction['labels'].cpu().numpy(),
                'scores': prediction['scores'].cpu().numpy(),
                'boxes': prediction['boxes'].cpu().numpy(),
                'masks': masks.cpu().numpy()
            }
            outputs.append(output)
            nr_instances = output['masks'].shape[0]
            outputs_nr_instances.append(nr_instances)

        if len(outputs) == 0:
            return outputs, 0
        if len(outputs) == 1:
            return outputs[0], outputs_nr_instances[0]
        else:
            return outputs, outputs_nr_instances

    def prediction_to_objects(self, prediction):

        labels = prediction['labels']
        masks = prediction['masks']
        bboxs = prediction['boxes']
        scores = prediction['scores']
        scores_t = np.where(scores > self.score_threshold)
        labels = labels[scores_t]

        instruments = []

        for instance, label in enumerate(labels):
            mask = masks[instance, ...]
            bbox = bboxs[instance, ...]
            score = scores[instance, ...]
            mask[mask > 0.5] = 1
            label_name = LABEL_MAPPING[label]
            object_rgb = INSTRUMENT_MAPPING[label_name][2]
            object_name = INSTRUMENT_MAPPING[label_name][-1]
            instrument = Instrument(object_name, label, mask, bbox, score.item())
            instrument.set_rgb(object_rgb)
            instruments.append(instrument)
        return instruments

    def apply_predictions(self, image, prediction):

        labels = prediction['labels']
        masks = prediction['masks']
        bboxs = prediction['boxes']
        scores = prediction['scores']
        scores_t = np.where(scores > self.score_threshold)
        labels = labels[scores_t]

        countour_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        caption = "Instances: {:d}".format(len(labels))
        cv2.putText(countour_image, caption, (5, 20), fontFace=FONT, fontScale=FONT_SCALE, color=(255, 255, 255))

        for instance, label in enumerate(labels):
            mask = masks[instance, ...]
            bbox = bboxs[instance, ...]

            label_name = LABEL_MAPPING[label]
            object_rgb = INSTRUMENT_MAPPING[label_name][2]
            object_name = INSTRUMENT_MAPPING[label_name][-1]

            # AnnotationUtil.add_bounding_box(countour_image, object_rgb, bbox)
            mask[mask > 0.5] = 1
            AnnotationUtil.add_contours(countour_image, mask, object_rgb)
            label = "{} {:d}: {:.4f}".format(object_name, instance, scores[instance, ...])
            AnnotationUtil.add_label(countour_image, label, bbox, object_rgb)

        return countour_image

    def apply_instruments(self, image, instruments):

        countour_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        caption = "Instances: {:d}".format(len(instruments))
        cv2.putText(countour_image, caption, (5, 20), fontFace=FONT, fontScale=FONT_SCALE, color=(255, 255, 255))

        for instance, instrument in enumerate(instruments):
            AnnotationUtil.add_contours(countour_image, instrument.mask, instrument.rgb)
            label = "{} {:d}: {:.4f}".format(instrument.label, instance + 1, instrument.score)
            AnnotationUtil.add_label(countour_image, label, instrument.bbox, instrument.rgb)

        return countour_image
