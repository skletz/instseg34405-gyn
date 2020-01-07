import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
from utils.annotation import AnnotationUtil


class ImageAugmentor(object):

    def __init__(self):
        ia.seed(0)
        sometimes = lambda aug: iaa.Sometimes(0.25, aug)
        self.seq = iaa.Sequential([
            iaa.Fliplr(0.25),  # horizontally flip 25% of the images
            iaa.Flipud(0.25),  # horizontally flip 25% of the images
            sometimes(
                iaa.OneOf([
                    iaa.GaussianBlur(sigma=(0, 3.0)),
                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5))
                ]),
            )
        ], random_order=True)

    def __call__(self, image, target):
        # augmentation choices
        seq_det = self.seq.to_deterministic()

        image = np.array(image.copy())
        # AnnotationUtil.display_image(image)
        image = seq_det.augment_image(image)  # , hooks=self.hooks)
        masks = target['masks']

        # AnnotationUtil.display_image(image)
        num_masks = masks.shape[0]
        masks_aug = []
        boxes = []
        for i in range(num_masks):
            mask = masks[i, ...].copy()
            # AnnotationUtil.display_mask(mask)
            segmmap = ia.SegmentationMapOnImage(mask, shape=image.shape, nb_classes=1 + 1)
            segmmap_aug = seq_det.augment_segmentation_maps([segmmap])[0]
            mask_aug = segmmap_aug.get_arr_int()
            masks_aug.append(mask_aug)
            # AnnotationUtil.display_mask(mask_aug)

            bbox = AnnotationUtil.get_bbox_from_mask(mask_aug)
            boxes.append(bbox)

        masks = np.asanyarray(masks_aug)
        boxes = np.array(boxes)
        target['masks'] = masks
        target['boxes'] = boxes
        return image, target
