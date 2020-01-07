import numpy as np
import cv2
from torchvision.transforms import functional as F
import skimage
from pycocotools import mask as COCOmask


class AnnotationUtil(object):
    @staticmethod
    def get_bbox_from_mask(mask):
        pos = np.where(mask)
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        bbox = [xmin, ymin, xmax, ymax]
        return bbox

    @classmethod
    def apply_tensor_mask_to_tensor_image(cls, image, masks):
        # import matplotlib.pyplot as plt
        nr_instances = masks.shape[0]

        rgb = (219, 249, 122)

        output_image = image.permute(1, 2, 0)
        output_image = output_image.data.cpu().numpy()
        output_image = output_image.copy()

        output_masks = masks.data.cpu().numpy()
        for instance in range(0, nr_instances):
            mask = output_masks[instance, ...]
            cls.add_contours(output_image, mask, rgb)

        # plt.imshow(output_image)
        # plt.axis('off')
        # plt.show()
        o = F.to_tensor(output_image).float()
        return o

    @staticmethod
    def add_bounding_box(image, rgb, bbox):
        """

        :param image:
        :param rgb:
        :param bbox:
        :return:
        """
        bgr = (int(rgb[0]), int(rgb[1]), int(rgb[2]))

        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])
        p1 = (x1, y1)
        p2 = (x2, y2)
        cv2.rectangle(image, p1, p2, bgr, thickness=2)

        return image

    @staticmethod
    def add_contours(image, mask, rgb, convert=False):
        """
        Apply the mask to the image.
        :param image: numpy array
        :param mask:
        :param rgb:
        :param alpha:
        :return:
        """
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)

        padded_mask[1:-1, 1:-1] = mask

        # we extract the counters of the instrument
        # a vertex is point at which two polygon edges of a polygon meet
        contours = skimage.measure.find_contours(padded_mask, 0.0)

        for coordinates in contours:
            # Make sure the coordinates are expressed as integers
            coordinates = np.fliplr(coordinates) - 2
            coordinates = coordinates.astype(int)
            cv2.drawContours(image, [coordinates], 0, rgb, 2)
        return image


    @staticmethod
    def add_label(image, label, bbox, rgb):
        """

        :param image:
        :param rgb:
        :param bbox:
        :param text:
        :param label:
        :return:
        """
        bgr = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
        #
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])

        caption = label
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        font_scale = 0.6
        (text_width, text_height) = cv2.getTextSize(caption, font, fontScale=font_scale, thickness=1)[0]

        # set the text start position
        if y1 < text_height:
            text_offset_y = y2 + text_height
            if text_offset_y > image.shape[0]:
                text_offset_y = y1 + text_height
        else:
            text_offset_y = y1

        if x1 + text_width >= image.shape[1]:
            text_offset_x = image.shape[1] - text_width
        else:
            text_offset_x = x1

        box_coords = (
            (text_offset_x, text_offset_y), (text_offset_x + text_width - 2, text_offset_y - text_height - 2))

        cv2.rectangle(image, box_coords[0], box_coords[1], bgr, cv2.FILLED)
        cv2.putText(image, caption, (text_offset_x, text_offset_y), font, font_scale, (255, 255, 255),
                    lineType=cv2.LINE_AA)

        return image

    @staticmethod
    def add_instance_heatmap(image, mask, viz_heatmap=False):
        """
        Apply the mask to the image.
        :param image: numpy array
        :param mask:
        :param rgb:
        :param alpha:
        :return:
        """
        weighted = image.copy()

        if viz_heatmap:
            inverted = np.uint8(mask * 255)
            heatmap = cv2.applyColorMap(inverted, cv2.COLORMAP_JET)
            heatmap[np.where(mask < 0.1)] = 0
            mask_ignore = ~(heatmap == [0, 0, 0]).all(-1)
            weighted[mask_ignore] = weighted[mask_ignore] * 0.5 + heatmap[mask_ignore] * 0.5

            # alpha = 0.5
            # weighted = cv2.addWeighted(weighted, alpha, heatmap, 1 - alpha, 0)
            # import matplotlib.pyplot as plt
            # plt.imshow(weighted)
            # plt.show()

        return weighted

    @staticmethod
    def display_image(image):

        import matplotlib.pyplot as plt
        plt.imshow(image)
        plt.axis('off')
        plt.show()

    @staticmethod
    def display_mask(mask):

        import matplotlib.pyplot as plt
        plt.imshow(mask)
        plt.axis('off')
        plt.show()

    @staticmethod
    def coco_annotation_to_mask(ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = AnnotationUtil.coco_annotation_to_RLE(ann, height, width)
        m = COCOmask.decode(rle)
        return m

    @staticmethod
    def coco_annotation_to_RLE(ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = COCOmask.frPyObjects(segm, height, width)
            rle = COCOmask.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = COCOmask.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle
