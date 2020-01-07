import os
import logging
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from pycocotools.coco import COCO
from model.data.data_directory import DataDirectory
from torchvision.transforms import functional
from augmentation.image_augmentor import ImageAugmentor
from utils.annotation import AnnotationUtil

import matplotlib.pyplot as plt


class LisLocDataset(Dataset):
    """
    Dataset specification for laparoscopic instruments localization
    describing annotations stored in the COCO JSON data format.
    """
    root_dir = ""
    annotation_file = ""
    image_dir = " "
    target_task = ""

    image_info = []
    class_info = [{"id": 0, "name": "BG"}]
    image_ids = []
    filenames = []
    labels = []

    augment = False
    augmentor = ImageAugmentor()

    def __init__(self, data_dir: DataDirectory, target_task: str):
        super(LisLocDataset, self).__init__()
        self.root_dir = data_dir.get_root_dir()
        self.annotation_file = data_dir.get_json_annotation_file()
        self.image_dir = data_dir.get_image_dir()
        self.target_task = target_task

        self.set_target_task(target_task)
        self.load_annotations(self.annotation_file)

        logging.debug("Init new dataset ...")
        self.filenames = [None] * len(self.image_info)
        self.labels = [None] * len(self.image_info)

        for i in range(len(self.image_info)):
            self.labels[i] = self.image_info[i]['cats']
            self.filenames[i] = self.image_info[i]['filename']

        self.prepare()
        self.get_multilabels()

        for i in self.image_info:
            self.image_ids.append(i['id'])

    def __len__(self):
        return len(self.image_info)

    def __getitem__(self, idx):

        image_id = self.image_info[idx]['id']
        image_fname = self.image_info[idx]['filename']

        image = self.load_image(idx)
        masks, labels = self.load_mask(idx)

        boxes = []
        nr_objects = len(labels)

        for i in range(nr_objects):
            bbox = AnnotationUtil.get_bbox_from_mask(masks[i, ...])
            boxes.append(bbox)

        boxes = np.array(boxes)
        labels = np.array(labels)
        areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        mask = np.zeros(shape=(masks.shape[1], masks.shape[2]), dtype=np.int64)
        for idx, label in enumerate(labels):
            m = masks[idx, ...]
            mask[m == 1] = label

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "mask": mask,
            "image_id": image_id,
            "area": areas,
            "iscrowd": [0] * nr_objects
        }

        if self.augment:
            image, target = self.augmentor(image, target)
        else:
            print()

        target['boxes'] = torch.as_tensor(target['boxes'], dtype=torch.float32)
        target['labels'] = torch.as_tensor(target['labels'], dtype=torch.int64)
        target['masks'] = torch.as_tensor(target['masks'], dtype=torch.uint8)
        target['mask'] = torch.as_tensor(target['mask'], dtype=torch.uint8)
        target['image_id'] = torch.tensor([target['image_id']], dtype=torch.int64)
        target['area'] = torch.tensor(target['area'], dtype=torch.int64)
        target['iscrowd'] = torch.tensor(target['iscrowd'], dtype=torch.uint8)

        image = functional.to_tensor(image)

        target = self._get_target(target)

        return (image, target), (image_fname, image_id)

    def set_target_task(self, mode):
        self.target_task = mode
        if self.target_task == "seg":
            self._get_target = self._get_segmentation_target
        elif self.target_task == "det":
            self._get_target = self._get_detection_target
        else:
            self._get_target = self._get_detection_and_segmentation_target

    def _get_segmentation_target(self, target):
        return target['mask']

    def _get_detection_target(self, target):
        """

        :param target:
        :return:
        """
        new_target = {
            "boxes": target["boxes"],
            "labels": target["labels"],
            "image_id": target["image_id"],
            "area": target["area"],
            "iscrowd": target["iscrowd"]
        }

        return new_target

    def _get_detection_and_segmentation_target(self, target):
        """

        :param target:
        :return:
        """

        return target

    def load_annotations(self, annotation_file):

        self.COCO = COCO(annotation_file)

        category_ids = self.COCO.getCatIds()
        for i in category_ids:
            cat = self.COCO.loadCats(i)[0]["name"]
            cat_image_ids = self.COCO.getImgIds(catIds=i)

            if not len(cat_image_ids):
                # no annotations exists
                continue

            self.add_class(i, cat)
        logging.info("self.class_info: %s" % self.class_info)

        # Set up dataset classes
        self.category_ids = self.COCO.getCatIds()
        for idx, i in enumerate(self.COCO.getImgIds()):
            path = os.path.join(self.image_dir, self.COCO.imgs[i]['file_name'])
            width = self.COCO.imgs[i]["width"]
            height = self.COCO.imgs[i]["height"]
            filename = self.COCO.imgs[i]['file_name']

            ann_ids = self.COCO.getAnnIds(imgIds=[i], catIds=self.category_ids, iscrowd=None)
            annotations = self.COCO.loadAnns(ann_ids)
            cats = [a['category_id'] for a in annotations]

            self.add_image(
                image_id=i,
                path=path,
                filename=os.path.basename(filename),
                width=width,
                height=height,
                cats=cats,
                annotations=annotations
            )

    def get_multilabels(self):

        tmp = []
        for label in self.labels:
            l = [0] * len(self.class_from_source_map)
            for i in label:
                index = self.class_from_source_map[i]
                l[index] = 1
            tmp.append(l)

        self.labels = tmp

    def prepare(self):
        """Prepares the Dataset class for use.

        TODO: class map is not supported yet. When done, it should handle mapping
              classes from different datasets to the same class ID.
        """

        def clean_name(name):
            """Returns a shorter version of object names for cleaner display."""
            return ",".join(name.split(",")[:1])

        # Build (or rebuild) everything else from the info dicts.
        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [clean_name(c["name"]) for c in self.class_info]
        self.num_images = len(self.image_info)

        # Mapping from source class to internal IDs
        self.class_from_source_map = {info['id']: id
                                      for info, id in zip(self.class_info, self.class_ids)}
        self.source_from_class_map = {id: info['id']
                                      for info, id in zip(self.class_info, self.class_ids)}
        self.source_to_class_names_map = {info['id']: info['name']
                                          for info, id in zip(self.class_info, self.class_ids)}

        logging.info("self.class_from_source_map %s" % self.class_from_source_map)
        logging.info("self.source_from_class_map %s" % self.source_from_class_map)
        logging.info("self.source_to_class_names_map %s" % self.source_to_class_names_map)

    def map_source_class_id(self, source_class_id):
        """Takes a source class ID and returns the int class ID assigned to it.

        For example:
        dataset.map_source_class_id("coco.12") -> 23
        """
        return self.class_from_source_map[source_class_id]

    def load_image(self, idx=None, id=None):

        if idx is not None and id is None:
            img_path = self.image_info[idx]['path']
        else:
            img_path = ""
            for i in self.image_info:
                if i['id'] == id:
                    img_path = i['path']
            # img_path = [i['path'] for i in self.image_info  if i['id'] == id][0]

        # @ToDo: most frameworks use BGR, own impl of googlenet assume BGR for mean substraction
        image = Image.open(img_path).convert("RGB")
        return image

    def add_class(self, class_id, class_name):

        # Does the class exist already?
        for info in self.class_info:
            if info["id"] == class_id:
                # source.class_id combination already available, skip
                return
        # Add the class
        self.class_info.append({
            "id": class_id,
            "name": class_name,
        })

    def add_image(self, image_id, path, **kwargs):
        image_info = {
            "id": image_id,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    def load_mask(self, image_id):
        """
        Generate instance masks for an image.
        :param image_id:
        :return: tuple
            - masks:  A bool array of shape [height, width, instance count] with
            one mask per instance.
            - class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            # class_id = int(annotation['category_id'])

            class_id = self.map_source_class_id(annotation['category_id'])

            if class_id:
                m = AnnotationUtil.coco_annotation_to_mask(annotation, image_info["height"],
                                                           image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=0).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            print("No class Id")


class LisLocSubset(Dataset):
    def __init__(self, dataset, indices, activate_augmentation):
        self.dataset = dataset
        self.indices = indices
        self.augment = activate_augmentation

    def __getitem__(self, idx):
        dataset_augmentation_setting = self.dataset.augment
        self.dataset.augment = self.augment

        (image, target), (filename, id) = self.dataset[self.indices[idx]]

        self.dataset.augment = dataset_augmentation_setting

        return (image, target), (filename, id)

    def __len__(self):
        return len(self.indices)

    def get_origin_category_id(self, mapped_category_id):
        return self.dataset.source_from_class_map[mapped_category_id]

    def get_category_name(self, category_id):
        return self.dataset.source_to_class_names_map[category_id]

    def get_origin_category_name(self, category_id):
        id = self.dataset.source_from_class_map[category_id]
        return self.dataset.source_to_class_names_map[id]
