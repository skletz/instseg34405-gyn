import json
import numpy as np
import copy
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import pycocotools.mask as mask_util
import logging

import torch
import torch.distributed as dist
import pickle

import os, sys
from contextlib import contextmanager


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


class CocoEvaluator(object):
    """
        Coco Evaluator encapsulates the evaluation process using the COCO API.

        Code copied and partially modified from class CocoEvaluator obtained from
        https://github.com/pytorch/vision/blob/master/references/detection/coco_eval.py

        Usage:
            - CocoEvaluator() -> set iou types
            - update() -> add results (predictions) obtained with build_coco_results
            - synchronize_between_processes()
            - accumulate()
            - summarize() > this prints values to std.out

    """

    def __init__(self, coco_gt, iou_types, maxDets=[1, 10, 100]):
        """
        Code modified from function update obtained from
        from https://github.com/pytorch/vision/blob/master/references/detection/coco_eval.py

        :param coco_gt:
        :param iou_types:
        :param maxDets:
        """
        assert isinstance(iou_types, (list, tuple))
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt

        self.iou_types = iou_types
        self.coco_eval = {}
        for iou_type in iou_types:
            with suppress_stdout():
                self.coco_eval[iou_type] = COCOeval(coco_gt, iouType=iou_type)

            self.coco_eval[iou_type].params.maxDets = maxDets

        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}
        self.maxDets = maxDets

    def update(self, results):
        """
        Code modified from function update obtained from
        from https://github.com/pytorch/vision/blob/master/references/detection/coco_eval.py

        :param results:
        :return:
        """
        keys = []
        for i in results:
            keys.append(i['image_id'])

        img_ids = set(keys)
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            with suppress_stdout():
                if results:
                    coco_dt = self.coco_gt.loadRes(results)
                else:
                    coco_dt = COCO()

            coco_eval = self.coco_eval[iou_type]

            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = list(img_ids)
            img_ids, eval_imgs = self.evaluate(coco_eval)

            self.eval_imgs[iou_type].append(eval_imgs)

    def update_per_cat(self, results, cat_id):
        """
        Code modified from function update obtained from
        from https://github.com/pytorch/vision/blob/master/references/detection/coco_eval.py

        :param results:
        :return:
        """
        keys = []
        for i in results:
            keys.append(i['image_id'])

        img_ids = set(keys)
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            with suppress_stdout():
                if results:
                    coco_dt = self.coco_gt.loadRes(results)
                else:
                    coco_dt = COCO()

            coco_eval = self.coco_eval[iou_type]

            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = list(img_ids)
            coco_eval.params.catIds = cat_id
            img_ids, eval_imgs = self.evaluate(coco_eval)

            self.eval_imgs[iou_type].append(eval_imgs)

    def synchronize_between_processes(self):
        """

        Code copied from function synchronize_between_processes obtained from
        from https://github.com/pytorch/vision/blob/master/references/detection/coco_eval.py

        :return:
        """
        for iou_type in self.iou_types:
            self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)
            COCOEvaluatorUtil.create_common_coco_eval(self.coco_eval[iou_type], self.img_ids, self.eval_imgs[iou_type])

    def accumulate(self):
        """

        Code copied from function accumulate obtained from
        from https://github.com/pytorch/vision/blob/master/references/detection/coco_eval.py

        :return:
        """
        for coco_eval in self.coco_eval.values():
            with suppress_stdout():
                coco_eval.accumulate()

    def evaluate(self, coco_eval):
        '''
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs

        Code copied from function evaluate obtained from
        from https://github.com/pytorch/vision/blob/master/references/detection/coco_eval.py

        :return: None
        '''
        # tic = time.time()
        # print('Running per image evaluation...')
        p = coco_eval.params
        # add backward compatibility if useSegm is specified in params
        if p.useSegm is not None:
            p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
            print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
        # print('Evaluate annotation type *{}*'.format(p.iouType))
        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        coco_eval.params = p

        coco_eval._prepare()
        # loop through images, area range, max detection number
        catIds = p.catIds if p.useCats else [-1]

        if p.iouType == 'segm' or p.iouType == 'bbox':
            computeIoU = coco_eval.computeIoU
        elif p.iouType == 'keypoints':
            computeIoU = coco_eval.computeOks
        coco_eval.ious = {
            (imgId, catId): computeIoU(imgId, catId)
            for imgId in p.imgIds
            for catId in catIds}

        evaluateImg = coco_eval.evaluateImg
        maxDet = p.maxDets[-1]
        evalImgs = [
            evaluateImg(imgId, catId, areaRng, maxDet)
            for catId in catIds
            for areaRng in p.areaRng
            for imgId in p.imgIds
        ]
        # this is NOT in the pycocotools code, but could be done outside
        evalImgs = np.asarray(evalImgs).reshape(len(catIds), len(p.areaRng), len(p.imgIds))
        coco_eval._paramsEval = copy.deepcopy(coco_eval.params)
        # toc = time.time()
        # print('DONE (t={:0.2f}s).'.format(toc-tic))
        return p.imgIds, evalImgs

    def summarize(self, log=False):
        """
        Code modified from function summarize obtained from
        https://github.com/pytorch/vision/blob/master/references/detection/coco_eval.py

        :param log:
        :return:
        """

        result_dict = {}

        for iou_type, coco_eval in self.coco_eval.items():
            if log:
                logging.info("IoU metric: {}".format(iou_type))

            stats, _dict = self._summarize(coco_eval.params, coco_eval.eval, log)

            for key, val in _dict.items():
                new_key = iou_type + "_" + key
                result_dict[new_key] = val

        return result_dict

    def _summarize(self, params, eval, log=False):
        """
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting

        Code modified from function summarize obtained from
        https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py

        :param params:
        :param eval:
        :param log:
        :return:
        """

        def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
            p = params

            iStr_dict_key = '{}_{}_{}_{}'
            typeStr_dict_key = 'AP' if ap == 1 else 'AR'

            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap == 1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, :, aind, mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, aind, mind]
            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])

            key = iStr_dict_key.format(maxDets, typeStr_dict_key, iouStr, areaRng)
            str = iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s)
            if log:
                logging.info(str)

            return mean_s, {key: mean_s}

        def _summarizeDets():
            stacked_stats = []
            _dict = {}
            for i, val in enumerate(params.maxDets):
                stats = np.zeros((10,))
                stats[0], d = _summarize(1, maxDets=params.maxDets[i])
                _dict.update(d)
                stats[1], d = _summarize(1, iouThr=.5, maxDets=params.maxDets[i])
                _dict.update(d)
                stats[2], d = _summarize(1, iouThr=.75, maxDets=params.maxDets[i])
                _dict.update(d)
                stats[3], d = _summarize(1, areaRng='small', maxDets=params.maxDets[i])
                _dict.update(d)
                stats[4], d = _summarize(1, areaRng='medium', maxDets=params.maxDets[i])
                _dict.update(d)
                stats[5], d = _summarize(1, areaRng='large', maxDets=params.maxDets[i])
                _dict.update(d)
                stats[6], d = _summarize(0, maxDets=params.maxDets[i])
                _dict.update(d)
                stats[7], d = _summarize(0, areaRng='small', maxDets=params.maxDets[i])
                _dict.update(d)
                stats[8], d = _summarize(0, areaRng='medium', maxDets=params.maxDets[i])
                _dict.update(d)
                stats[9], d = _summarize(0, areaRng='large', maxDets=params.maxDets[i])
                _dict.update(d)
                stacked_stats.extend(stats)

            return stacked_stats, _dict

        def _summarizeKps():
            logging.warning("Check method to summarize detections of keypoints")
            stats = np.zeros((10,))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=.75)
            stats[3] = _summarize(1, maxDets=20, areaRng='medium')
            stats[4] = _summarize(1, maxDets=20, areaRng='large')
            stats[5] = _summarize(0, maxDets=20)
            stats[6] = _summarize(0, maxDets=20, iouThr=.5)
            stats[7] = _summarize(0, maxDets=20, iouThr=.75)
            stats[8] = _summarize(0, maxDets=20, areaRng='medium')
            stats[9] = _summarize(0, maxDets=20, areaRng='large')
            return stats

        if not eval:
            raise Exception('Please run accumulate() first')
        iouType = params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets
        elif iouType == 'keypoints':
            summarize = _summarizeKps
        stats, _dict = summarize()

        return stats, _dict

class COCOEvaluatorUtil:
    @staticmethod
    def merge(img_ids, eval_imgs):
        """

        Code copied from function merge obtained from
        https://github.com/pytorch/vision/blob/master/references/detection/coco_eval.py

        :param img_ids:
        :param eval_imgs:
        :return:
        """
        all_img_ids = COCOEvaluatorUtil.all_gather(img_ids)
        all_eval_imgs = COCOEvaluatorUtil.all_gather(eval_imgs)

        merged_img_ids = []
        for p in all_img_ids:
            merged_img_ids.extend(p)

        merged_eval_imgs = []
        for p in all_eval_imgs:
            merged_eval_imgs.append(p)

        merged_img_ids = np.array(merged_img_ids)
        merged_eval_imgs = np.concatenate(merged_eval_imgs, 2)

        # keep only unique (and in sorted order) images
        merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
        merged_eval_imgs = merged_eval_imgs[..., idx]

        return merged_img_ids, merged_eval_imgs

    @staticmethod
    def create_common_coco_eval(coco_eval, img_ids, eval_imgs):
        """

        Code copied from function create_common_coco_eval obtained from
        https://github.com/pytorch/vision/blob/master/references/detection/coco_eval.py

        :param coco_eval:
        :param img_ids:
        :param eval_imgs:
        :return:
        """
        img_ids, eval_imgs = COCOEvaluatorUtil.merge(img_ids, eval_imgs)
        img_ids = list(img_ids)
        eval_imgs = list(eval_imgs.flatten())

        coco_eval.evalImgs = eval_imgs
        coco_eval.params.imgIds = img_ids
        coco_eval._paramsEval = copy.deepcopy(coco_eval.params)

    @staticmethod
    def build_coco_results(image_id, rois, class_ids, scores, masks):
        """

        :param image_id:
        :param rois:
        :param class_ids:
        :param scores:
        :param masks:
        :return:
        """

        rois_encoded = rois.copy()
        masks_encoded = []

        if rois.size == 0:
            return [], [], []

        results = []
        masks = masks > 0.5

        squeeze = False
        if len(masks[0, :, :].shape) >= 3:
            # logging.warning("Shape of mask has more than 2 dims: %s first dim will be squeezed" % (masks[0, :, :].shape,))
            if masks[0, :, :].shape[0] == 1:
                # logging.warning("First dim will be squeezed")
                squeeze = True

        for i in range(rois.shape[0]):

            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[i, :, :].astype(np.uint8)
            if squeeze:
                mask = np.squeeze(mask, axis=0)

            box = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
            l_box = [float(i) for i in box]

            result = {
                "image_id": int(image_id),
                "category_id": int(class_id),
                "bbox": l_box,
                "score": float(score),
                "segmentation": mask_util.encode(np.asfortranarray(mask))
            }
            results.append(result)

            masks_encoded.append(result['segmentation'])
            rois_encoded[i] = result["bbox"]

        return results, masks_encoded, rois_encoded

    @staticmethod
    def encode_masks(masks):
        if masks.size == 0:
            return []
        masks_encoded = []
        for i in range(masks.shape[0]):

            mask = masks[i, :, :].astype(np.uint8)
            if len(masks[i, :, :].shape) >= 3 and masks[i, :, :].shape[0] == 1:
                mask = np.squeeze(mask, axis=0)

            masks_encoded.append(mask_util.encode(np.asfortranarray(mask)))

        return masks_encoded

    @staticmethod
    def encode_bboxes(boxes):
        if boxes.size == 0:
            return []
        bbox_encoded = boxes.copy()
        for i in range(boxes.shape[0]):
            bbox = np.around(boxes[i], 1)
            box = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
            bbox_encoded[i] = box
        return bbox_encoded

    @staticmethod
    def convert_to_empty_result(gt, prediction_template):
        """
        :return:
        """

        prediction_template['boxes'] = np.copy(gt['boxes'])
        prediction_template['labels'] = np.copy(gt['labels'])
        prediction_template['scores'] = np.copy(gt['labels'])
        prediction_template['masks'] = np.copy(gt['masks'])

        prediction_template['boxes'].fill(0)
        prediction_template['labels'].fill(0)
        prediction_template['scores'].fill(0)
        prediction_template['masks'].fill(0)

        return prediction_template

    @staticmethod
    def convert_to_numpy(labels, predictions):
        """
        :return:
        """

        for lind, label in enumerate(labels):
            for tind, (key, val) in enumerate(label.items()):
                label[key] = val.numpy()
                print()

        for pind, prediction in enumerate(predictions):

            if len(prediction) == 0:
                continue

            label = labels[pind]
            image_id = label["image_id"]

            for tind, (key, val) in enumerate(prediction.items()):
                prediction[key] = val.numpy()

            # check if there exits predictions
            if prediction['labels'].size == 0:
                # reproduce shapes of inputs for empty predictions
                # filled with zero values, indicates that no predictions are outputted
                prediction['boxes'] = np.copy(labels[pind]['boxes'])
                prediction['labels'] = np.copy(labels[pind]['labels'])
                prediction['scores'] = np.copy(labels[pind]['labels'])
                prediction['masks'] = np.copy(labels[pind]['masks'])

                prediction['boxes'].fill(0)
                prediction['labels'].fill(0)
                prediction['scores'].fill(0)
                prediction['masks'].fill(0)

                print()

        # for idx, target in enumerate(targets):
        #     image_id = target["image_id"].item()
        #     boxes_gt = target["boxes"].tolist()
        #     labels_gt = target["labels"].tolist()
        #     masks_gt = target["masks"].tolist()
        #
        #     boxes = outputs[idx]["boxes"].tolist()
        #     labels = outputs[idx]["labels"].tolist()
        #     scores = outputs[idx]["scores"].tolist()
        #     masks = outputs[idx]["masks"].tolist()
        #
        #     if len(labels) == 0:
        #         logging.warning("Nothing detected in image: %d" % image_id)
        #         gt = target["boxes"].numpy()
        #         gt.fill(0)
        #         boxes = gt.tolist()
        #
        #         gt = target["labels"].numpy()
        #         gt.fill(0)
        #         labels = gt.tolist()
        #         scores = gt.tolist()
        #
        #         gt = target["masks"].numpy()
        #         gt.fill(0)
        #         masks = gt.tolist()
        #
        #     t[image_id] = {"gt": {}, "dt": {}}
        #     t[image_id]["gt"]['boxes'] = boxes_gt
        #     t[image_id]["gt"]['labels'] = labels_gt
        #     t[image_id]["gt"]['masks'] = masks_gt
        #
        #     t[image_id]["dt"]['boxes'] = boxes
        #     t[image_id]["dt"]['labels'] = labels
        #     t[image_id]["dt"]['scores'] = scores
        #     t[image_id]["dt"]['masks'] = masks

        return labels, predictions

    @staticmethod
    def build_coco_results_tensors(image_id, rois, class_ids, scores, masks):
        results = []

        return results


    import PIL

    @staticmethod
    def convert_to_coco_api(ds):
        """

        Code modified from function convert_to_coco_api obtained from
        https://github.com/pytorch/vision/blob/master/references/detection/coco_eval.py

        :param ds:
        :return:
        """

        coco_ds = COCO()
        # ann_id has to start with 1, because generated coco results starts also by 1
        # If it is zero, then annos does not match with one element
        # because gt list starts with 0 and dt list starts with 1
        # load
        ann_id = 1
        dataset = {'images': [], 'categories': [], 'annotations': []}
        categories = set()
        for img_idx in range(len(ds)):
            # find better way to get target
            # targets = ds.get_annotations(img_idx)
            (img, targets), (filenames) = ds[img_idx]
            image_id = targets["image_id"].item()
            img_dict = {}
            img_dict['id'] = image_id

            if hasattr(img, "height"):
                img_dict['height'] = img.height
                img_dict['width'] = img.width
            else:
                img_dict['height'] = img.shape[-2]
                img_dict['width'] = img.shape[-1]

            dataset['images'].append(img_dict)
            bboxes = targets["boxes"]
            bboxes[:, 2:] -= bboxes[:, :2]
            bboxes = bboxes.tolist()
            labels = targets['labels'].tolist()
            areas = targets['area'].tolist()
            iscrowd = targets['iscrowd'].tolist()
            if 'masks' in targets:
                masks = targets['masks']
                # make masks Fortran contiguous for coco_mask
                masks = masks.permute(0, 2, 1).contiguous().permute(0, 2, 1)
            if 'keypoints' in targets:
                keypoints = targets['keypoints']
                keypoints = keypoints.reshape(keypoints.shape[0], -1).tolist()
            num_objs = len(bboxes)
            for i in range(num_objs):
                ann = {}
                ann['image_id'] = image_id
                ann['bbox'] = bboxes[i]
                ann['category_id'] = labels[i]
                categories.add(labels[i])
                ann['area'] = areas[i]
                ann['iscrowd'] = iscrowd[i]
                ann['id'] = ann_id
                if 'masks' in targets:
                    ann["segmentation"] = mask_util.encode(masks[i].numpy())
                if 'keypoints' in targets:
                    ann['keypoints'] = keypoints[i]
                    ann['num_keypoints'] = sum(k != 0 for k in keypoints[i][2::3])
                dataset['annotations'].append(ann)
                ann_id += 1
        dataset['categories'] = [{'id': i} for i in sorted(categories)]
        coco_ds.dataset = dataset
        with suppress_stdout():
            coco_ds.createIndex()
        return coco_ds

    @staticmethod
    def deserialize(_dict):
        """

        :param _dict:
        :return:
        """
        converted = {}

        for img_id, result in _dict.items():
            boxes = result['gt']["boxes"]
            labels = result['gt']["labels"]
            masks = result['gt']["masks"]

            boxes = np.asarray(boxes, dtype=np.float32)
            labels = np.asarray(labels, dtype=np.int)
            masks = np.asarray(masks, dtype=np.float32)

            converted[img_id] = {"gt": {}, "dt": {}}
            converted[img_id]["gt"]['boxes'] = boxes
            converted[img_id]["gt"]['labels'] = labels
            converted[img_id]["gt"]['masks'] = masks

            dt_boxes = result['dt']["boxes"]
            dt_labels = result['dt']["labels"]
            dt_masks = result['dt']["masks"]
            scores = result['dt']["scores"]
            dt_boxes = np.asarray(dt_boxes, dtype=np.float32)
            dt_labels = np.asarray(dt_labels, dtype=np.int)
            dt_masks = np.asarray(dt_masks, dtype=np.float32)

            converted[img_id]["dt"]['boxes'] = dt_boxes
            converted[img_id]["dt"]['labels'] = dt_labels
            converted[img_id]["dt"]['masks'] = dt_masks
            converted[img_id]["dt"]['scores'] = scores

        return converted

    @staticmethod
    def deserialize_detection_with_gt(_dict):
        """
        Set the gt as dt in order to test evaluation metrics.
        :param _dict:
        :return:
        """
        converted = {}

        for img_id, result in _dict.items():
            boxes = result['gt']["boxes"]
            labels = result['gt']["labels"]
            masks = result['gt']["masks"]

            boxes = np.asarray(boxes, dtype=np.float32)
            labels = np.asarray(labels, dtype=np.int)
            masks = np.asarray(masks, dtype=np.float32)

            converted[img_id] = {"gt": {}, "dt": {}}
            converted[img_id]["gt"]['boxes'] = boxes
            converted[img_id]["gt"]['labels'] = labels
            converted[img_id]["gt"]['masks'] = masks

            scores = np.ones(labels.shape, dtype=np.float32)

            converted[img_id]["dt"]['boxes'] = boxes
            converted[img_id]["dt"]['labels'] = labels
            converted[img_id]["dt"]['masks'] = masks
            converted[img_id]["dt"]['scores'] = scores

        return converted

    @staticmethod
    def is_dist_avail_and_initialized():
        if not dist.is_available():
            return False
        if not dist.is_initialized():
            return False
        return True

    @staticmethod
    def get_world_size():
        if not COCOEvaluatorUtil.is_dist_avail_and_initialized():
            return 1
        return dist.get_world_size()

    @staticmethod
    def all_gather(data):
        """
        Run all_gather on arbitrary picklable data (not necessarily tensors)
        Args:
            data: any picklable object
        Returns:
            list[data]: list of data gathered from each rank
        """
        world_size = COCOEvaluatorUtil.get_world_size()
        if world_size == 1:
            return [data]

        # serialized to a Tensor
        buffer = pickle.dumps(data)
        storage = torch.ByteStorage.from_buffer(buffer)
        tensor = torch.ByteTensor(storage).to("cuda")

        # obtain Tensor size of each rank
        local_size = torch.tensor([tensor.numel()], device="cuda")
        size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
        dist.all_gather(size_list, local_size)
        size_list = [int(size.item()) for size in size_list]
        max_size = max(size_list)

        # receiving Tensor from all ranks
        # we pad the tensor because torch all_gather does not support
        # gathering tensors of different shapes
        tensor_list = []
        for _ in size_list:
            tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
        if local_size != max_size:
            padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
            tensor = torch.cat((tensor, padding), dim=0)
        dist.all_gather(tensor_list, tensor)

        data_list = []
        for size, tensor in zip(size_list, tensor_list):
            buffer = tensor.cpu().numpy().tobytes()[:size]
            data_list.append(pickle.loads(buffer))

        return data_list