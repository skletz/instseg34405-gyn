import logging
from tqdm import tqdm
import torch
import time
from evaluation.coco_evaluator import CocoEvaluator, COCOEvaluatorUtil


def evaluate_with_coco_metric(results, dataset, iou_types, maxDets):
    metrics = {}

    since_coco_api = time.time()
    coco_gt = COCOEvaluatorUtil.convert_to_coco_api(dataset)
    coco_evaluator = CocoEvaluator(coco_gt, iou_types, maxDets)

    empty_result_counter = 0
    for ind, (key, val) in enumerate(results.items()):
        image_id = val['image_id']
        boxes = val['boxes']
        labels = val['labels']
        scores = val['scores']
        masks = val['masks']
        image_results, _, _ = COCOEvaluatorUtil.build_coco_results(image_id, boxes, labels, scores, masks)
        if len(image_results) > 0:
            coco_evaluator.update(image_results)
        else:
            logging.warning("Image ID %s has no results!" % image_id)
            empty_result_counter = empty_result_counter + 1

    if empty_result_counter < len(results):
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        stats = coco_evaluator.summarize(log=False)
        metrics.update(stats)
    else:
        logging.warning("No results to evaluate!")
        metrics = {}

    time_cocoapi_elapsed = time.time() - since_coco_api
    logging.debug('Average (evaluation) time to evaluate with coco api {:.5f}ms'.format(time_cocoapi_elapsed * 1000))

    if len(metrics) == 0:
        logging.warning("No results are found!")

    return metrics


def predict(experimetnal_setup, dataloader, stdout=None):
    results = {}

    experimetnal_setup.model.eval()

    n_threads = torch.get_num_threads()
    cpu_device = torch.device("cpu")
    torch.set_num_threads(1)

    with tqdm(desc="Predict-Batch", total=len(dataloader), file=stdout, dynamic_ncols=True) as t:
        for i, data in enumerate(dataloader, 0):
            (inputs, labels), files = data

            inputs = list(image.to(experimetnal_setup.device) for image in inputs)
            labels = [{k: v.to(experimetnal_setup.device) for k, v in t.items()} for t in labels]

            outputs = experimetnal_setup.model(inputs)

            outputs = [{k: v.to(cpu_device).detach() for k, v in t.items()} for t in outputs]
            labels = [{k: v.to(cpu_device).detach() for k, v in t.items()} for t in labels]

            num_results = len(outputs)
            for i in range(num_results):
                predictions = outputs[i]

                result = {
                    'image_id': labels[i]["image_id"].item(),
                    'labels': predictions['labels'].numpy(),
                    'scores': predictions['scores'].numpy(),
                    'boxes': predictions['boxes'].numpy(),
                    'masks': predictions['masks'].numpy()
                }

                results[files[i][0]] = result

            t.update()

    torch.set_num_threads(n_threads)

    return results
