import logging
from tqdm import tqdm
import time
import math
import io
from typing import Dict, Optional
from utils.metric_logger import MetricLogger
from experiment.experimental_setup import ExperimentalSetup


def validate(experimetnal_setup: ExperimentalSetup, stdout: Optional[io.TextIOWrapper] = None) -> Dict:
    """

    :param experimetnal_setup:
    :param stdout:
    :return:
    """
    dataloader = experimetnal_setup.get_val_dataloader()

    metriclogger = MetricLogger()

    experimetnal_setup.model.train()

    with tqdm(desc="Validate-Batch", total=len(dataloader), file=stdout, dynamic_ncols=True) as t:

        for i, data in enumerate(dataloader, 0):
            since_batch = time.time()

            (inputs, labels), files = data

            inputs = list(image.to(experimetnal_setup.device) for image in inputs)
            labels = [{k: v.to(experimetnal_setup.device) for k, v in t.items()} for t in labels]

            experimetnal_setup.model.train()
            loss_dict = experimetnal_setup.model(inputs, labels)

            loss = sum(loss for loss in loss_dict.values())
            loss_value = loss.item()

            if not math.isfinite(loss_value):
                logging.error("Loss is {}, stopping training".format(loss_value))
                continue

            metriclogger.update(time=time.time() - since_batch)
            metriclogger.update(loss=loss.item())
            metriclogger.update(loss_classifier=loss_dict['loss_classifier'].item())
            metriclogger.update(loss_box_reg=loss_dict['loss_box_reg'].item())
            metriclogger.update(loss_mask=loss_dict['loss_mask'].item())
            metriclogger.update(loss_objectness=loss_dict['loss_objectness'].item())
            metriclogger.update(loss_rpn_box_reg=loss_dict['loss_rpn_box_reg'].item())

            t.set_postfix(loss='{:05.5f}'.format(metriclogger.loss()),
                          cls='{:05.5f}'.format(metriclogger.loss_classifier()),
                          box='{:05.5f}'.format(metriclogger.loss_box_reg()),
                          mask='{:05.5f}'.format(metriclogger.loss_mask()),
                          rpn_obj='{:05.5f}'.format(metriclogger.loss_objectness()),
                          rpn_box='{:05.5f}'.format(metriclogger.loss_rpn_box_reg()))
            t.update()

    metrics = metriclogger.get_dictionary()
    logging.info("Validation performance: " + str(metriclogger))
    return metrics
