import io
import argparse
import os
import logging
import time
from typing import Dict, Optional

from utils.argument_parser import ArgumentParser
from experiment.experimental_setup import ExperimentalSetup
from utils.logging import LoggingUtil
from utils.time import TimeUtil
from tensorboardX import SummaryWriter
from tqdm import tqdm
from utils.metric_logger import MetricLogger
from utils.model import ModelUtil
import math

from experiment import validate
from experiment import evaluate
import json


def train(experimetnal_setup: ExperimentalSetup, stdout: Optional[io.TextIOWrapper] = None) -> Dict:
    """

    :param experimetnal_setup:
    :param stdout:
    :return:
    """
    metriclogger = MetricLogger()
    dataloader = experimetnal_setup.get_train_dataloader()

    experimetnal_setup.model.train()

    with tqdm(desc="Train-Batch", total=len(dataloader), file=stdout, dynamic_ncols=True) as t:

        for i, data in enumerate(dataloader, 0):
            since_batch = time.time()

            (images, labels), files = data

            images = list(image.to(experimetnal_setup.device) for image in images)
            labels = [{k: v.to(experimetnal_setup.device) for k, v in t.items()} for t in labels]

            loss_dict = experimetnal_setup.model(images, labels)

            loss = sum(loss for loss in loss_dict.values())
            loss_value = loss.item()

            if not math.isfinite(loss_value):
                logging.error("Loss is {}, stopping training".format(loss_value))
                continue

            experimetnal_setup.optimizer.zero_grad()
            loss.backward()
            experimetnal_setup.optimizer.step()

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
    logging.info("Train performance: " + str(metriclogger))
    return metrics


def log_summary(writer, metrics, epoch=0, tab="trainval"):
    for metric in metrics:
        if not (metric == "time" or metric == "epoch"):
            writer.add_scalar(tab + '/{}'.format(metric.replace(":", "_")), metrics[metric], epoch)


def log_coco_eval_metric(metrics):
    if 'segm_10_AP_0.50:0.95_all' in metrics:
        logging.debug(
            "Performance (Segmentation) COCO AP50:95 {:.5f}".format(metrics['segm_10_AP_0.50:0.95_all']))
    if 'bbox_10_AP_0.50:0.95_all' in metrics:
        logging.debug(
            "Performance (Bounding Box) COCO AP50:95 {:.5f}".format(metrics['bbox_10_AP_0.50:0.95_all']))


def train_and_validate(experimetnal_setup: ExperimentalSetup, stdout: Optional[io.TextIOWrapper] = None) -> None:
    """

    :param experimetnal_setup:
    :param stdout:
    :return:
    """

    start_epoch = experimetnal_setup.restore_model()

    logging.info("Load model to %s" % experimetnal_setup.device)
    experimetnal_setup.model.to(experimetnal_setup.device)

    experimetnal_setup.init_tensorboard_logs()

    writer_train = SummaryWriter(log_dir=experimetnal_setup.log_train_dir_path, flush_secs=15, filename_suffix="_train")
    writer_val = SummaryWriter(log_dir=experimetnal_setup.log_val_dir_path, flush_secs=15, filename_suffix="_val")

    remaining_epochs = experimetnal_setup.params.optimizer.epochs - start_epoch

    with tqdm(desc="Epoch", total=remaining_epochs, file=stdout, dynamic_ncols=True) as tepoch:

        # best_val_acc = 0.0
        # best_val_acc_epoch = 0
        best_val_loss = float("inf")
        best_val_loss_epoch = 0

        for epoch in range(start_epoch + 1, experimetnal_setup.params.optimizer.epochs + 1):
            since_epoch = time.time()

            logging.info("Processing epoch %04d ..." % epoch)

            lr = experimetnal_setup.optimizer.param_groups[0]['lr']
            logging.warning("Learning rate: %f" % lr)
            writer_train.add_scalar('trainval/lr', lr, epoch)

            since_training = time.time()

            train_metrics = train(experimetnal_setup, stdout=stdout)

            train_metrics['epoch'] = epoch
            train_metrics['lr'] = lr

            time_training_elapsed = time.time() - since_training
            logging.debug('Training complete in {:.0f}m {:.4f}s'.format(*TimeUtil.get_m_s(time_training_elapsed)))

            logging.debug("Evaluate train data, switch off data augmentation")
            dataset_augmentation_setting = experimetnal_setup.get_train_dataloader().dataset.augment
            experimetnal_setup.get_train_dataloader().dataset.augment = False

            results = evaluate.predict(experimetnal_setup,
                                       experimetnal_setup.get_train_dataloader(),
                                       stdout=stdout)

            coco_metric = evaluate.evaluate_with_coco_metric(results,
                                                             experimetnal_setup.get_train_dataloader().dataset,
                                                             ["bbox", "segm"], [10])
            experimetnal_setup.get_train_dataloader().dataset.augment = dataset_augmentation_setting
            logging.debug("Evaluate train data finished, reset data augmentation setting")

            log_summary(writer_train, train_metrics, epoch, tab="trainval")
            log_summary(writer_train, coco_metric, epoch, tab="trainval-coco")
            log_coco_eval_metric(coco_metric)
            train_metrics.update(coco_metric)

            since_validation = time.time()
            val_metrics = validate.validate(experimetnal_setup, stdout=stdout)

            logging.debug(
                'Validation complete in {:.0f}m {:.4f}s'.format(*TimeUtil.get_m_s(time.time() - since_validation)))

            val_metrics['epoch'] = epoch
            train_metrics['epoch'] = epoch

            results = evaluate.predict(experimetnal_setup,
                                       experimetnal_setup.get_val_dataloader(),
                                       stdout=stdout)

            logging.debug("Evaluate validation data")
            coco_metric = evaluate.evaluate_with_coco_metric(results,
                                                             experimetnal_setup.get_val_dataloader().dataset,
                                                             ["bbox", "segm"], [10])

            log_summary(writer_val, val_metrics, epoch, tab="trainval")
            log_summary(writer_val, coco_metric, epoch, tab="trainval-coco")
            log_coco_eval_metric(coco_metric)
            val_metrics.update(coco_metric)

            val_loss = val_metrics['loss']
            is_best_loss = val_loss <= best_val_loss

            if is_best_loss:
                best_val_loss = val_loss
                best_val_loss_epoch = epoch
                logging.debug('***Epoch %04d***: new best %s: %0.5f' % (epoch, 'loss', best_val_loss))

            ModelUtil.save_checkpoint(state={
                'epoch': epoch,
                'model_state_dict': experimetnal_setup.model.state_dict(),
                'optimizer_state_dict': experimetnal_setup.optimizer.state_dict(),
                'scheduler_state_dict': experimetnal_setup.scheduler.state_dict()
            },
                is_best_loss=is_best_loss,
                checkpoint_dir=experimetnal_setup.model_checkpoint_dir_path)

            if experimetnal_setup.scheduler is not None:
                experimetnal_setup.scheduler.step()

            tepoch.update()

            logging.debug('Epoch complete in {:.0f}m {:.4f}s'.format(*TimeUtil.get_m_s(time.time() - since_epoch)))

            store_json(experimetnal_setup.model_checkpoint_dir_path, "log_metrics", "metrics_train", epoch,
                       train_metrics)
            store_json(experimetnal_setup.model_checkpoint_dir_path, "log_metrics", "metrics_val", epoch,
                       val_metrics)

        writer_train.close()
        writer_val.close()

        logging.info("Best loss obtained at epoch %04d: %.5f" % (best_val_loss_epoch, best_val_loss))


def store_json(dir, subdir, prefix, epoch, json):
    filename = prefix + "_{:04d}".format(epoch) + ".json"

    dir = os.path.join(dir, subdir)
    if not os.path.exists(dir):
        os.makedirs(dir)

    filepath = os.path.join(dir, filename)
    save_dict_to_json(json, filepath)


def save_dict_to_json(_dict, json_path):
    """
    Saves dict in json file
    :param _dict: (dict) of values
    :param json_path: (string) path to json file
    """

    with open(json_path, 'w') as f:
        json.dump(_dict, f, indent=4)


def main(args: argparse.Namespace) -> None:
    """
    Train.
    :param args: (argparse.Namespace) attribute accessible command line arguments
    """

    LoggingUtil.setup_logger(args.log_dir, "train.log",
                             level_terminal=logging.DEBUG,
                             reset=args.reset_log_file)
    logging.debug("TRAIN")
    logging.info("Training started ...")

    logging.debug("Setting up experiment ...")
    experimental_setup = ExperimentalSetup()

    logging.debug("Init device")
    experimental_setup.init_device()

    logging.debug("Setting random seed: %d with repeatable option=%s" % (args.seed, args.is_repeatable))
    experimental_setup.to_deterministic(args.seed, args.is_repeatable)

    logging.debug("Init parameters ...")
    experimental_setup.init_parameters(args.conf_file)

    logging.debug("Init model ...")
    experimental_setup.init_model(args.model_dir, args.resume)

    logging.debug("Init dataset ...")
    experimental_setup.init_dataset(args.data_dir, args.restore)

    logging.debug("Init optimizer ...")
    experimental_setup.init_optimizer()

    logging.debug("Init scheduler ...")
    experimental_setup.init_scheduler()

    logging.debug("Init data splits ...")
    experimental_setup.init_data_splits(args.split_names, args.train_size, args.restore)

    logging.debug("Init dataloaders ...")
    experimental_setup.init_dataloaders(mode="train")

    since = time.time()
    with LoggingUtil.std_out_err_redirect_tqdm() as orig_stdout:
        train_and_validate(experimental_setup, stdout=orig_stdout)
    time_elapsed = time.time() - since
    logging.debug('Training complete in {:.0f}m {:.0f}s'.format(*TimeUtil.get_m_s(time_elapsed)))

    logging.info("Training finished.")


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == '__main__':
    """
    Script to train neural network models.
    """

    from dotenv import load_dotenv

    load_dotenv(dotenv_path='.envs/.train')

    parser = argparse.ArgumentParser(
        description='Training.')

    # input example --split_names train val test
    parser.add_argument('--split_names', required=False,
                        metavar="train, val, test", default=["train", "val", "test"], nargs="+", type=str,
                        help='Optional, (list) the names of the splits to evaluate.')

    parser.add_argument('--train_size', required=False,
                        metavar="0.7", default=0.7, type=float,
                        help='Optional, (float) size of the train split.')

    parsed_args = ArgumentParser.parse_args(parser)
    main(parsed_args)
