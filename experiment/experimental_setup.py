import os
import logging
import numpy
import random
import torch
from utils.config_parser import ConfigParser
from model.data.data_directory import DataDirectory
from model.mrcnn_resnet import MaskRCNNResNet
from model.data.lisloc_dataset import LisLocDataset, LisLocSubset
from utils.dataset import DatasetUtil
from utils.io import IOUtil
from utils.model import ModelUtil
from utils.batch import BatchUtil
from torch.utils.data import DataLoader
import numpy as np
from torch import device
from typing import List, Tuple, Dict


class ExperimentalSetup:
    __model_checkpoint_dir_name = "weights"
    __date_checkpoint_dir_name = "log_samples"
    __log_train_dir_name = "train"
    __log_val_dir_name = "val"

    data_dir_path = None
    model_dir_path = None
    model_checkpoint_dir_path = None
    model_resume_file_path = None
    log_train_dir_path = None
    log_val_dir_path = None

    device = None
    model = None
    dataset = None
    dataloaders = List[Dict[str, List]]
    optimizer = None
    scheduler = None
    params = None

    sample_split_indices: List[Tuple[np.array, np.array, np.array]]
    sample_split_names = List[str]
    sample_split_mapping = {'train': 0, 'val': 1, 'test': 2}

    seed = 0
    is_repeatable = True
    is_benchmark = False

    config_file = None

    restore_sample_splits = False
    restore_model_state = False
    restore_optimizer = False
    restore_scheduler = False

    def __init__(self):
        self.seed = 0
        self.is_repeatable = True
        self.is_benchmark = False
        self.dataloaders = []

    def init_device(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            logging.info("GPU")
        else:
            logging.info("CPU")

    def to_deterministic(self, seed=0, is_repeatable=True):
        self.seed = seed
        self.is_repeatable = is_repeatable
        torch.manual_seed(seed)
        numpy.random.seed(seed)
        random.seed(seed)  # Python

        # @Important: if it's enabled, the process will be much slower
        # but it is repeatable
        if torch.cuda.is_available() and is_repeatable:
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # gpu vars
            torch.backends.cudnn.deterministic = self.is_repeatable  # needed
            torch.backends.cudnn.benchmark = self.is_benchmark

    def init_parameters(self, config_file):
        self.set_config_file(config_file)

        self.config_file = config_file
        config = ConfigParser.parser_args(config_file)
        self.params = config

    def set_model_dir(self, model_dir):
        if not os.path.exists(model_dir):
            message = 'Model dir "{}" does not exist.'.format(model_dir)
            raise FileNotFoundError(str(message))

        self.model_dir_path = model_dir

    def set_model_resume_file(self, model_resume_file):
        if not os.path.exists(model_resume_file):
            message = 'Model resume file "{}" does not exist.'.format(model_resume_file)
            raise FileNotFoundError(str(message))

        self.model_resume_file_path = model_resume_file

    def set_model_checkpoint_dir(self, model_checkpoint_dir):
        parent_dir = os.path.dirname(model_checkpoint_dir)
        if not os.path.exists(parent_dir):
            message = 'Parent dir of model checkpoint dir "{}" does not exist.'.format(parent_dir)
            raise FileNotFoundError(str(message))
        else:
            IOUtil.create_dir(model_checkpoint_dir)

        self.model_checkpoint_dir_path = model_checkpoint_dir

    def set_data_dir(self, data_dir):

        if not os.path.exists(data_dir):
            message = 'Data dir "{}" does not exist.'.format(data_dir)
            raise FileNotFoundError(str(message))

        self.data_dir_path = data_dir

    def set_config_file(self, config_file):
        if not os.path.exists(config_file):
            message = 'Config file "{}" does not exist.'.format(config_file)
            raise FileNotFoundError(str(message))

        self.config_file = config_file

    def set_augment_train(self, augment):
        self.dataloaders[0]['train'].dataset.augment = augment

    def init_model(self, model_root_dir, model_resume_file):
        self.set_model_dir(model_root_dir)
        self.set_model_checkpoint_dir(os.path.join(model_root_dir, self.__model_checkpoint_dir_name))

        if model_resume_file:
            self.set_model_resume_file(model_resume_file)

        name = self.params.model.name
        num_classes = self.params.model.output_size
        image_size = self.params.model.input_size
        box_detections_per_img = self.params.model.box_detections_per_img

        logging.info("Loading model %s" % name)
        logging.info("Loading Image size %s" % (image_size,))
        logging.info("Loading Classes %s" % num_classes)
        logging.info("Loading box detections per image %s" % box_detections_per_img)

        pretrained = self.params.model.pretrained
        if pretrained:
            logging.debug("Init the model with a pre-trained model ...")

        if name == "torchvision.mrcnn":
            self.model = MaskRCNNResNet(num_classes=num_classes,
                                        pre_trained=pretrained,
                                        min_size=image_size[1],
                                        max_size=image_size[0],
                                        box_detections_per_img=box_detections_per_img,
                                        rpn_pre_nms_top_n_train=2000,
                                        rpn_post_nms_top_n_train=2000,
                                        rpn_pre_nms_top_n_test=2000,
                                        rpn_post_nms_top_n_test=2000)
        else:
            logging.error("Model '%s' is not implemented." % name)

    def init_dataset(self, data_root_dir_path, restore):
        """

        :param data_dir:
        :param samples:
        :return:
        """
        self.set_data_dir(data_root_dir_path)
        name = self.params.dataset.name
        image_size = self.params.model.input_size
        logging.info("Loading the dataset %s; image size %s;" % (name, (image_size,)))

        self.data_dir_path = DataDirectory(data_root_dir_path)
        annotation_file = os.path.join(self.data_dir_path.get_json_annotation_file())

        if not os.path.exists(annotation_file):
            message = 'Annotation file "{}" does not exist.'.format(annotation_file)
            raise FileNotFoundError(str(message))

        if name == "lisloc.detseg":
            self.dataset = LisLocDataset(self.data_dir_path, target_task="detseg")
        elif name == "lisloc.det":
            self.dataset = LisLocDataset(self.data_dir_path, target_task="det")
        elif name == "lisloc.seg":
            self.dataset = LisLocDataset(self.data_dir_path, target_task="seg")
        else:
            logging.error("Dataset '%s' is not implemented." % name)

        if restore:
            logging.info("Restore dataset")
            self.dataset = DatasetUtil.restore_dataset(self.model_dir_path)
        else:
            DatasetUtil.save_dataset(self.dataset, self.model_dir_path)

    def init_data_splits(self, split_names, train_size, restore):
        logging.debug("init_data_splits(): %s %s %s" % (split_names, train_size, restore))
        self.sample_split_names = split_names
        self.sample_split_indices = DatasetUtil.train_val_test_split_multilabel(
            self.dataset.filenames, self.dataset.labels, train_size=train_size)

        if restore:
            logging.debug("Restore the the sample splits of the dataset ...")
            self.sample_split_indices = DatasetUtil.restore_samples_split_indices(restore_file=restore)
        else:
            DatasetUtil.save_samples_split_indices(
                self.sample_split_indices, split_names, self.model_dir_path)

        logging.debug("init_data_splits():")

    def init_dataloaders(self, mode: str):

        batch_size = self.params.dataloader.batch_size
        num_workers = self.params.dataloader.num_workers
        pin_memory = self.params.dataloader.cuda
        augment = self.params.dataloader.augment

        logging.info("Dataloader's batch size: %d" % (batch_size))

        one_step_dataloaders = {}
        for sample in self.sample_split_indices:
            for _, split_name in enumerate(self.sample_split_names):

                index = self.sample_split_mapping[split_name]

                if mode == "train" and split_name == "train":
                    shuffle = True
                    augment = True
                else:
                    shuffle = False
                    augment = False

                data = LisLocSubset(
                    dataset=self.dataset,
                    indices=sample[index],
                    activate_augmentation=augment)

                one_step_dataloaders[split_name] = DataLoader(dataset=data,
                                                              batch_size=batch_size,
                                                              shuffle=shuffle,
                                                              sampler=None,
                                                              batch_sampler=None,
                                                              num_workers=num_workers,
                                                              pin_memory=pin_memory,
                                                              drop_last=False,
                                                              collate_fn=BatchUtil.collate_fn)

            self.dataloaders.append(one_step_dataloaders)

    def init_optimizer(self):
        type = self.params.optimizer.type.lower()
        logging.info("Loading the optimizer %s " % type)

        # self.params_to_update = model.params_to_update
        if type == "sgd":
            logging.info("LR: %.8f; Weight decay %.8f; Momentum %.8f" %
                         (self.params.optimizer.base_lr, self.params.optimizer.weight_decay,
                          self.params.optimizer.momentum))

            self.optimizer = torch.optim.SGD(self.model.params_to_update,  # self.params_to_update, #model.parameters(),
                                             lr=self.params.optimizer.base_lr,
                                             momentum=self.params.optimizer.momentum,
                                             weight_decay=self.params.optimizer.weight_decay)
        elif type == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                              lr=self.params.optimizer.base_lr,
                                              eps=self.params.optimizer.eps)
        else:
            logging.error("Type '%s' is not implemented." % type)

    def init_scheduler(self):
        policy = self.params.optimizer.lr_policy
        logging.info("Loading the scheduler %s " % policy)

        if policy == "step":
            logging.info("Step_size: %d; Gamma %.8f" %
                         (self.params.optimizer.stepsize, self.params.optimizer.gamma))
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.params.optimizer.stepsize,
                                                             gamma=self.params.optimizer.gamma)
        elif policy == "cyclic":

            logging.info("Lower %.8f; Upper %.8f; Step size: %d" %
                         (self.params.optimizer.lower, self.params.optimizer.upper, self.params.optimizer.step_up))

            cyclic_mom = True
            if self.params.optimizer.momentum == 0.0:
                cyclic_mom = False
                lower_momentum = 0.0
                upper_momentum = 0.0

            logging.info("Cyclic momentum %s; lower momentum: %.3f; upper momentum %.3f; " % (
                cyclic_mom, lower_momentum, upper_momentum))

            self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer,
                                                               base_lr=self.params.optimizer.lower,
                                                               max_lr=self.params.optimizer.upper,
                                                               step_size_up=self.params.optimizer.step_up,
                                                               cycle_momentum=cyclic_mom,
                                                               base_momentum=lower_momentum,
                                                               max_momentum=upper_momentum)
        elif policy == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                        mode=self.params.optimizer.monitor_mode,
                                                                        factor=self.params.optimizer.factor,
                                                                        patience=self.params.optimizer.patience)

        elif policy == "None":
            # @Work-Around: if no scheduler should be used
            # then step size is set to max epochs,
            # so no change in learning rate it keeps constant during training
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                             step_size=self.params.optimizer.epochs,
                                                             gamma=1.0)
        else:
            logging.error("Policy '%s' is not implemented." % policy)

    def init_tensorboard_logs(self):
        self.log_train_dir_path = IOUtil.create_dir(
            IOUtil.join_dir(self.model_dir_path, self.__log_train_dir_name))

        self.log_val_dir_path = IOUtil.create_dir(
            IOUtil.join_dir(self.model_dir_path, self.__log_val_dir_name))

    def get_train_dataloader(self):
        return self.dataloaders[0]['train']

    def get_val_dataloader(self):
        return self.dataloaders[0]['val']

    def get_test_data(self):
        return self.dataloaders[0]['test']

    def get_dataset(self, name):
        return self.dataloaders[0][name].dataset

    def get_dataloader(self, name):
        return self.dataloaders[0][name]

    def restore_model(self):
        if not self.model_resume_file_path:
            return 0

        t, epoch = ModelUtil.load_state(self.model, self.model_resume_file_path, self.optimizer, self.scheduler)

        # @Work-Around: https://github.com/pytorch/pytorch/issues/2830
        # resume training problem is that the optimizer state is on CPU
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        return epoch
