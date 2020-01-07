import math
import torch
import torch.nn as nn
import torch.nn.init as init
import os
import torchvision
import logging
from functools import reduce
import shutil

class ModelUtil():

    @staticmethod
    def reset_parameters(net):
        """

        :param net:
        :return:
        """
        for m in net.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=math.sqrt(5))
                fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(m.bias, -bound, bound)

    @staticmethod
    def load_state(model, checkpoint, optimizer=None, scheduler=None):
        """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
        optimizer assuming it is present in checkpoint.
        Args:
            checkpoint: (string) filename which needs to be loaded
            model: (torch.nn.Module) model for which the parameters are loaded
            optimizer: (torch.optim) optional: resume optimizer from checkpoint
        """
        if not os.path.exists(checkpoint):
            raise ("File doesn't exist {}".format(checkpoint))
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        epoch = checkpoint['epoch']
        return checkpoint, epoch

    @staticmethod
    def load_checkpoint(checkpoint):
        """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
        optimizer assuming it is present in checkpoint.
        Args:
            checkpoint: (string) filename which needs to be loaded
        """
        if not os.path.exists(checkpoint):
            raise ("File doesn't exist {}".format(checkpoint))
        checkpoint = torch.load(checkpoint)

        return checkpoint

    @staticmethod
    def pytorch_count_params(model):
        "count number trainable parameters in a pytorch model"
        total_params = sum(reduce(lambda a, b: a * b, x.size()) for x in model.parameters() if x.requires_grad)
        return total_params

    @staticmethod
    def save_checkpoint(checkpoint_dir, state, is_best_loss=False, is_best_acc=None, save_best_only=False):
        """
         Saves model and training parameters at checkpoint 'last.pth.tar'. If is_best==True,
         also saves checkpoint 'best.pth.tar'. If filename and suffix is provided additionally
         it is saved at '%filename%_%suffix%.pth.tar'. If no filename is provided
         it is saved at 'model_%suffix%.pth.tar'.

         Code modified from function save_checkpoint obtained from
         https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/vision/utils.py

         :param state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
         """

        checkpoint_dir = os.path.join(checkpoint_dir, "log_weights")

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # we store at least the last and the best checkpoint file
        filepath = os.path.join(checkpoint_dir, 'last.pth.tar')
        torch.save(state, filepath)

        if is_best_loss:
            shutil.copyfile(filepath, os.path.join(checkpoint_dir, 'best_loss.pth.tar'))

        if is_best_acc is not None and is_best_acc:
            shutil.copyfile(filepath, os.path.join(checkpoint_dir, 'best_accuracy.pth.tar'))

        if not save_best_only:
            filename = "model"

            filename = filename + "_{:04d}".format(state['epoch']) + ".pth.tar"
            shutil.copyfile(filepath, os.path.join(checkpoint_dir, filename))

    @staticmethod
    def decode_epoch_from_file(filename):
        basename = os.path.basename(filename)
        basename = os.path.splitext(basename)[0]
        basename = os.path.splitext(basename)[0]
        last_character = (basename[-4:])
        try:
            epoch_nr = int(last_character)
        except:
            epoch_nr = -1

        return epoch_nr