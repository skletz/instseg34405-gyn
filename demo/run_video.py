""" (Version 2.0).

    Implementation is based on Mask R-CNN and ResNet.
    ******************************************************
          _    _      ()_()
         | |  | |    |(o o)
      ___| | _| | ooO--`o'--Ooo
     / __| |/ / |/ _ \ __|_  /
     \__ \   <| |  __/ |_ / /
     |___/_|\_\_|\___|\__/___|
    ******************************************************
    @author skletz
    @version 2.0 15/09/19
    @version 1.0 23/08/19
    @filename run_video.py
    @description:
    @input:
    @output:
"""

import logging
import os
import sys
import argparse

cwd = os.getcwd()
print("ADD module: {}".format(cwd))
sys.path.append(cwd)

import numpy as np
import random
import torch
from tqdm import tqdm
import cv2
from demo.predictor import InstSegPredictorDemo


def args_validity_check(args):
    if not os.path.exists(args.input):
        message = "Input directory '{}' does not exist.".format(args.input)
        raise FileNotFoundError(str(message))

    if not os.path.exists(args.output):
        message = 'Output directory "{}" does not exist.'.format(args.output)
        logging.warning(message)
        logging.info("Directory will be created.")

    if not os.path.exists(args.det_model):
        message = 'Detection model "{}" does not exist.'.format(args.det_model)
        raise FileNotFoundError(str(message))


def run_on_video(demo,
                 vdo_input_path=None,
                 vdo_output_path=None,
                 image_frame_output_dir=None, enable_tracking=False):
    vdo_capture = cv2.VideoCapture(vdo_input_path)
    width = int(vdo_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vdo_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vdo_capture.get(cv2.CAP_PROP_FPS)
    frame_cnt = int(vdo_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    vdo_writer = cv2.VideoWriter(vdo_output_path + ".mp4",
                                 cv2.VideoWriter_fourcc(*'MP4V'),
                                 fps, (width, height))

    demo.set_image_dimension(width, height)

    success = True
    frame_counter = 0

    with tqdm(total=frame_cnt) as pbar:
        while success:
            # Read next image
            success, bgr_image = vdo_capture.read()

            frame_counter = frame_counter + 1
            if success:
                # OpenCV returns images as BGR, convert to RGB
                rgb_image = bgr_image[..., ::-1].copy()

                prediction, nr_objects = demo.predict_instruments(rgb_image)

                instruments = demo.prediction_to_objects(prediction)

                result_image = demo.apply_instruments(bgr_image, instruments)

                bgr_result_image = result_image[..., ::-1].copy()

                vdo_writer.write(bgr_result_image)

                output_fname = "{:06d}.jpg".format(frame_counter)
                cv2.imwrite(os.path.join(image_frame_output_dir, output_fname), bgr_result_image)

            pbar.set_postfix(timecode=frame_counter)
            pbar.update()

    vdo_writer.release()
    vdo_capture.release()


def main(args):
    logging.info("Start ...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    demo = InstSegPredictorDemo(device, det_model_uri=args.det_model, )

    append = "_result"
    if args.track:
        append += "_tracklets"

    vdo_fname = os.path.basename(args.input)
    vdo_output_path = os.path.join(args.output, vdo_fname + append)
    image_frame_output_dir = vdo_output_path
    if not os.path.exists(vdo_output_path):
        os.makedirs(image_frame_output_dir)

    run_on_video(demo, args.input, vdo_output_path, image_frame_output_dir, args.track)

    logging.info("Finished.")


def setup():
    """

    :return:
    """

    logger = logging.getLogger()
    format_stream = logging.Formatter("%(asctime)s | %(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(format_stream)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    torch.manual_seed(88)
    np.random.seed(88)
    random.seed(88)  # Python


def parse_args(parser):
    """
    Parse command line arguments and prints help when no commands are given.
        - Environment variables in paths are replaced.

    :return: (argparse.Namespace) object with attribute-accessible variables.
                Example: The attribute s is accessibly by args.s
    """

    if len(sys.argv) == 1:
        parser.print_help()

    args = parser.parse_args()
    args_copy = argparse.Namespace(**vars(args))
    # replace environment variables
    for arg in vars(args_copy):
        value = getattr(args_copy, arg)
        if value is not None:
            if not isinstance(value, list):
                try:
                    setattr(args_copy, arg, os.path.expandvars(value))
                except TypeError:
                    print("Type error in expandvars for variable:", arg)

    return args_copy


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':

    from dotenv import load_dotenv

    load_dotenv(dotenv_path='./envs/.run_video')

    parser = argparse.ArgumentParser(
        description='Run demo on video.')

    parser.add_argument('--input', required=True,
                        metavar="/path/to/input/dir", default=None, type=str,
                        help='Path to the video.')

    parser.add_argument('--output', required=True,
                        metavar="/path/to/output/dir", default=None, type=str,
                        help='Directory to store result.')

    parser.add_argument('--model', required=False,
                        metavar="/path/to/output/dir",
                        default=os.path.join(os.path.split(os.getcwd())[0], "test", "model",
                                             "instseg34405-gyn.pth.tar"),
                        type=str,
                        help='Path to the model.')

    parsed_args = parse_args(parser)

    setup()
    log_message = ""
    for idx, arg in enumerate(vars(parsed_args)):
        suffix = " \ \n" if idx < len(vars(parsed_args)) - 1 else ""
        log_message += ("--{}={}{}".format(arg, getattr(parsed_args, arg), suffix))
    logging.info("python -u run_video.py \ \n{}".format(log_message))

    args_validity_check(parsed_args)
    main(parsed_args)
