import argparse
import os


class Arguments:

    @staticmethod
    def init_arguments(parser=None):
        """
        Initialize command line arguments.
        :return: (argparse.ArgumentParser)
        """

        if parser is None:
            parser = argparse.ArgumentParser(
                description='Additional Command Line Arguments.')

        parser.add_argument('--data_dir',
                            required=False,
                            metavar="experiments/data",
                            default="experiments/data",
                            help="Directory containing the dataset")

        parser.add_argument('--model_dir',
                            required=False,
                            metavar="experiments/model",
                            default="experiments/model",
                            help="Directory containing the .conf file")

        parser.add_argument('--resume',
                            required=False,
                            metavar="None",
                            default=None,
                            type=str,
                            help='Optional, name of the checkpoint file in --model_dir containing weights to reload.')

        parser.add_argument('--restore',
                            required=False,
                            metavar="None",
                            default=None,
                            type=str,
                            help='Optional, name of the data file in --model_dir containing samples to reload.')

        parser.add_argument('--seed', required=False,
                            metavar="0",
                            default=0,
                            type=int,
                            help='Integer to init the random number generator for reproducibility.')

        parser.add_argument('--is_repeatable', required=False,
                            metavar="True", default=True, type=lambda x: (str(x).lower() == 'true'),
                            help='Optional, .')

        parser.add_argument('--reset_log_file', required=False,
                            metavar="True", default=True, type=lambda x: (str(x).lower() == 'true'),
                            help='Optional, .')

        return parser

    @staticmethod
    def expand_paths(args: argparse.Namespace) -> argparse.Namespace:
        """
        
        :param args: argparse.Namespace
        :return: argparse.Namespace
        """

        args_copy = argparse.Namespace(**vars(args))

        for arg in vars(args_copy):
            value = getattr(args_copy, arg)
            if value is not None:
                if not isinstance(value, list):
                    if not isinstance(value, bool):
                        if not isinstance(value, int):
                            if not isinstance(value, float):
                                setattr(args_copy, arg, os.path.expandvars(value))

        return args_copy

    @staticmethod
    def setup_experiment_dir(args):
        """

        :param args: argparse.Namespace
        :return: argparse.Namespace
        """
        args_copy = argparse.Namespace(**vars(args))
        args_copy.log_dir = args_copy.model_dir
        args_copy.conf_file = os.path.join(args_copy.model_dir, "params.conf")
        return args_copy
