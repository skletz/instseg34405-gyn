import sys
from utils.arguments import Arguments


class ArgumentParser:

    @classmethod
    def parse_args(cls, parser=None, add_default_args=True):
        """
        Parse command line arguments and prints help when no commands are given.
            - Environment variables in paths are replaced.
            - Relative paths are expanded to absolute paths.
              Either based on a given WORKING_DIR env variable or the current working directory.

        :return: (argparse.Namespace) object with attribute-accessible variables.
                    Example: The attribute s is accessibly by args.s
        """

        if add_default_args:
            parser = Arguments.init_arguments(parser)

        if len(sys.argv) == 1:
            parser.print_help()

        args = parser.parse_args()
        # replace environment variables
        args = Arguments.expand_paths(args)

        if add_default_args:
            # replace relative paths
            args = Arguments.setup_experiment_dir(args)

        return args


