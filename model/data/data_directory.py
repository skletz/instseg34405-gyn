import os


class DataDirectory(object):
    def __init__(self, root_dir="/"):
        self.annotation_dir = "annotations"
        self.image_dir = "jpegimages"
        self.mask_dir = "masks"
        self.imagesets_dir = "imagesets"
        self.evaluation_dir = "evaluations"
        self.stats_dir = "statistics"
        self.viz_dir = "visualization"
        self.tmp_dir = "tmp"
        self.experiment_dir = "experiments"
        self.log_dir = "logs"
        self.dirs = {
            'annotations': self.annotation_dir,
            'images': self.image_dir,
            'sets': self.imagesets_dir,
            'evaluations': self.evaluation_dir,
            'experiments': self.experiment_dir,
            'statistics': self.stats_dir,
            'masks': self.mask_dir,
            'visualization': self.viz_dir,
            'tmp': self.tmp_dir,
            'logs': self.log_dir,
        }

        self.root_dir = root_dir

    def create_structure(self):
        self.generate_dir_structure(self.root_dir, self.dirs)

    def get_annotation_dir(self):
        return os.path.join(self.root_dir, self.annotation_dir)

    def get_image_dir(self):
        return os.path.join(self.root_dir, self.image_dir)

    def getMaskDir(self):
        return os.path.join(self.root_dir, self.mask_dir)

    def getSetDir(self):
        return os.path.join(self.root_dir, self.imagesets_dir)

    def getEvalDir(self):
        return os.path.join(self.root_dir, self.evaluation_dir)

    def getExpDir(self):
        return os.path.join(self.root_dir, self.experiment_dir)

    def getStatDir(self):
        return os.path.join(self.root_dir, self.stats_dir)

    def getVizDir(self):
        return os.path.join(self.root_dir, self.viz_dir)

    def getTmpDir(self):
        return os.path.join(self.root_dir, self.tmp_dir)

    def get_json_annotation_file(self):
        file = os.path.basename(self.root_dir) + ".json"
        return os.path.join(self.root_dir, self.annotation_dir, file)

    def getDatasetName(self):
        return os.path.basename(self.root_dir)

    def getLogDir(self):
        return os.path.join(self.root_dir, self.log_dir)

    def get_root_dir(self):
        return self.root_dir

    @classmethod
    def generate_dir_structure(cls, output_dir, data_dir_dict):
        for key, val in data_dir_dict.items():
            dir_path = os.path.join(output_dir, val)
            os.makedirs(dir_path, exist_ok=True)
