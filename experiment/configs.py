class ModelConfig:
    name = "torchvision.mrcnn"
    backbone = "resnet50"
    input_size = (540, 360)
    output_size = 7
    pretrained = True
    min_size = 360
    max_size = 540
    box_detections_per_img = 10


class DatasetConfig:
    name = "lisloc.detseg"


class DataloaderConfig:
    batch_size = 2
    num_workers = 4
    cuda = True
    augment = True


class OptimizerConfig:
    epochs = 50
    type = "sgd"
    momentum = 0.9
    base_lr = 0.001
    lr_policy = "step"
    gamma = 0.50
    stepsize = 25
    weight_decay = 0.0001
