import attr
import yaml
from icecream import ic


@attr.s(frozen=True)
class MLConfig:
    # Number of classes
    N_CLASSES = attr.ib()
    # Number of input channels (e.g. RGB)
    IN_CHANNELS = attr.ib()
    # Patch size
    WINDOW_SIZE = attr.ib()
    # Name of the architecture
    NET_NAME = attr.ib()
    # Train on a subset of data: 1 for full train set, 0 for none of it
    SUB_TRAIN = attr.ib()

    # Train parameters
    EPOCHS = attr.ib()
    OPTIM_BASELR = attr.ib()
    OPTIM_STEPS = attr.ib()
    # Default epoch size is 10 000 samples
    EPOCH_SIZE = attr.ib()
    # Number of threads to use during training
    WORKERS = attr.ib()
    # Number of samples in a mini-batch per GPU
    BATCH_SIZE = attr.ib()
    # Weight the loss for class balancing
    WEIGHTED_LOSS = attr.ib()
    # Data augmentation (flip vertically and horizontally)
    TRANSFORMATION = attr.ib()
    # Keeps x data for validation. If the value is 1,
    # train and test on the full dataset (ie no validation set)
    test_size = attr.ib()
    # Only for interactivity: True enables distance transform to dilate annotations
    DISTANCE_TRANSFORM = attr.ib()

    # Test parameters
    # stride at test time
    STRIDE = attr.ib()
    # Number of threads to use during testing. Has to be
    # lower than at training since we now load full images.
    TEST_WORKERS = attr.ib()

    # Interactivity
    REVOLVER_WEIGHTED = attr.ib()

    SAVE_FOLDER = attr.ib()
    PATH_MODELS = attr.ib()

    # extension added to the saved files
    ext = attr.ib(default="")


def config_factory(config_file: str) -> MLConfig:
    """
    parse the config file and instanciate the config object
    """
    with open(config_file, "rb") as f:
        params = yaml.load(f)
        ic(params)

    return MLConfig(**params)
