from copy import deepcopy
import numpy as np
from torch.utils.data import Dataset
import os
import h5py

try:
    from inferno.io.core import Zip, Concatenate
    from inferno.io.transform import Compose, Transform
    from inferno.io.transform.generic import AsTorchBatch
    from inferno.io.transform.volume import RandomFlip3D, VolumeAsymmetricCrop
    from inferno.io.transform.image import RandomRotate, ElasticTransform
    from torch.utils.data.dataloader import default_collate
    from inferno.io.volumetric.volume import HDF5VolumeLoader
    from inferno.io.volumetric.volumetric_utils import slidingwindowslices
except ImportError:
    raise ImportError("CremiDataset requires inferno")

try:
    from neurofire.datasets.loader import RawVolume, SegmentationVolume
    from neurofire.transform.artifact_source import RejectNonZeroThreshold
    from neurofire.transform.volume import RandomSlide
except ImportError:
    raise ImportError("CremiDataset requires neurofire")

from ..utils.various import yaml2dict
from ..transform.volume import DownSampleAndCropTensorsInBatch, ReplicateTensorsInBatch
from ..transform.affinities import affinity_config_to_transform, Segmentation2AffinitiesDynamicOffsets


'''
A simple Pytorch Dataset Class for training a network to predict seeds for small segments.
Input is a path to a h5py file containing an 'input' dataset of shape [2, z, x, y] and a 'seeds' dataset of shape [z, x, y]
'''


class CremiSeedDataset(Dataset):
    def __init__(self, path):
        assert os.path.exists(path)
        self.path = path
        f = h5py.File(self.path, 'r')
        assert 'input' in f.keys()
        assert 'seeds' in f.keys()
        self.slice_iter = list(slidingwindowslices(shape=f['seeds'].shape, window_size=(8, 320, 320), strides=(2,100,100)))
        f.close()
        self.transforms = self.data_transforms()

    def __len__(self):
        return len(self.slice_iter)

    def __getitem__(self, item):
        slice = self.slice_iter[item]
        f = h5py.File(self.path, 'r')
        input = np.stack((f['input'][0][slice], f['input'][1][slice]), axis=0)
        seeds = f['seeds'][slice][np.newaxis,...]
        f.close()
        return input, seeds

    def data_transforms(self):
        transforms = Compose()
        transforms.add(RandomFlip3D())
        transforms.add(RandomRotate())
        return transforms
