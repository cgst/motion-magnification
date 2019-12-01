import os

import numpy as np
import torch
from torchvision import transforms, datasets
from torchvision.datasets.folder import default_loader


class MotionMag(datasets.VisionDataset):
    def __init__(self, root, image_loader=default_loader, transform=None):
        super().__init__(root, transform=transform)
        amp_path = os.path.join(self.root, "train_mf.txt")
        self.amp_f = torch.from_numpy(np.loadtxt(amp_path, dtype=np.float32))
        self.image_loader = image_loader

    def __len__(self):
        return len(self.amp_f)

    def __getitem__(self, index):
        fname = "%06d.png" % index
        paths = (os.path.join(self.root, "frameA", fname),
                 os.path.join(self.root, "frameB", fname),
                 os.path.join(self.root, "frameC", fname),
                 os.path.join(self.root, "amplified", fname))
        frame_a, frame_b, frame_perturbed, target = map(self._im_load, paths)
        return {'frame_a': frame_a,
                'frame_b': frame_b,
                'frame_perturbed': frame_perturbed,
                'amplification_f': self.amp_f[index],
                'frame_amplified': target}

    def _im_load(self, path):
        im = self.image_loader(path)
        if self.transform:
            im = self.transform(im)
        return im


def from_dir(root_dir):
    ds = MotionMag(root_dir, transform=transforms.ToTensor())
    return ds
