import os

import numpy as np
import torch
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader


class DeepMagDataset(VisionDataset):
    def __init__(self, root, image_loader=default_loader):
        super().__init__(root, transforms=None)
        amp_path = os.path.join(self.root, "train_mf.txt")
        self.amp_f = torch.from_numpy(np.loadtxt(amp_path))
        self.image_loader = image_loader

    def _load_image(self, path):
        im = self.image_loader(path)
        return np.array(im) / 127.5 - 1.0

    def __len__(self):
        return len(self.amp)

    def __getitem__(self, index):
        fname = "%06d.png" % index
        paths = (os.path.join(self.root, "frameA", fname),
                 os.path.join(self.root, "frameB", fname),
                 os.path.join(self.root, "frameC", fname),
                 os.path.join(self.root, "amplified", fname))
        frame_a, frame_b, frame_perturbed, target = map(self._load_image, paths)
        return (frame_a, frame_b, frame_perturbed, self.amp_f[index]), target
