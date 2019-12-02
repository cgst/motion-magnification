import os
import time

import clize
import numpy as np
import torch
from torch.functional import F
from torchsummary import summary
from tqdm import trange

from deepmag import dataset
from deepmag.model import MagNet
from deepmag.train import train_epoch


def train(dataset_root_dir, model_output_dir, *, num_epochs=3, batch_size=4,
          device="cuda:0", regularization_weight=0.1, learning_rate=0.0001):
    device = torch.device(device)
    ds = dataset.from_dir(dataset_root_dir)
    model = MagNet().to(device)
    with trange(num_epochs, desc="Epoch") as pbar:
        for epoch_idx in pbar:
            train_epoch(model, ds, device, learning_rate=learning_rate,
                        batch_size=batch_size, reg_weight=regularization_weight)
            save_path = os.path.join(model_output_dir,
                                     '%s-b%s-r%s-lr%s-%02d.pt' % (time.strftime("%Y%m%d"), batch_size,
                                                                  regularization_weight, learning_rate,
                                                                  epoch_idx))
            torch.save(model, save_path)
            pbar.write("Saved snapshot to %s" % save_path)


if __name__ == "__main__":
    clize.run((train,))