import logging
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
from moviepy.editor import VideoFileClip
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image


def train(dataset_root_dir, model_output_dir, *, num_epochs=3, batch_size=4,
          device="cuda:0", regularization_weight=0.1, learning_rate=0.0001,
          skip_epochs=0, load_model_path=None):
    device = torch.device(device)
    ds = dataset.from_dir(dataset_root_dir)
    if load_model_path:
        model = torch.load(load_model_path).to(device)
        logging.info("Loaded model from %s", load_model_path)
    else:
        model = MagNet().to(device)
    with trange(skip_epochs, num_epochs, 1, desc="Epoch") as pbar:
        for epoch_idx in pbar:
            train_epoch(model, ds, device, learning_rate=learning_rate,
                        batch_size=batch_size, reg_weight=regularization_weight)
            save_path = os.path.join(model_output_dir,
                                     '%s-b%s-r%s-lr%s-%02d.pt' % (time.strftime("%Y%m%d"), batch_size,
                                                                  regularization_weight, learning_rate,
                                                                  epoch_idx))
            torch.save(model, save_path)
            pbar.write("Saved snapshot to %s" % save_path)


def _video_output_path(input_path, amp_f):
    output_dir = os.path.dirname(input_path)
    output_basename, output_ext = os.path.splitext(os.path.basename(input_path))
    output_basename += '@{}x'.format(amp_f)
    output_path = os.path.join(output_dir, output_basename+output_ext)
    return output_path


def amplify(model_path, video_path, *, amplification=1, batch_size=4, device="cuda:0", skip_frames=1):
    device = torch.device(device)
    model = torch.load(model_path).to(device)
    video = VideoFileClip(video_path)
    _to_tensor = transforms.ToTensor()
    last_frames = []
    num_skipped_frames = 5

    def _video_process_frame(input_frame):
        nonlocal last_frames
        frame = _to_tensor(to_pil_image(input_frame)).to(device)
        frame = torch.unsqueeze(frame, 0)
        if len(last_frames) < num_skipped_frames:
            last_frames.append(frame)
            return input_frame
        amp_f_tensor = torch.tensor([[float(amplification)]], dtype=torch.float, device=device)
        pred_frame, _, _ = model.forward(last_frames[0], frame, amp_f_tensor)
        pred_frame = to_pil_image(pred_frame.squeeze(0).detach().cpu())
        pred_frame = np.array(pred_frame)
        last_frames.append(frame)
        last_frames = last_frames[-num_skipped_frames:]
        return pred_frame

    amp_video = video.fl_image(_video_process_frame)
    amp_video.write_videofile(_video_output_path(video_path, amplification))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    clize.run((train, amplify))
