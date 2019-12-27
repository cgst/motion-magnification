import logging
import os
import time

import clize
import numpy as np
import torch
from moviepy.editor import (CompositeVideoClip, TextClip, VideoFileClip,
                            clips_array)
from torch.functional import F
from torchsummary import summary
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from tqdm import trange

from deepmag import dataset
from deepmag.model import MagNet
from deepmag.train import train_epoch


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
    output_basename, output_ext = os.path.splitext(
        os.path.basename(input_path))
    output_basename += '@{}x'.format(amp_f)
    output_path = os.path.join(output_dir, output_basename+output_ext)
    return output_path


def amplify(model_path, input_video, *, amplification=1.0, device="cuda:0", skip_frames=1):
    device = torch.device(device)
    model = torch.load(model_path).to(device)
    video = VideoFileClip(input_video)
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
        amp_f_tensor = torch.tensor(
            [[float(amplification)]], dtype=torch.float, device=device)
        pred_frame, _, _ = model.forward(last_frames[0], frame, amp_f_tensor)
        pred_frame = to_pil_image(pred_frame.squeeze(0).clamp(0, 1).detach().cpu())
        pred_frame = np.array(pred_frame)
        last_frames.append(frame)
        last_frames = last_frames[-num_skipped_frames:]
        return pred_frame

    amp_video = video.fl_image(_video_process_frame)
    output_path = _video_output_path(input_video, amplification)
    amp_video.write_videofile(output_path)
    return output_path


def demo(model_path, input_video, output_video, *amplification_factors, device="cuda:0",
         skip_frames=1):
    amplified = []
    for amp_f in amplification_factors:
        path = amplify(model_path, input_video, amplification=amp_f,
                       device=device, skip_frames=skip_frames)
        amplified.append(path)
    collage(output_video, input_video, *amplified)


def collage(output_video, *input_videos):
    input_clips = []
    for path in input_videos:
        video_clip = VideoFileClip(path)
        _, _, amp = os.path.basename(path).partition("@")
        amp, _, _ = amp.partition('.')
        text_clip = (TextClip(txt='Amplified {}'.format(amp) if amp else 'Input',
                              color='white', method='label', fontsize=32,
                              font='Helvetica-Bold')
                     .set_duration(video_clip.duration)
                     .set_position(('center', 0.05), relative=True))
        clip = CompositeVideoClip((video_clip, text_clip), use_bgclip=True)
        input_clips.append(clip)
    if len(input_clips) < 4:
        num_columns = 1
    elif len(input_clips) < 5:
        num_columns = 2
    else:
        num_columns = 3
    final_clip = clips_array([input_clips[i:i+num_columns]
                              for i in range(0, len(input_clips), num_columns)])
    final_clip.write_videofile(output_video, audio=False)
    return output_video


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    clize.run((train, amplify, demo, collage))
