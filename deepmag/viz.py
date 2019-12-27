import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import ImageChops
from torchvision.transforms.functional import to_pil_image


def show_sample(sample):
    frame_a, frame_b, frame_pert, frame_amp = \
        map(to_pil_image,
            (sample['frame_a'], sample['frame_b'],
             sample['frame_perturbed'], sample['frame_amplified']))
    fig = plt.figure(figsize=(16, 10))
    grid = ImageGrid(fig, 111, nrows_ncols=(1, 4), axes_pad=(0.1, 0.6))
    grid[0].imshow(frame_a)
    grid[0].title.set_text('A')
    grid[1].imshow(frame_b)
    grid[1].title.set_text('B')
    grid[2].imshow(ImageChops.subtract(frame_b, frame_a))
    grid[2].title.set_text('Motion 1x')
    grid[3].imshow(ImageChops.subtract(frame_amp, frame_a))
    grid[3].title.set_text('Motion %.2fx' % sample['amplification_f'])


def show_pred(frame_a, frame_b, amp_f, frame_pred):
    frame_a, frame_b, frame_pred = \
        map(to_pil_image,
            (frame_a, frame_b, frame_pred))
    fig = plt.figure(figsize=(16, 10))
    grid = ImageGrid(fig, 111, nrows_ncols=(1, 4), axes_pad=(0.1, 0.6))
    grid[0].imshow(frame_a)
    grid[0].title.set_text('A')
    grid[1].imshow(frame_b)
    grid[1].title.set_text('Amplified (Ground Truth)')
    grid[2].imshow(frame_pred)
    grid[2].title.set_text('Predicted Frame')
    grid[3].imshow(ImageChops.subtract(frame_pred, frame_b))
    grid[3].title.set_text('Predicted Motion %.2fx' % amp_f)
