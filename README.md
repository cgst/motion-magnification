# Video Motion Magnification

This repository implements [T. Oh, R. Jaroensri, C. Kim, M. Elgharib, F. Durand, W. Freeman, W. Matusik "Learning-based Video Motion Magnification" arXiv preprint arXiv:1804.02684 (2018)](https://people.csail.mit.edu/tiam/deepmag/) in PyTorch.

Please visit the homepage of the research publication above to see the authors' excellent paper, high-quality TensorFlow code, and a ready-to-use training dataset. The authors did an amazing job, which made it easy for me to create this PyTorch implementation. 🙏🏽

"Motion magnification" means you pick up and amplify small motions so they're easier to see. It's like a microscope for motion.

## Samples

**👉 [Click here to see sample videos from this implementation](https://twitter.com/cgst/status/1210691577078636544) 👈**

You can download the videos [here](https://github.com/cgst/motion-magnification/tree/master/data/examples).

## Set up

This implementation requires Python 3 and PyTorch. Install dependencies with `pip`:

    pip install -r requirements.txt

## Run (examples)

    # Get CLI help.
    python main.py -h

    # Amplify video 5x.
    python main.py amplify data/models/20191204-b4-r0.1-lr0.0001-05.pt data/examples/baby.mp4 --amplification=5

    # Amplify 7x and render side-by-side comparison video.
    python main.py demo data/models/20191204-b4-r0.1-lr0.0001-05.pt data/examples/baby.mp4 baby-demo.mp4 7

    # Make video collage with input, 2x and 4x amplified side by side.
    python main.py demo data/models/20191204-b4-r0.1-lr0.0001-05.pt data/examples/baby.mp4 baby-demo.mp4 2 4


## Training

I've included a pre-trained model so you can try it out of the box, but you can also train from scratch if you wish.

First, [download and extract the training dataset](https://groups.csail.mit.edu/graphics/deep_motion_mag/data/readme.txt) published by the authors. It's ~84GB in total compressed, and slighlty larger deflated. Make sure you have enough disk space.

Example train commands:

    # Get CLI help and adjust training parameters with flags.
    python main.py train -h

    # Train from scratch - this will save a model on disk after each epoch.
    python main.py train data/train data/models --num-epochs=10

    # Resume training from previously saved snapshot and skip 3 epochs.
    python main.py train data/train data/models --num-epochs=10 --skip-epochs=3 --load-model-path=data/models/20191203-b10-r0.1-lr0.0001-02.pt

This implementation takes ~2 days to arrive at a decent loss on a Tesla P100 GPU. Let me know if you speed it up.
