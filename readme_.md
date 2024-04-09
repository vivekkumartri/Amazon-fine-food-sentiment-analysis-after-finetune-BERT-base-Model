# README

## Introduction
This repository contains code for training and testing a Uformer network for image deblurring. The Uformer network is a deep learning architecture designed for image restoration tasks.

## Environment Setup
To replicate the environment used for training and testing, you can create a conda environment using the provided `environment.yml` file. Run the following command:

conda env create -f environment.yml

This will create a conda environment named `image_deblur_env` with all the necessary dependencies.

## Training
To train the Uformer network, you can run the `train.py` file. Make sure you have activated the conda environment before running the training script.

python train.py <datadir>

Replace `<datadir>` with the path to the directory containing your training data. The script will use this directory as the root directory for training images.

## Evaluation
For evaluation of the trained model, you can use the `evaluation.py` script. This script takes several command-line arguments:

python evaluation.py <blur_image_dir> <output_dir> <checkpoint_path> <batch_size>

- `<blur_image_dir>`: Path to the directory containing the blurry images to be deblurred.
- `<output_dir>`: Path to the directory where the deblurred images will be saved.
- `<checkpoint_path>`: Path to the trained model checkpoint.
- `<batch_size>`: Batch size for evaluation (must be an integer).

Example usage:

python evaluation.py /path/to/blurry/images /path/to/output/directory /path/to/checkpoint/model.ckpt 16

This script will take the blurry images from `<blur_image_dir>`, deblur them using the trained model checkpoint specified by `<checkpoint_path>`, and save the deblurred images to `<output_dir>`.

## Note
Ensure that your system has sufficient resources (especially GPU memory) to train and evaluate the Uformer network, as these tasks can be computationally intensive.

For any further questions or issues, please feel free to contact the repository owner.
