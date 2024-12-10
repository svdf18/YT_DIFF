# Audio Generation Model Training

This repository contains the implementation of a hybrid audio generation model combining VAE (Variational Autoencoder) and EDM2 (Enhanced Denoising Diffusion Model) approaches.

## Setup

1. Clone the repository
2. Install dependencies:

`pip install requirements.txt`


###Run the VAE model
`PYTHONPATH=$PYTHONPATH:. python src/training/train.py --data_dir data --batch_size 32 --num_epochs 100 --model_type vae`

###Run the EDM2 model
`PYTHONPATH=$PYTHONPATH:. python src/training/train.py --data_dir data --batch_size 32 --num_epochs 100 --model_type edm2`