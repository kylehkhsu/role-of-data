FROM images.borgy.elementai.net/nvcr.io/pytorch-gpu-jupyter:19.07-py3-latest

RUN pip install -U tqdm ipdb wandb torch torchvision
