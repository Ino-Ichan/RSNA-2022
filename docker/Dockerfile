FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y git tmux libopencv-dev

RUN pip install tensorboard opencv-python
RUN conda install -y jupyter jupyterlab
RUN conda install -y pandas scikit-learn matplotlib seaborn
RUN pip install albumentations==0.5.2
RUN pip install timm==0.5.4
RUN pip install pytorch-gradcam==0.2.1
RUN pip install numpy==1.20.0
RUN pip install iterative-stratification==0.1.6
RUN pip install wandb==0.10.31
RUN pip install geffnet==0.9.8 warmup_scheduler==0.3