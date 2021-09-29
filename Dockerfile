FROM tensorflow/tensorflow:2.4.2-gpu

RUN apt-get update && apt-get install -y \
  ffmpeg libgl1-mesa-dev python3-pip unrar wget vim \
  git curl wget gcc make cmake g++ x11-xserver-utils \
  openjdk-8-jdk sudo xvfb ffmpeg zip unzip \
  && apt-get clean

ENV CONDALINK "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"

ENV HOME=/root
RUN curl -so /root/miniconda.sh $CONDALINK \
 && chmod +x /root/miniconda.sh \
 && /root/miniconda.sh -b -p ~/miniconda \
 && rm /root/miniconda.sh
ENV PATH=/root/miniconda/bin:$PATH

RUN conda install conda-build \
 && conda create -y --name py37 python=3.7.3 \
 && conda clean -ya
ENV CONDA_DEFAULT_ENV=py37
ENV CONDA_PREFIX=/root/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH

RUN pip install --no-cache-dir \
  gym==0.18.3 \
  elements==0.3.2 \
  ruamel.yaml \
  tensorflow_probability==0.12.2 \
  tensorflow==2.4.2 \
  moviepy \
  wandb \
  imageio \
  docker==5.0.2 requests==2.26.0 tqdm==4.62.3 pyyaml==5.4.1

RUN pip install git+https://github.com/iglu-contest/iglu.git
RUN python -c 'import iglu;'

ENV TF_XLA_FLAGS --tf_xla_auto_jit=2
