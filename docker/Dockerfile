# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Adapted from https://github.com/microsoft/nlp-recipes/blob/master/docker/Dockerfile

FROM nvidia/cuda

# Install Anaconda
# Non interactive installation instructions can be found 
# https://hub.docker.com/r/continuumio/anaconda/dockerfile
# https://hub.docker.com/r/continuumio/miniconda/dockerfile
ENV PATH /opt/conda/bin:$PATH
SHELL ["/bin/bash", "-c"]

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion


RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# Get the latest staging version repository
WORKDIR /root
RUN apt-get install -y zip && \
    wget --quiet https://github.com/microsoft/mt-dnn/archive/staging.zip -O staging.zip && \
    unzip staging.zip  && rm staging.zip 

# Install the packages
WORKDIR /root/mt-dnn-staging
RUN python /root/mt-dnn-staging/scripts/generate_conda_file.py --gpu && \
    conda env create -n mtdnn_gpu -f mtdnn_gpu.yaml 
RUN source activate mtdnn_gpu && \
    pip install -e . && \
    python -m ipykernel install --user --name mtdnn_gpu --display-name "Python (mtdnn_gpu)"

# Run notebook
EXPOSE 8888/tcp
WORKDIR /root/mt-dnn-staging
CMD source activate mtdnn_gpu && \
    jupyter notebook --allow-root --ip 0.0.0.0 --port 8888 --no-browser --notebook-dir .

