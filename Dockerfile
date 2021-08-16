# set up the environement
FROM ubuntu:18.04
# add paths
ENV PATH /root/miniconda3/bin:$PATH
ARG PATH /root/miniconda3/bin:$PATH

# install packages
RUN apt update
RUN apt install -y htop python3-dev wget

# install conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN mkdir root/.conda
RUN sh Miniconda3-latest-Linux-x86_64.sh -b
RUN rm -f Miniconda3-latest-Linux-x86_64.sh
# create conda environement
RUN conda create -y -n ml python=3.7

# copy files and install packages
COPY . src/
# create directory output for training if not exist (must be the same as in config file)
RUN mkdir -p src/output
RUN /bin/bash -c "cd src \
    && source activate ml \
    && pip install -r requirements.txt"

# specify a port
EXPOSE 5000