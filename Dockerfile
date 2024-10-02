FROM continuumio/miniconda3:23.3.1-0
MAINTAINER Lindsay Romo <lindsay.romo@childrens.harvard.edu>


LABEL \
 version="0.0.1" \
 description="Image for RegVar(https://github.com/RomoL2/RegVar)"


RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
   build-essential \
   curl \
   git \
   gcc-4.8\
   git-lfs \
   libcurl4-openssl-dev \
   libfontconfig1-dev \
   libfreetype6-dev \
   libfribidi-dev \
   libharfbuzz-dev \
   libjpeg-dev \
   libpng-dev \
   libssl-dev \
   libtiff5-dev \
   libxml2-dev \
   python3 \
   bedtools \
   python3-pip \
   r-base \
   vim \
   wget \
   && apt-get clean \
   && rm -rf /var/lib/apt/lists/*


RUN conda install -n base -c conda-forge -y mamba

RUN R -e "install.packages('devtools', dependencies=TRUE)"
RUN R -e "devtools::install_github('RomoL2/RegVar')"


WORKDIR /usr/local/lib/R/site-library/RegVar
RUN rm -r extdata \
 && wget https://zenodo.org/record/13738622/files/extdata.tar.gz \
 && tar -xf extdata.tar.gz \
 && rm extdata.tar.gz \
 && cd extdata \
 && wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz \
 && gunzip hg38.fa.gz 


# install RBPamp
WORKDIR /usr/local/lib/R/site-library/RegVar/extdata/
RUN rm -r RBPamp \
  && git clone https://marjens@bitbucket.org/marjens/RBPamp.git

WORKDIR /usr/local/lib/R/site-library/RegVar/extdata/RBPamp
RUN mamba create --name RBPamp --file requirements.txt -c conda-forge --yes


RUN /opt/conda/envs/RBPamp/bin/pip install future-fstrings --force-reinstall
RUN export CC=gcc
RUN /opt/conda/envs/RBPamp/bin/python setup.py build \
 && /opt/conda/envs/RBPamp/bin/python setup.py install


WORKDIR /