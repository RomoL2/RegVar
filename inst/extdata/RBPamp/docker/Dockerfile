# Make an intermediate image, just to get source code and compile cython code
FROM ubuntu:xenial as git_and_build
# install git and minimal dependencies to compile Cython modules
RUN apt-get update && \
  apt-get install --no-install-recommends -y \
  git \
  ssh \
  # python3-numpy \
  python3-pip \
  python3-dev \
  build-essential

RUN pip3 install --no-cache-dir --compile --trusted-host pypi.python.org 'Cython==0.29.13'
RUN pip3 install --no-cache-dir --compile --trusted-host pypi.python.org 'setuptools==41.2.0'
RUN pip3 install --no-cache-dir --compile --trusted-host pypi.python.org 'numpy==1.12.0'

RUN python3 -c "import numpy; print(numpy.__version__)"

# add credentials on build
ARG SSH_PRIVATE_KEY
RUN mkdir /root/.ssh/
RUN chmod 700 /root/.ssh
RUN echo "${SSH_PRIVATE_KEY}" > /root/.ssh/id_rsa
RUN chmod 600 /root/.ssh/id_rsa
RUN cat /root/.ssh/id_rsa

# make sure your domain is accepted
RUN touch /root/.ssh/known_hosts
RUN ssh-keyscan bitbucket.org >> /root/.ssh/known_hosts
# Pull source code
WORKDIR /
RUN pwd
RUN pwd
RUN git clone git@bitbucket.org:marjens/RBPamp.git

WORKDIR /RBPamp
RUN git describe --always > git_commit
RUN python3 -c "import numpy; print(numpy.__version__)"
RUN python3 -m pip list
RUN python3 setup.py build_ext --inplace
RUN rm -rf /RBPamp/build



## The final image. Start fresh
FROM ubuntu:xenial
RUN pwd
COPY --from=git_and_build /RBPamp /RBPamp
RUN apt-get update && apt-get install  --no-install-recommends -y \
    python3-pip \
    python3.5 \
    libgomp1
    # python3-matplotlib

WORKDIR /

RUN pwd
COPY run_req.txt /req.txt
RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir --compile --trusted-host pypi.python.org "setuptools==41.2.0"
RUN pip3 install --no-cache-dir --compile --trusted-host pypi.python.org -r req.txt
RUN python3 -c "import numpy; print(numpy.__version__)"
ENV PYTHONPATH "/RBPamp"
RUN mkdir target
WORKDIR /target
RUN pip3 uninstall --yes setuptools pip
RUN apt-get -yq purge gcc g++ perl manpages python3-pip
RUN apt-get -yq clean && apt-get -yq remove ca-certificates dh-python python-pip-whl
# RUN apt-get clean && apt-get autoremove
RUN rm -rf /var/lib/apt/lists
RUN rm -rf /usr/lib/gcc/x86_64-linux-gnu
RUN rm -rf share/man
RUN rm -rf share/locale
ENTRYPOINT ["python3", "/RBPamp/bin/RBPamp"]
# CMD ["bash"]
