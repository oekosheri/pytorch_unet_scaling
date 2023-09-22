#!/usr/local_rwth/bin/zsh

# create virtual environment based on loaded python version
python3 -m venv horovod-env-rocky
source horovod-env-rocky/bin/activate


# install pytorch and supporting libraries
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

pip3 install scikit-learn
pip3 install pandas
pip3 install opencv-python
pip3 install black

# build and install horovod (will be linked against loaded MPI version)
HOROVOD_GPU_OPERATIONS=NCCL     \
HOROVOD_WITH_MPI=1              \
HOROVOD_WITHOUT_TENSORFLOW=1    \
HOROVOD_WITH_PYTORCH=1          \
pip3 install --no-cache-dir horovod
