# Source:
# 1. https://gist.github.com/andyweizhao/639e94b60c166f57964aafedeb465e90(not valid)
# 2. https://stackoverflow.com/questions/60831967/downgrade-cuda-version-on-colab-to-9 (final)

echo "Original Cuda and Nvidia drivers"

echo "==================== Nvidia Driver =================="
nvidia-smi
echo "==================== Python ========================="
python -V
echo "==================== nvcc Driver ===================="
nvcc --version
echo "==================== OS ============================="
cat /etc/os-release
pip uninstall -y torch==1.13


echo "Install conda environment '3dsdn' "
cd /pvc-ssd/Danger_model
if [ -d "/pvc-ssd/Danger_model/DANGER" ]; then
    cd /pvc-ssd/Danger_model/DANGER
    git reset --hard
    # git checkout HEAD -- /pvc-ssd/Danger_model/DANGER
    git pull
  else
    git clone https://github.com/jayhsu0627/DANGER
fi
# git init
conda config --append channels conda-forge
conda env create --name 3dsdn --file /pvc-ssd/Danger_model/DANGER/3D-SDN/environment.yml
conda env list
eval "$(conda shell.bash hook)"
conda activate 3dsdn
pip3 install https://download.pytorch.org/whl/cu100/torch-1.0.0-cp36-cp36m-linux_x86_64.whl
pip3 install torchvision

echo "Install gcc-6 "

# sudo su -c "echo 'deb http://dk.archive.ubuntu.com/ubuntu/ bionic main universe' >> /etc/apt/sources.list"
# sudo apt-get update
# sudo apt-get install -y gcc-6 g++-6 g++-6-multilib gfortran-6
# sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-6 60 --slave /usr/bin/g++ g++ /usr/bin/g++-6
# sudo update-alternatives --config gcc
# gcc --version

echo "Install cuda 9.0 "
# # remove if exists
# rm -rf cuda-repo-ubuntu1704-9-0-176-local-patch-4_1.0-1_amd64-deb || true
# rm -rf cuda-repo-ubuntu1704-9-0-176-local-patch-4_1.0-1_amd64-deb.1 || true
# rm -rf cuda-repo-ubuntu1704-9-0-local_9.0.176-1_amd64-deb || true
# rm -rf cuda-repo-ubuntu1704-9-0-local_9.0.176-1_amd64-deb.1 || true
# # wget https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda-repo-ubuntu1604-9-0-local_9.0.176-1_amd64-deb
# # sudo dpkg -i cuda-repo-ubuntu1604-9-0-local_9.0.176-1_amd64-deb
# # sudo apt-key add /var/cuda-repo-9-0-local/7fa2af80.pub
# # sudo apt-get update
# # sudo apt-get install cuda=9.0.176-1
# # export PATH=$PATH:/usr/local/cuda-9.0/bin
# # export CUDADIR=/usr/local/cuda-9.0
# # export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-9.0/lib64
# # rm -rf cuda-repo-ubuntu1604-9-0-local_9.0.176-1_amd64-deb

# wget https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda_9.0.176_384.81_linux-run
# sudo chmod +x cuda_9.0.176_384.81_linux-run
# sudo sh cuda_9.0.176_384.81_linux-run

echo "Verification of cuda 9.0"

echo "==================== Nvidia Driver =================="
nvidia-smi
echo "==================== Python ========================="
python -V
echo "==================== nvcc Driver ===================="
nvcc --version
echo "==================== Pytorch ========================"

# eval "$(conda shell.bash hook)"
# conda activate 3dsdn
python -c "import torch;print('torch version:',torch.__version__)"
