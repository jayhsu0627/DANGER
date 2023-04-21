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


echo "Install conda environment '3dsdn' "
cd /pvc-ssd/Danger_model
if [ -d "/pvc-ssd/Danger_model/DANGER" ]; then
    git reset --hard
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
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 -c pytorch


echo "Install gcc-6 "

sudo su -c "echo 'deb http://dk.archive.ubuntu.com/ubuntu/ bionic main universe' >> /etc/apt/sources.list"
sudo apt-get update

sudo apt-get install gcc-6 g++-6 g++-6-multilib gfortran-6
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-6 60 --slave /usr/bin/g++ g++ /usr/bin/g++-6
sudo update-alternatives --config gcc

# # remove if exists
# rm -rf cuda-repo-ubuntu1704-9-0-176-local-patch-4_1.0-1_amd64-deb || true
# rm -rf cuda-repo-ubuntu1704-9-0-176-local-patch-4_1.0-1_amd64-deb.1 || true
# rm -rf cuda-repo-ubuntu1704-9-0-local_9.0.176-1_amd64-deb || true
# rm -rf cuda-repo-ubuntu1704-9-0-local_9.0.176-1_amd64-deb.1 || true

echo "Install pytorch and cuda 9.2 "
# wget https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda-repo-ubuntu1604-9-0-local_9.0.176-1_amd64-deb
# sudo dpkg -i cuda-repo-ubuntu1604-9-0-local_9.0.176-1_amd64-deb
# apt-key add /var/cuda-repo-9-0-local/7fa2af80.pub
# apt-get update
# apt-get install cuda=9.0.176-1

# export PATH=$PATH:/usr/local/cuda-9.0/bin
# export CUDADIR=/usr/local/cuda-9.0
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-9.0/lib64

echo "Verification of cuda 9.0"

echo "==================== Nvidia Driver =================="
nvidia-smi
echo "==================== Python ========================="
python -V
echo "==================== nvcc Driver ===================="
nvcc --version
rm -rf cuda-repo-ubuntu1604-9-0-local_9.0.176-1_amd64-deb
conda activate 3dsdn
