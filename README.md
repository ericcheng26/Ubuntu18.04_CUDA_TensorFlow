> 安裝ubuntu 18.04要記得選取"安裝第三方驅動"

# Ubuntu18.04、python3、TensorFlow安裝教學

---
### 440顯卡驅動(灌完cuda 11.0 後會自動改成450驅動)
1.重灌完後，在grub界面按e進入修改參數，quite splash 後面空一格後加入nomodeset後進入系統  
2.利用ubuntu自帶的"軟體與更新"，選擇欲用的"顯卡驅動"  
3.之後也不需要永久修改grub參數了，重啟後仍然會使用"專用顯卡驅動"
### 藍芽、聲音控制界面
1.bluez从5.48更新到5.50
```
dpkg --status bluez | grep '^Version:'` #查看bluez版本
sudo add-apt-repository ppa:bluetooth/bluez #添加套件源
sudo apt-get update
sudo apt upgrade

#進入藍芽界面scan on、pair和trust mac地址
bluetoothctl
scan on
pair mac地址
trust mac 地址
```
2.聲音控制界面(開啟界面後將"線路輸入"改成"耳機"就可以讓後方面板音源線有聲音)
```
sudo apt install pavucontrol
pavucontrol #開啟聲音控制界面（因為gnome內建的再後方面板音源線預設設定有Bug）
```
### 安裝pip3、conda
1.pip3
```
sudo apt install python3-pip
pip3 install setuptools
```
2.conda
[下載](https://www.anaconda.com/products/individual)
[參考源](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)
```
bash Anaconda3-2020.07-Linux-x86_64.sh
```

### cuda 11.0在turing顯卡下apt安裝
1.安裝前硬體資訊檢查 [[參考源]](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#pre-installation-actions)  
2.進行Runfile Installation [[參考源]](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#pre-installation-actions)
```
lsmod | grep nouveau #確保沒有任何文字被輸出（詳細請見參考源）
```
>這邊不繼續下去～因為比較喜歡用套件管理軟體安裝  

3.進行package Manager Installation [[參考源]](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#pre-installation-actions)
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget http://developer.download.nvidia.com/compute/cuda/11.0.2/local_installers/cuda-repo-ubuntu1804-11-0-local_11.0.2-450.51.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804-11-0-local_11.0.2-450.51.05-1_amd64.deb
sudo apt-key add /var/cuda-repo-ubuntu1804-11-0-local/7fa2af80.pub
sudo apt-get update
sudo apt-get install --no-install-recommends cuda
```
4.新增環境變數至檔案最後面 [[參考源]](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions)
```
vim ~/.bashrc
```
* export PATH=/usr/local/cuda-11.0/bin${PATH:+:${PATH}}
* ~~export LDLIBRARYPATH="/usr/local/cuda-11.0/lib64:${LDLIBRARYPATH}"~~

```
source ~/.bashrc #載入新的bash設定檔
```
5.驗證安裝是否成功
Verify the Driver Version
```
cat /proc/driver/nvidia/version #出現kernel version表成功
```
5.1 查看NVIDIA-SMI和Driver Version的版本號是否一致
```
nvidia-smi #順便看CUDA Version是不是你裝好的！
```
應該會顯示error，因為cuda11自動裝了相容的顯卡驅動450，須重新開機讓系統應用。  

5.2 重開機再確認一次版本號
```
reboot
nvidia-smi #看看NVIDIA-SMI和Driver Version是否一致
```
應該一致了，如果不一致，那肯定沒做好之前的步驟。  

6.（選擇性安裝套件 #我有裝，但好像沒必要）
```
sudo apt-get install g++ freeglut3-dev build-essential libx11-dev \
    libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev
```
7.(當你想要移除cuda)
```
sudo apt-get --purge remove "*cublas*" "*cufft*" "*curand*" \
 "*cusolver*" "*cusparse*" "*npp*" "*nvjpeg*" "cuda*" "nsight*"
```
8.(當你想要移除NVIDIA driver)
```
sudo apt-get --purge remove "*nvidia*"
```
### cudnn 8.0.2在cuda 11.0下(.deb)安裝
0.下載cuDNN v8.0.2 (July 24th, 2020)forCUDA 11.0 [註冊後下載](https://developer.nvidia.com/rdp/cudnn-download)  
|cuDNN Runtime Library for Ubuntu18.04 x86_64 (Deb)  
|cuDNN Developer Library for Ubuntu18.04 x86_64 (Deb)  
|cuDNN Code Samples and User Guide for Ubuntu18.04 x86_64 (Deb)  

1.安裝cudnn庫
```
sudo dpkg -i libcudnn8_8.0.2.39-1+cuda11.0_amd64.deb
```
2.安裝開發者cudnn庫
```
sudo dpkg -i libcudnn8-dev_8.0.2.39-1+cuda11.0_amd64.deb
```
3.驗證cudnn和cuda成功運行
```
sudo dpkg -i libcudnn8-doc_8.0.2.39-1+cuda11.0_amd64.deb
cp -r /usr/src/cudnn_samples_v8/ $HOME
cd  $HOME/cudnn_samples_v8/mnistCUDNN
make clean && make #會出現很多warning沒差拉！
./mnistCUDNN # "Test passed!" 出現表示成功
```
### TensorRT 7.1.3.4在cuda 11.0下(.deb)安裝
0.下載TensorRT 7.1 GA [註冊後下載](https://developer.nvidia.com/nvidia-tensorrt-7x-download)  
|TensorRT 7.1.3.4 for Ubuntu 1804 and CUDA 11.0 DEB local repo packages

1.確認RT支援的推理精度和特殊硬體的支援功能[參考源](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-713/support-matrix/index.html)  

2.先安裝PyCUDA[參考源](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-713/install-guide/index.html#installing-pycuda)

```
nvcc --version #確認nvcc能被bash找到
```
顯示以下資訊表示成功  

:::info
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2020 NVIDIA Corporation
Built on Thu_Jun_11_22:26:38_PDT_2020
Cuda compilation tools, release 11.0, V11.0.194
Build cuda_11.0_bu.TC445_37.28540450_0
:::  

```
pip3 install 'pycuda>=2019.1.1'
```
3.開始TensorRT(.deb)安裝
[參考源](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-713/install-guide/index.html#installing-debian)
```
cd 至"nv-tensorrt-repo-ubuntu1804-cuda11.0-trt7.1.3.4-ga-20200617_1-1_amd64.deb" 檔案資料夾位置

sudo dpkg -i nv-tensorrt-repo-ubuntu1804-cuda11.0-trt7.1.3.4-ga-20200617_1-1_amd64.deb

sudo apt-key add /var/nv-tensorrt-repo-cuda11.0-trt7.1.3.4-ga-20200617/7fa2af80.pub

sudo apt-get update
sudo apt-get install tensorrt cuda-nvrtc-11-0
sudo apt-get install python3-libnvinfer-dev
sudo apt-get install uff-converter-tf
```
4.驗證TensorRT安裝
```
dpkg -l | grep TensorRT
```
顯示以下資訊表示成功  

:::info
ii  graphsurgeon-tf                                             7.1.3-1+cuda11.0                                 amd64        GraphSurgeon for TensorRT package
ii  libnvinfer-bin                                              7.1.3-1+cuda11.0                                 amd64        TensorRT binaries
ii  libnvinfer-dev                                              7.1.3-1+cuda11.0                                 amd64        TensorRT development libraries and headers
ii  libnvinfer-doc                                              7.1.3-1+cuda11.0                                 all          TensorRT documentation
ii  libnvinfer-plugin-dev                                       7.1.3-1+cuda11.0                                 amd64        TensorRT plugin libraries
ii  libnvinfer-plugin7                                          7.1.3-1+cuda11.0                                 amd64        TensorRT plugin libraries
ii  libnvinfer-samples                                          7.1.3-1+cuda11.0                                 all          TensorRT samples
ii  libnvinfer7                                                 7.1.3-1+cuda11.0                                 amd64        TensorRT runtime libraries
ii  libnvonnxparsers-dev                                        7.1.3-1+cuda11.0                                 amd64        TensorRT ONNX libraries
ii  libnvonnxparsers7                                           7.1.3-1+cuda11.0                                 amd64        TensorRT ONNX libraries
ii  libnvparsers-dev                                            7.1.3-1+cuda11.0                                 amd64        TensorRT parsers libraries
ii  libnvparsers7                                               7.1.3-1+cuda11.0                                 amd64        TensorRT parsers libraries
ii  python3-libnvinfer                                          7.1.3-1+cuda11.0                                 amd64        Python 3 bindings for TensorRT
ii  python3-libnvinfer-dev                                      7.1.3-1+cuda11.0                                 amd64        Python 3 development package for TensorRT
ii  tensorrt                                                    7.1.3.4-1+cuda11.0                               amd64        Meta package of TensorRT
ii  uff-converter-tf                                            7.1.3-1+cuda11.0                                 amd64        UFF converter for TensorRT package
:::  

5.如果你要當App Server用來推理
```
sudo apt-get update
sudo apt-get install libnvinfer7 cuda-nvrtc-11-0
sudo apt-get install python3-libnvinfer
```
### TensorFlow 安裝
安裝前重啟一次電腦
[參考源](https://www.tensorflow.org/install/pip)
```
sudo apt update
sudo apt install python3-dev python3-pip
sudo pip3 install -U virtualenv  # system-wide install

#建立虛擬環境
virtualenv --system-site-packages -p python3 ./venv 
source ./venv/bin/activate  # sh, bash, ksh, or zsh
pip install --upgrade pip
pip install --upgrade tensorflow

#驗證tensorflow安裝
python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"

#要關掉虛擬環境（注意tensorflow不能在運行的情況下關閉）
deactivate  # don't exit until you're done using TensorFlow
```
## 我們成功裝好cuda；cudnn；TensorRT;TensorFlow囉！

---
