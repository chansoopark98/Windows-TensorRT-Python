[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fchansoopark98%2FWindows-TensorRT-Python&count_bg=%2379C83D&title_bg=%232980E5&icon=python.svg&icon_color=%23E7E7E7&title=hits&edge_flat=true)](https://hits.seeyoufarm.com)

# Windows-TensorRT-Python
Repository on how to install and infer TensorRT Python on Windows

윈도우 환경에서 Tensorflow model을 TensorRT로 변환하고, 변환된 모델을 추론하는 예제를 포함합니다.

<br>
<hr>

# Table of Contents

 ## 1. [Install CUDA & CuDNN & TensorRT](#1-models-1)
 ## 2. [Install TensorRT python](#2-dependencies-1)
 ## 3. [Convert DL Models](#3-preparing-datasets-1)
 ## 4. [Inference](#4-train-1)

<br>
<hr>

## Dependency

| Type | Name |
| :-- | :-: |
| **OS** | Windows 11 Pro (22H2 Version) |
| **CPU** | Intel i7-12650H 2.30GHz |
| **RAM** | 16GB |
| **GPU** | Nvidia RTX 3050ti laptop |
| **Tensorflow** | Tensorflow 2.9.1 |
| **TensorRT** | TensorRT-8.2.5.1 |
| **CUDA Toolkit** | CUDA Toolkit 11.4 |
| **CuDNN** | CuDNN v8.4.1 (May 27th, 2022), for CUDA 11.x |

<hr>
<br>

# 1. Install CUDA & CuDNN & TensorRT

## 1.1 Installation CUDA
<br>

- **CUDA Toolkit 설치 ** :

    https://developer.nvidia.com/cuda-11-4-4-download-archive

- **환경 변수 설정** :

    ![image](https://github.com/chansoopark98/Windows-TensorRT-Python/assets/60956651/cb362cd5-5a64-4579-9aa9-5756b4370fd8)


<br>

## 1.2 Installation CuDNN

- **CuDNN** : https://developer.nvidia.com/rdp/cudnn-archive#a-collapse841-
<br>

- **파일 복사** :

    설치받은 CuDNN **'bin', 'include', 'lib'** 폴더를 설치한 버전의 CUDA 폴더 안으로 이동
    <br>

    ![image](https://github.com/chansoopark98/Windows-TensorRT-Python/assets/60956651/c603a448-8fcf-4d0e-90cc-6939c0ad0fba)

    <br>

    재부팅 후 CUDA 설치 확인

        cmd -> nvcc -V
    
    ![image](https://github.com/chansoopark98/Windows-TensorRT-Python/assets/60956651/86d47736-2976-4430-ab86-afc66b59210f)

## 1.3 Installation TensorRT

- **TensorRT** : https://developer.nvidia.com/nvidia-tensorrt-8x-download (TensorRT 8.2 GA Update 4)

<br>

- **폴더 이동** :

    설치받은 TensorRT .zip 파일을 C: 루트 디렉토리로 이동

        c:\TensorRT-8.2.5.1>

    <br>

- CUDA Install


# 2.1 Install TensorRT python

## 2.1 Create virtual enviroments

- **가상 환경 설정**:

        conda create -n tensorrt python=3.8

<br>

- **TensorRT Python 설치**:

        conda activate tensorrt

        cd c:\TensorRT-8.2.5.1

        pip install python/tensorrt-8.2.5.1-cp38-none-win_amd64.whl (가상환경 버전에 따라 cp36, cp37, cp38, cp39 선택)

        pip install uff/uff-0.6.9-py2.py3-none-any.whl

        pip install graphsurgeon/graphsurgeon-0.4.5-py2.py3-none-any.whl

        pip install onnx_graphsurgeon/onnx_graphsurgeon-0.3.12-py2.py3-none-any.whl

- **설치 확인**:

    ![image](https://github.com/chansoopark98/Windows-TensorRT-Python/assets/60956651/0e44b042-b5aa-492d-9e0b-53c0c62319a9)
    


        


