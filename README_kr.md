[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fchansoopark98%2FWindows-TensorRT-Python&count_bg=%2379C83D&title_bg=%232980E5&icon=python.svg&icon_color=%23E7E7E7&title=hits&edge_flat=true)](https://hits.seeyoufarm.com)

# Windows-TensorRT-Python
Repository on how to install and infer TensorRT Python on Windows

윈도우 환경에서 Tensorflow, PyTorch model을 TensorRT로 변환하고, 변환된 모델을 추론하는 예제를 포함합니다.

<br>
<hr>

# Table of Contents

 ## 1. [Install CUDA & CuDNN & TensorRT](#1-install-cuda--cudnn--tensorrt)
 ## 2. [Install TensorRT python](#2-install-tensorrt-python)
 ## 3. [Convert DL Models](#3-convert-dl-models)
 ## 4. [Inference](#4-inference)

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

<br>
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

## 1.3 Installation TensorRT SDK

- **TensorRT** : https://developer.nvidia.com/nvidia-tensorrt-8x-download (TensorRT 8.2 GA Update 4)

<br>

- **폴더 이동** :

    설치받은 TensorRT .zip 파일을 C:\ 루트 디렉토리로 이동
    ```cmd
    cd c:\TensorRT-8.2.5.1>
    ```

    <br>

- **Copy & Paste .dll, .lib files**

    명령 프롬포트(cmd) -> 순차적으로 명령어 실행
    ```cmd
    copy c:\TensorRT-8.2.5.1\include "c:\Program Files\NVIDIA GPU Computing     Toolkit\CUDA\v11.4\include"

    robocopy c:\TensorRT-8.2.5.1\lib "c:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.4\lib\x64" *.lib

    robocopy c:\TensorRT-8.2.5.1\lib "c:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.4\bin" *.dll
    ```

<br>
<hr>
<br>


# 2.1 Install TensorRT python

## 2.1 Create virtual enviroments

- **가상 환경 설정**:
    ```cmd
    conda create -n tensorrt python=3.8
    ```

<br>

- **TensorRT Python 설치**:
    ```cmd
    conda activate tensorrt

    cd c:\TensorRT-8.2.5.1

    pip install python/tensorrt-8.2.5.1-cp38-none-win_amd64.whl (가상환경 버전에 따라 cp36, cp37, cp38, cp39 선택)

    pip install uff/uff-0.6.9-py2.py3-none-any.whl

    pip install graphsurgeon/graphsurgeon-0.4.5-py2.py3-none-any.whl

    pip install onnx_graphsurgeon/onnx_graphsurgeon-0.3.12-py2.py3-none-any.whl
    ```

- **설치 확인**:

    ![image](https://github.com/chansoopark98/Windows-TensorRT-Python/assets/60956651/0e44b042-b5aa-492d-9e0b-53c0c62319a9)
    

- **pycuda 설치**:
    ```cmd
    pip install pycuda
    ```

<br>
<hr>
<br>

# 3. Convert DL Models

TensorRT는 Tensorflow, PyTorch, ONNX 등 다양한 DL 프레임워크를 지원합니다.

본 레포지토리에서는 ONNX를 통한 TensorRT 모델 변환 예제를 포함합니다.

## 3.1 Install ONNX

ONNX 설치를 위해 가상환경이 활성화된 상태에서 설치합니다.
```cmd
pip install onnx onnxruntime
```

<br>

## 3.2 Convert to ONNX

- 3.2.1 Tensorflow to ONNX

    - 모델 저장 :

        Tensorflow에서 ONNX로 쉽게 변환하기 위해 tensorflow saved model format을 기준으로 합니다.

        학습 또는 추론코드에서 tensorflow 모델 객체를 저장합니다.

        ```python
        import tensorflow as tf
        """ load your tensorflow model """
        model = load_model_func(*args)
        tf.saved_model.save(model, your_save_path)
        ```

        **your_save_path** 는 저장 경로이며 별도의 확장자는 명시하지 않아도 됩니다.

    - tf2onnx 설치:
        ```cmd
        pip install -U tf2onnx
        ```

    - 모델 변환:
        ```cmd
        python -m tf2onnx.convert --saved-model ./your_save_path/ --output model.onnx --opset 13
        ```

        <br>

        **주의 사항**

        1. onnx 버전에 따라 **--opset** 버전을 조정해야 합니다.
        2. saved model format이 아닌 다른 형태로도 변환이 가능합니다. (frozen_graph, checkpoint)
        3. 자세한 사항은 **python -m tf2onnx.convert --help**를 통해 확인할 수 있습니다.
        

<br>

- 3.2.2 PyTorch to ONNX

    파이토치 프레임워크는 내장 함수를 이용하여 ONNX 모델로 export 합니다.

    - 모델 변환

        ```python
        import torch
        model = load_your_model()
        torch.onnx.export(model,               
        x,                         
        your_save_path + '.onnx',
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names = ['input'],
        output_names = ['output'],
        dynamic_axes={'input' : {0 : 'batch_size'},
                    'output' : {0 : 'batch_size'}})
        ```


        **주의 사항**

        1. onnx 버전에 따라 **--opset** 버전을 조정해야 합니다.
        2.  PyTorch 모델마다 input_names, output_names가 다르니 layer name에 맞게 지정해서 변환을 수행합니다.

<br>

- 3.3 ONNX to TensorRT

    Tensorflow/PyTorch에서 변환된 ONNX 모델을 TensorRT engine으로 변환합니다.

    변환된 .onnx 파일을 아래 경로로 복사합니다.
    ```cmd
    copy your_saved_onnx_file.onnx c:\TensorRT-8.2.5.1\bin\
    ```

    <br>
    
     trtexec를 사용하여 tensorRT engine으로 변환합니다.
    ```cmd
    .\trtexec.exe --onnx=your_saved_onnx_file.onnx --saveEngine=model.trt
    ```

    <br>

    변환 시 **--help** 명령어를 이용해 추가 최적화 옵션 등 설정이 가능합니다.
    ```cmd
    .\trtexec.exe --help
    ```

    <br>

    변환이 완료된 경우, 아래 경로에 tensorRT engine 파일이 생성됩니다.
    ```cmd
    c:\TensorRT-8.2.5.1\bin\model.trt
    ```

<br>
<hr>
<br>
    
# 4. Inference

TensorRT engine파일의 추론 속도와 출력 결과를 확인할 수 있습니다.

```cmd
python tensorRT_inference_example.py --model=model.trt --b 1 --h 224 --w 224 -c 3
```

PyTorch 모델 shape(B,C,H,W)은 --torch_mode를 활성화합니다.
```cmd
python tensorRT_inference_example.py --model=model.trt --b 1 --h 224 --w 224 -c 3 --torch_mode
```