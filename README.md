[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fchansoopark98%2FWindows-TensorRT-Python&count_bg=%2379C83D&title_bg=%232980E5&icon=python.svg&icon_color=%23E7E7E7&title=hits&edge_flat=true)](https://hits.seeyoufarm.com)

# Windows-TensorRT-Python
Repository on how to install and infer TensorRT Python on Windows

Includes examples of converting Tensorflow and PyTorch models to TensorRT in the Windows environment and inferring the converted models.


## 한국어 [README.md](https://github.com/chansoopark98/Windows-TensorRT-Python/blob/main/README_kr.md) 지원

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

- **Install CUDA Toolkit** :

    https://developer.nvidia.com/cuda-11-4-4-download-archive

- **Set windows environment variable** :

    ![image](https://github.com/chansoopark98/Windows-TensorRT-Python/assets/60956651/cb362cd5-5a64-4579-9aa9-5756b4370fd8)


<br>

## 1.2 Installation CuDNN

- **CuDNN** : https://developer.nvidia.com/rdp/cudnn-archive#a-collapse841-
<br>

- **Copy files** :

    Move the installed CuDNN **'bin', 'include', 'lib'** folder into the CUDA folder of the installed version
    <br>

    ![image](https://github.com/chansoopark98/Windows-TensorRT-Python/assets/60956651/c603a448-8fcf-4d0e-90cc-6939c0ad0fba)

    <br>

    Verify CUDA installation after reboot

        cmd -> nvcc -V
    
    ![image](https://github.com/chansoopark98/Windows-TensorRT-Python/assets/60956651/86d47736-2976-4430-ab86-afc66b59210f)

## 1.3 Installation TensorRT SDK

- **TensorRT** : https://developer.nvidia.com/nvidia-tensorrt-8x-download (TensorRT 8.2 GA Update 4)

<br>

- **Move directory** :

    Move the installed TensorRT .zip file to C:\ root directory
    ```cmd
    cd c:\TensorRT-8.2.5.1>
    ```

    <br>

- **Copy & Paste .dll, .lib files**

    Command Prompt (cmd) -> Run commands sequentially
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

- **Setting up a virtual environment**:
    ```cmd
    conda create -n tensorrt python=3.8
    ```

<br>

- **Install TensorRT Python**:
    ```cmd
    conda activate tensorrt

    cd c:\TensorRT-8.2.5.1

    pip install python/tensorrt-8.2.5.1-cp38-none-win_amd64.whl (가상환경 버전에 따라 cp36, cp37, cp38, cp39 선택)

    pip install uff/uff-0.6.9-py2.py3-none-any.whl

    pip install graphsurgeon/graphsurgeon-0.4.5-py2.py3-none-any.whl

    pip install onnx_graphsurgeon/onnx_graphsurgeon-0.3.12-py2.py3-none-any.whl
    ```

- **Check installation**:

    ![image](https://github.com/chansoopark98/Windows-TensorRT-Python/assets/60956651/0e44b042-b5aa-492d-9e0b-53c0c62319a9)
    

- **Install pycuda**:
    ```cmd
    pip install pycuda
    ```

<br>
<hr>
<br>

# 3. Convert DL Models

TensorRT supports various DL frameworks including Tensorflow, PyTorch, and ONNX.

This repository contains examples of converting TensorRT models via ONNX.

## 3.1 Install ONNX

For ONNX installation, install with the virtual environment activated.

```cmd
pip install onnx onnxruntime
```

<br>

## 3.2 Convert to ONNX

- 3.2.1 Tensorflow to ONNX

    - Save model :

        It is based on the tensorflow saved model format for easy conversion from Tensorflow to ONNX.

        Store tensorflow model objects in your training or inference code.

        ```python
        import tensorflow as tf
        """ load your tensorflow model """
        model = load_model_func(*args)
        tf.saved_model.save(model, your_save_path)
        ```

        **your_save_path** is the save path, and no extension is required.

    - Install tf2onnx:
        ```cmd
        pip install -U tf2onnx
        ```

    - Model conversion:
        ```cmd
        python -m tf2onnx.convert --saved-model ./your_save_path/ --output model.onnx --opset 13
        ```

        <br>

        **Caution**

        1. You need to adjust the **--opset** version according to your onnx version.
        2. It can be converted to other forms other than the saved model format. (frozen_graph, checkpoint)
        3. Details can be checked through **python -m tf2onnx.convert --help**.
        

<br>

- 3.2.2 PyTorch to ONNX

    The PyTorch framework uses built-in functions to export ONNX models.

    - Model conversion

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


        **Caution**

        1. You need to adjust the **--opset** version according to your onnx version.
        2. Input_names and output_names are different for each PyTorch model, so convert according to the layer name.

<br>

- 3.3 ONNX to TensorRT

    Convert ONNX models converted from Tensorflow/PyTorch to TensorRT engine.

    Copy the converted .onnx file to the path below.
    ```cmd
    copy your_saved_onnx_file.onnx c:\TensorRT-8.2.5.1\bin\
    ```

    <br>
    
    Convert to tensorRT engine using trtexec.
    ```cmd
    .\trtexec.exe --onnx=your_saved_onnx_file.onnx --saveEngine=model.trt
    ```

    <br>

    During conversion, additional optimization options can be set using the **--help** command.
    ```cmd
    .\trtexec.exe --help
    ```

    <br>

    When the conversion is complete, the tensorRT engine file is created in the path below.
    ```cmd
    c:\TensorRT-8.2.5.1\bin\model.trt
    ```

<br>
<hr>
<br>
    
# 4. Inference

You can check the inference speed and output results of the TensorRT engine file.

```cmd
python tensorRT_inference_example.py --model=model.trt --b 1 --h 224 --w 224 -c 3
```

PyTorch model shape(B,C,H,W) enable --torch_mode.
```cmd
python tensorRT_inference_example.py --model=model.trt --b 1 --h 224 --w 224 -c 3 --torch_mode
```