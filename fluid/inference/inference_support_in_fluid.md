<!-- $theme: default -->


<!-- $size: 16:9 -->

# <center>Fluid Inference使用指南

## <center>刘益群

---

<!-- page_number: true -->

- Python Inference API
- 编译C++ Inference库
- C++ Inference API

---

## Python Inference API

---

## 编译C++ Inference库

- **不需要额外的CMake选项**, [源码编译PaddlePaddle](http://www.paddlepaddle.org/docs/develop/documentation/zh/build_and_install/build_from_source_cn.html)
<small>
  - 配置CMake命令
    ```
    $ git clone https://github.com/PaddlePaddle/Paddle.git
    $ cd Paddle
    $ mkdir build
    $ cd build
    $ cmake -DCMAKE_INSTALL_PREFIX=your/path/to/paddle_inference_lib \
        -DCMAKE_BUILD_TYPE=Release \
        -DWITH_PYTHON=ON \
        -DWITH_MKL=OFF \
        -DWITH_GPU=OFF \
        ..
    ```
  - 编译PaddlePaddle
    ```
    $ make
    ```
  - 部署PaddlePaddle Fluid Inference库，执行该命令后，
    ```
    $ make inference_lib_dist
    ```
</small>

---

## C++ Inference API
### 基本概念
- Tensor，LoDTensor
- Place：CPUPlace，CUDAPlace
- Scope
- Executor
- 初始化函数：InitDevices
