<!-- $theme: default -->


<!-- $size: 16:9 -->

# <center>Fluid Inference使用指南

## <center>刘益群

---

<!-- page_number: true -->

- Python Inference API
- 编译Fluid Inference库
- Inference C++ API

---

## Python Inference API **[正在改进中]**
- 保存Inference模型 [:arrow_forward:](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/io.py#L295)
  ```python
  def save_inference_model(dirname,
                           feeded_var_names,
                           target_vars,
                           executor,
                           main_program=None,
                           model_filename=None,
                           params_filename=None):
  ```

---

## 编译Fluid Inference库

  - **不需要额外的CMake选项**<small>
    - :one: 配置CMake命令，更多配置请参考[源码编译PaddlePaddle](http://www.paddlepaddle.org/docs/develop/documentation/zh/build_and_install/build_from_source_cn.html)
      ```bash
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
    - :two: 编译PaddlePaddle
      ```bash
      $ make
      ```
    - :three: 部署。执行如下命令将PaddlePaddle Fluid Inference库部署到`your/path/to/paddle_inference_lib`目录。
      ```bash
      $ make inference_lib_dist
      ```
  </small>

---

## 编译Fluid Inference库

- 目录结构<small>

  ```bash
  $ cd your/path/to/paddle_inference_lib
  $ tree
  .
  |-- paddle
  |   `-- fluid
  |       |-- framework
  |       |-- inference
  |       |   |-- io.h
  |       |   `-- libpaddle_fluid.so
  |       |-- memory
  |       |-- platform
  |       `-- string
  |-- third_party
  |   |-- eigen3
  |   `-- install
  |       |-- gflags
  |       |-- glog
  |       `-- protobuf
  `-- ...
  ```

  假设`PADDLE_ROOT=your/path/to/paddle_inference_lib`。
</small>

---

## 链接Fluid Inference库
- [示例项目](https://github.com/luotao1/fluid_inference_example.git)
  <small>
  - GCC配置
    ```bash
    $ g++ -o a.out main.cc \
          -I${PADDLE_ROOT}/ \
          -I${PADDLE_ROOT}/third_party/install/gflags/include \
          -I${PADDLE_ROOT}/third_party/install/glog/include \
          -I${PADDLE_ROOT}/third_party/install/protobuf/include \
          -I${PADDLE_ROOT}/third_party/eigen3 \
          -L{PADDLE_ROOT}/paddle/fluid/inference -lpaddle_fluid \
          -lrt -ldl -lpthread
    ```
  - CMake配置
    ```cmake
    include_directories(${PADDLE_ROOT}/)
    include_directories(${PADDLE_ROOT}/third_party/install/gflags/include)
    include_directories(${PADDLE_ROOT}/third_party/install/glog/include)
    include_directories(${PADDLE_ROOT}/third_party/install/protobuf/include)
    include_directories(${PADDLE_ROOT}/third_party/eigen3)
    target_link_libraries(${TARGET_NAME}
                          ${PADDLE_ROOT}/paddle/fluid/inference/libpaddle_fluid.so
                          -lrt -ldl -lpthread)
    ```
  </small>

---

## C++ Inference API

- 推断流程  [:arrow_forward:](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/inference/tests/test_helper.h#L91)
  <small>
  - :zero: 初始化设备
    ```cpp
    #include "paddle/fluid/framework/init.h"
    paddle::framework::InitDevices();
    ```
  - :one: 定义place，executor，scope
    ```cpp
    auto place = paddle::platform::CPUPlace();
    auto executor = paddle::framework::Executor(place);
    auto* scope = new paddle::framework::Scope();
    ```
  - :two: 加载模型
    ```cpp
    #include "paddle/fluid/inference/io.h"
    auto inference_program = paddle::inference::Load(executor, *scope, dirname);
    // or
    auto inference_program = paddle::inference::Load(executor,
                                                     *scope,
                                                     dirname + "/" + prog_filename,
                                                     dirname + "/" + param_filename);
    ```
  </small>
---

## C++ Inference API

- 推断流程  [:arrow_forward:](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/inference/tests/test_helper.h#L91)
  <small>

  - :three: 获取`feed_target_names`和`fetch_target_names`
    ```cpp
    const std::vector<std::string>& feed_target_names = inference_program->GetFeedTargetNames();
    const std::vector<std::string>& fetch_target_names = inference_program->GetFetchTargetNames();
    ```
  - :four: 准备`feed`数据
    ```cpp
    std::vector<paddle::framework::LoDTensor*> cpu_feeds;
    ...
    std::map<std::string, const paddle::framework::LoDTensor*> feed_targets;
    for (size_t i = 0; i < feed_target_names.size(); ++i) {
      // Please make sure that cpu_feeds[i] is right for feed_target_names[i]
      feed_targets[feed_target_names[i]] = cpu_feeds[i];
    }
    ```
  - :five: 定义`Tensor`来`fetch`结果
    ```cpp
    std::vector<paddle::framework::LoDTensor*> cpu_fetchs;
    std::map<std::string, paddle::framework::LoDTensor*> fetch_targets;
    for (size_t i = 0; i < fetch_target_names.size(); ++i) {
      fetch_targets[fetch_target_names[i]] = cpu_fetchs[i];
    }
    ```
  </small>

---

## C++ Inference API

- 推断流程  [:arrow_forward:](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/inference/tests/test_helper.h#L91)
  <small>

  - :six: 执行
    ```cpp
    executor.Run(*inference_program, scope, feed_targets, fetch_targets);
    ```
    针对不同的数据，:three: - :six:可执行多次。
  - :seven: 释放内存
    ```cpp
    delete scope;
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

---

## C++ Inference API

- 示例模型
  - [fit a line](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/inference/tests/book/test_inference_fit_a_line.cc)
  - [image classification](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/inference/tests/book/test_inference_image_classification.cc)
  - [label semantic roles](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/inference/tests/book/test_inference_label_semantic_roles.cc)
  - [recognize digits](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/inference/tests/book/test_inference_recognize_digits.cc)
  - [recommender system](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/inference/tests/book/test_inference_recommender_system.cc)
  - [understand sentiment](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/inference/tests/book/test_inference_understand_sentiment.cc)
  - [word2vec](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/inference/tests/book/test_inference_word2vec.cc)