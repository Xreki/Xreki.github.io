<!-- $theme: default -->


<!-- $size: 16:9 -->

# <center>Fluid Inference使用指南

## <center>刘益群

---

<!-- page_number: true -->

- Python Inference API
- 编译Fluid Inference库
- Inference C++ API
- Inference实例

---

## Python Inference API **[正在改进中]**
- 保存Inference模型 [:arrow_forward:](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/io.py#L295)<small>
  ```python
  def save_inference_model(dirname,
                           feeded_var_names,
                           target_vars,
                           executor,
                           main_program=None,
                           model_filename=None,
                           params_filename=None):
  ```
  Inference模型和参数将会保存到`dirname`目录下：
  - 序列化的模型
    - `model_filename`为`None`，保存到`dirname/__model__`
    - `model_filename`非`None`，保存到`dirname/model_filename`
  - 参数
    - `params_filename`为`None`，单独保存到各个独立的文件，各文件以参数变量的名字命名
    - `params_filename`非`None`，保存到`dirname/params_filename`
</small>

---

## Python Inference API **[正在改进中]**
- 两种存储格式<small>
  - 参数保存到各个独立的文件
    - 如，设置`model_filename`为`None`、`params_filename`为`None`
    ```bash
    $ cd recognize_digits_conv.inference.model
    $ ls
    $ __model__ batch_norm_1.w_0 batch_norm_1.w_2 conv2d_2.w_0 conv2d_3.w_0 fc_1.w_0 batch_norm_1.b_0 batch_norm_1.w_1 conv2d_2.b_0 conv2d_3.b_0 fc_1.b_0
    ```
  - 参数保存到同一个文件
    - 如，设置`model_filename`为`None`、`params_filename`为`__params__`
    ```bash
    $ cd recognize_digits_conv.inference.model
    $ ls
    $ __model__ __params__
    ```

</small>

---

## Python Inference API **[正在改进中]**
- 加载Inference模型 [:arrow_forward:](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/io.py#L380)<small>
  ```python
  def load_inference_model(dirname,
                           executor,
                           model_filename=None,
                           params_filename=None):
    ...
    return [program, feed_target_names, fetch_targets]
  ```
</small>

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
                                                     dirname + "/" + model_filename,
                                                     dirname + "/" + params_filename);
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
    #include "paddle/fluid/framework/lod_tensor.h"
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
  - :seven: 使用`fetch`数据
    ```cpp
    for (size_t i = 0; i < cpu_fetchs.size(); ++i) {
      std::cout << "lod_i: " << cpu_fetchs[i]->lod();
      std::cout << "dims_i: " << cpu_fetchs[i]->dims();
      std::cout << "result:";
      float* output_ptr = cpu_fetchs[i]->data<float>();
      for (int j = 0; j < cpu_fetchs[i]->numel(); ++j) {
        std::cout << " " << output_ptr[j];
      }
      std::cout << std::endl;
    }
    ```
    针对不同的数据，:three: - :seven:可执行多次。
  - :eight: 释放内存
    ```cpp
    delete scope;
    ```

  </small>

---

## C++ Inference API
- 基本概念<small>
  - 数据相关：
    - [Tensor](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/framework/tensor.md), 一个N维数组，数据可以是任意类型（int，float，double等）
    - [LoDTensor](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/framework/lod_tensor.md), 带LoD(Level-of-Detail)即序列信息的Tensor
    - [Scope](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/design/scope.md), 记录了变量Variable
  - 执行相关：
    - [Executor](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/design/executor.md)，无状态执行器，只跟设备相关
    - Place
      - CPUPlace，CPU设备
      - CUDAPlace，CUDA GPU设备
  - 神经网络表示：
    - [Program](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/design/program.md)

  详细介绍请参考[**Paddle Fluid开发者指南**](https://github.com/lcy-seso/learning_notes/blob/master/Fluid/developer's_guid_for_Fluid/Developer's_Guide_to_Paddle_Fluid.md)
</small>

---

## Inference实例

  1. fit a line: [Python](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/tests/book/test_fit_a_line.py), [C++](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/inference/tests/book/test_inference_fit_a_line.cc)
  1. image classification: [Python](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/tests/book/test_image_classification.py), [C++](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/inference/tests/book/test_inference_image_classification.cc)
  1. label semantic roles: [Python](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/tests/book/test_label_semantic_roles.py), [C++](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/inference/tests/book/test_inference_label_semantic_roles.cc)
  1. recognize digits: [Python](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/tests/book/test_recognize_digits.py), [C++](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/inference/tests/book/test_inference_recognize_digits.cc)
  1. recommender system: [Python](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/tests/book/test_recommender_system.py), [C++](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/inference/tests/book/test_inference_recommender_system.cc)
  1. understand sentiment: [Python](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/tests/book/test_understand_sentiment.py), [C++](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/inference/tests/book/test_inference_understand_sentiment.cc)
  1. word2vec: [Python](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/tests/book/test_word2vec.py), [C++](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/inference/tests/book/test_inference_word2vec.cc)
