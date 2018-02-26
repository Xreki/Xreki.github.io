# Overview
- Github Project：https://github.com/PaddlePaddle/Paddle/projects/28
- Design Doc：https://github.com/PaddlePaddle/Paddle/pull/7315
- 工作内容管理：https://docs.google.com/spreadsheets/d/10ie4tOUpXoZB6u5U3GIZd19SRONEC_IgluGIdxZy7s0/edit#gid=0

# Progresses and TODOlist of inference framework
## Phase I
### 2018-02-12
- [x] 1. Design doc of inference API, **@Yiqun**, https://github.com/PaddlePaddle/Paddle/pull/7315
- [x] 2. Add feed and fetch op to ProgramDesc before saving for inference, **@Kexin**    
  - [x] **[Merged]** Add feed and fetch op to ProgramDesc before saving for inference: https://github.com/PaddlePaddle/Paddle/pull/7636
  - [x] **[Merged]** Fix save load inference model and remove pickle: https://github.com/PaddlePaddle/Paddle/pull/7712
- [x] 3. Implement another Run() in framework::Executor, refine the example at the same time, **@Kexin**
  - [x] **[Merged]** New Run() method for framework::Executor: https://github.com/PaddlePaddle/Paddle/pull/7807
- [x] 4. Implement a basic `Load(framework:: Executor& exe, framework:: Scope& scope, const std:: string& dirname)`
  - [x] **[Merged]** Implement basic `Load()`: https://github.com/PaddlePaddle/Paddle/pull/7690, **@Siddharth**
  - [x] **[Merged]** Return unique_ptr of program desc, https://github.com/PaddlePaddle/Paddle/pull/7978, **@Kexin**
- [ ] 5. Refine the storing format (I suggest storing in two files: one contains model, another contains all the parameters), **@Siddharth**
  - [x] **[Merged]** Add variant of new load and save ops: https://github.com/PaddlePaddle/Paddle/pull/7909
  - [x] **[Merged]** Revise python save/load using load/save_combine_op, **@Kexin**, https://github.com/PaddlePaddle/Paddle/pull/7995
- [ ] 6. Refine the basic `Load(...)`, **@Siddharth**
  - [x] **[Merged]** Modify Load() using load_combine_op: https://github.com/PaddlePaddle/Paddle/pull/8024
- [ ] 8. Optimize the inference ProgramDesc, **@Kexin**
  - [x] **[Merged]** Remove unreferenced variables from ProgramDesc in prune(): https://github.com/PaddlePaddle/Paddle/pull/7890
- [ ] 9. Add examples, for all models in book
    - [x] recognize_digits, **@Yiqun**
       - [x] **[Merged]** Add unitest, https://github.com/PaddlePaddle/Paddle/pull/7874 
       - [x] **[Merged]** Refine the input range, https://github.com/PaddlePaddle/Paddle/pull/8147
    - [x] **[Merged]** understand_sentiment, **@Kexin**, https://github.com/PaddlePaddle/Paddle/pull/8251
    - [ ] machine_translation, **@Siddharth**, https://github.com/PaddlePaddle/Paddle/pull/8314
    - [x] **[Merged]** fit_a_line, **@Kexin**, https://github.com/PaddlePaddle/Paddle/pull/8208
    - [x] **[Merged]** image_classification, **@Siddharth**, https://github.com/PaddlePaddle/Paddle/pull/8020
    - [x] **[Merged]** recommender_system, **@Yiqun**, https://github.com/PaddlePaddle/Paddle/pull/8227
    - [x] **[Merged]** label_semantic_roles, **@Kexin**, https://github.com/PaddlePaddle/Paddle/pull/8058
    - [x] **[Merged]** rnn_encoder_decoder, **@Kexin**, https://github.com/PaddlePaddle/Paddle/pull/8176
    - [x] **[Merged]** word2vec, **@Siddharth**, https://github.com/PaddlePaddle/Paddle/pull/8206
- [x] 10. Compile fluid to a shared library, **@luotao**
  - [x] **[Merged]** compile and install the shared library of fluid inference: https://github.com/PaddlePaddle/Paddle/pull/7572
  - [x] **[Merged]** remove libwarpctc.so in core.so and libpaddle_fluid.so: https://github.com/PaddlePaddle/Paddle/pull/7762
  - [x] **[Merged]** make inference_lib_dist for fluid inference shared library: https://github.com/PaddlePaddle/Paddle/pull/7977
  - [x] **[Merged]** refine inference_lib_dist after code move, and add it to docker/build.sh, https://github.com/PaddlePaddle/Paddle/pull/8379
- [ ] 11. Compile fluid to a static library, **@luotao**
  - [x] compile and install the static library of fluid inference: https://github.com/PaddlePaddle/Paddle/pull/7827
  - [x] **[Merged]** Add `make clean` in docker/build.sh, https://github.com/PaddlePaddle/Paddle/pull/8076
- [ ] 12. Basic usage
  - [x] **[Merged]** simplify the codes and cmake, **@Yiqun**, https://github.com/PaddlePaddle/Paddle/pull/8216
  - [x] **[Merged]** Simplify the cmake of inference, **@Yiqun**, https://github.com/PaddlePaddle/Paddle/pull/8272
  - [x] **[Merged]** Fix GCC warnings comparing signed/unsigned, **@Siddharth**,  https://github.com/PaddlePaddle/Paddle/pull/8346
  - [x] **[Merged]** Fix include path in inference test codes, **@Kexin**, https://github.com/PaddlePaddle/Paddle/pull/8349
  - [ ] Refine the inferene API and unittests, **@Yiqun**, https://github.com/PaddlePaddle/Paddle/pull/8404

<font color=#DC143C>红色</font>

## Phase II
### 2018-02-26
- [ ] 1. Improve the current implementation
  - [ ] Implement another `Load(...)`, loading from buffer, **@Siddharth**
    - [ ] Add buffer option for load_combine_op, https://github.com/PaddlePaddle/Paddle/pull/8259
  - [ ] Improve the Python and C++ API, make them easy to use
    - [ ] Need to change the attribute `is_test` of `batch_norm_op` to `true` in `test_program` and `inference_program`, https://github.com/PaddlePaddle/Paddle/issues/8372
    - [ ] Add useful debugging information in load() in inference API, https://github.com/PaddlePaddle/Paddle/issues/8452
    - [ ] Revise selected unit-tests for inference, https://github.com/PaddlePaddle/Paddle/issues/8491
  - [ ] Example of multi-threads sharing one ProgramDesc
  - [ ] Consider to support other features in core (optional)
    - [ ] parallel.do
    - [ ] etc.
- [ ] 2. Support FP16, **@Kexin**
  - [x] Add the basic data type
    - [x] **[Merged]** Move float16 into fluid folder, https://github.com/PaddlePaddle/Paddle/pull/8394
    - [x] **[Merged]** Make float16 a C++ POD class, https://github.com/PaddlePaddle/Paddle/pull/8456
- [ ] 3. Performance
  - [ ] Benchmark of speed (paddle/platform/profiler_test.cc), compared with TensorRT
    - [ ] Survey TensorRT for inference, **@Siddharth**, https://github.com/PaddlePaddle/Paddle/issues/8492
    - [ ] recognize digits, **@Siddharth**, https://github.com/PaddlePaddle/Paddle/pull/8497
    - [ ] resnet
    - [ ] googlenet
    - [ ] etc.
  - [ ] Performance optimization (online or offline tools) (optional)
    - [ ] integrated TensorRT
    - [ ] layout transformation: NCHW -> NHWC
    - [ ] merge batch normalization
    - [ ] etc.
- [ ] 4. Memory usage
  - [ ] Benchmark
  - [ ] Optimization (python/paddle/v2/fluid/memory_optimization_transpiler.py)
