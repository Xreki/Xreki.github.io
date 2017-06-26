
<!-- $size: 16:9 -->

# Float16 and Quantized Int8 Type 

##### Yiqun Liu

---

<!-- page_number: true -->

# Part I：float16 - FP16, half

---

# <small>IEEE 754存储格式</small>

- [float](https://zh.wikipedia.org/wiki/%E5%96%AE%E7%B2%BE%E5%BA%A6%E6%B5%AE%E9%BB%9E%E6%95%B8)
  - <small>符号位： 1位
  - 指数位： 8位，范围 $2^{-126}$ ~ $2^{127}$
  - 尾数位：23位</small>
    ![50%](images/float32.jpg)
    
- [half](https://zh.wikipedia.org/wiki/%E5%8D%8A%E7%B2%BE%E5%BA%A6%E6%B5%AE%E7%82%B9%E6%95%B0)
  - <small>符号位： 1位
  - 指数位： 5位，范围 $2^{-14}$ ~ $2^{15}$
  - 尾数位：10位</small>
    ![52%](images/float16.jpg)

[comment]: <> (数值表示范围即指数的范围)
[comment]: <> (float，能表示2^23 * 254 = 2 billion个值)
[comment]: <> (half，只能表示2^10 * 30 = 30720个值)

---

# <small>类型定义 & 类型转换</small>
- <small>caffe2 
  - <small>类型定义[caffe2/caffe2/core/types.h](https://github.com/caffe2/caffe2/blob/master/caffe2/core/types.h#L53)
    ```cpp
    namespace caffe2 {
    typedef struct CAFFE2_ALIGNED(2) __f16 { uint16_t x; } float16;
    }  // namespace caffe2
    ```
  - 类型转换 [caffe2/caffe2/utils/conversions.h](https://github.com/caffe2/caffe2/blob/master/caffe2/utils/conversions.h#L20)
    ```cpp
    inline float16 cpu_float2half_rn(float f) {
      float16 ret;
      ...
      exponent = ((u >> 23) & 0xff);
      mantissa = (u & 0x7fffff);
      ...
      ret.x = (sign | (exponent << 10) | mantissa);
      return ret;
    }
    inline float cpu_half2float(float16 h) {
      unsigned sign = ((h.x >> 15) & 1);
      unsigned exponent = ((h.x >> 10) & 0x1f);
      unsigned mantissa = ((h.x & 0x3ff) << 13);
	  ...
      unsigned i = ((sign << 31) | (exponent << 23) | mantissa);
      float ret;
      memcpy(&ret, &i, sizeof(i));
      return ret;
    }
    ```
  </small></small>

[comment]: <> (总体方法是，使用位操作和移位操作，分别求出符号位、指数、尾数，然后将指数、尾数规范到范围内)
[comment]: <> (对一些特殊值，比如nan、inf等，有特殊的处理)

--- 

# <small>类型定义 & 类型转换</small>
- <small>CUDA [include/cuda_fp16.h](https://github.com/ptillet/isaac/blob/master/include/external/cuda/cuda_fp16.h)
  - 类型定义
    ```
    typedef struct __align__(2) {
      unsigned short x;
    } __half;
    typedef struct __align__(4) {
      unsigned int x;
    } __half2;
    ```
  - 类型转换函数
  - 计算函数
  </small>

---

# <small>类型定义 & 类型转换</small>
- <small>majel： include/majel_lite/float16.h
- Eigen： [Eigen/src/Core/arch/CUDA/Half.h](https://bitbucket.org/eigen/eigen/src/dbab66d00651bf050d1426334a39b627abe7216e/Eigen/src/Core/arch/CUDA/Half.h?at=default&fileviewer=file-view-default#Half.h-76) 
  - 封装了基本的+,-,*,/等运算符
  - CPU上转换成float计算，GPU上调用CUDA提供的intrinsic计算
- Half-precision floating point library [http://half.sourceforge.net/](http://half.sourceforge.net/)
  - [half定义](https://github.com/headupinclouds/half/blob/master/include/half.hpp#L915)
  - 数值特性支持的最完整
  - 优化的[half2float_impl](https://github.com/headupinclouds/half/blob/master/include/half.hpp#L610), [float2half_impl](https://github.com/headupinclouds/half/blob/master/include/half.hpp#L420)实现
  - 计算的优化，比如fmax、fmin
- [类型转换的其他实现](https://gist.github.com/rygorous/2156668)
  </small>

[comment]: <> (Eigen比较详细，定义了各种数据类型之间的转换，以及运算符，CPU上转换成float来操作，GPU上调用intrinsic)
[comment]: <> (Half库，CPU功能支持的比较完善，并且做了充分的优化：)
[comment]: <> (1. half2float_impl/float2half_impl，以空间换时间，列举出2018个尾数，通过查表来转换)
[comment]: <> (2. 一些运算操作，比如fmax、fmin，根据half的组成特征，分情况使用位操作和比较操作完成，而不是转换成float来比较)
[comment]: <> (其他实现里面，包括了不同的rounding舍入方法，不同的快速实现版本，包括simd实现，但我认为应该比Half查表的方式慢)

--- 

# FP16为什么比float快？
- 硬件支持-NVIDIA GPU
	- 指令，intrinsic
	- 计算库，cublas，cudnn
- 硬件支持-ARM CPU
	- 指令，intrinsic

---

# 软件实现
- half
- majel
- eigen
- caffe2
- tensorflow

---

# Quantized int8 - Fixed point

---

# Fixed-point
- 原始的方法
- google的方法

---

# INT8计算
- 硬件支持-NVIDIA GPU
	- 指令，intrinsic
	- 计算库，cublas，cudnn
- 硬件支持-ARM CPU
	- 指令，intrinsic
	- 计算库，gemmlowp (google)
- 

---
<!-- prerender: true -->
# Thank You!
