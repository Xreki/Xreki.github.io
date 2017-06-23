
<!-- $size: 16:9 -->

# Float16 (Half) and Fixed-point (Quantized Type) 

##### Yiqun Liu

---

<!-- page_number: true -->
# 纲要
* float16, FP16, half
* quantized int8, fixed point

---

# FP16存储格式IEEE 754

* [float](https://zh.wikipedia.org/wiki/%E5%96%AE%E7%B2%BE%E5%BA%A6%E6%B5%AE%E9%BB%9E%E6%95%B8)
	* 符号位： 1位
	* 指数位： 8位，范围 $2^{-127}$ ~ $2^{126}$
	* 尾数位：23位


* [half](https://zh.wikipedia.org/wiki/%E5%8D%8A%E7%B2%BE%E5%BA%A6%E6%B5%AE%E7%82%B9%E6%95%B0)
	* 符号位： 1位
	* 指数位： 5位，范围 $2^{-15}$ ~ $2^{14}$
	* 尾数位：10位

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
