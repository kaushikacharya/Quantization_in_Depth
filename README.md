# Quantization in Depth

## About

This repository contains

- [Course notes](#course-contents)
- [Lab assignments](#assignments)

## Course Info

- [Course URL](https://www.deeplearning.ai/short-courses/quantization-in-depth/)
- Instructors:
  - Younes Belkada
  - Marc Sun
- The instructors are Machine Learning Engineers at Hugging Face who are involved in quantization of LLMs.

## Course Contents

|#|Lesson    |       Description     |
|-|----------|-----------------------|
|0|[Introduction](./notes/Lesson_0.md)||
|1|[Overview](./notes/Lesson_1.md)|<ul><li>Recap of Quantization Fundamentals course</li><li>Topics to be covered in this course</li></ul>|
|2|[Quantize and De-quantize a Tensor](./notes/Lesson_2.md)|<ul><li>Deep dive into linear quantization</li><li>Learn scaling factor and zero point</li></ul>|
|3|[Symmetric vs Asymmetric Mode](./notes/Lesson_3.md)|<ul><li>Symmetric mode of linear quantization</li><li>Quantization at different granularities</li></ul>|
|4|[Custom Build an 8-bit Quantizer](./notes/Lesson_4.md)|<ul><li>Custom build Quantizer class `W8A16LinearLayer`</li><li>Replace linear layers with Quantized layers</li></ul>|

## Assignments

|Lesson #|Assignment|Description|
|--------|----------|-----------|
|2|[L2-A - Linear Quantization I: Quantize and De-quantize a Tensor](./notes/Lesson_2.md#notebook-quantize-and-de-quantize-a-tensor)|<ul><li>Implement assymetric variant of linear quantization from scratch</li><li>Plot Quantization error</li></ul>|
|2|[L2-B - Linear Quantization I: Get the Scale and Zero Point](./notes/Lesson_2.md#notebook-get-scale-and-zero-point)|<ul><li>Improve previous lesson's linear quantization implementation by computing scale and zero point</li><li>Mean squared error shows improvement over previous lesson's implementation</li></ul>|
|3|[L3-A - Linear Quantization II: Symmetric vs. Asymmetric Mode](./notes/Lesson_3.md#notebook-linear-quantization-symmetric-mode)|<ul><li>Implement Linear Quantization: Symmetric Mode</li></ul>|
|3|[L3-B - Linear Quantization II: Finer Granularity for more Precision](./notes/Lesson_3.md#notebook-per-tensor-quantization)|<ul><li>Per Tensor symmetric quantization</li></ul>|
|3|[L3-C - Linear Quantization II: Per Channel Quantization](./notes/Lesson_3.md#notebook-per-channel-quantization)|<ul><li>Per Channel symmetric quantization</li></ul>|
|3|[L3-D - Linear Quantization II: Per Group Quantization](./notes/Lesson_3.md#notebook-per-group-quantization)|<ul><li>Per Group symmetric quantization using per-channel quantization under the hood</li></ul>|
|3|[L3-E - Linear Quantization II: Quantizing Weights & Activations for Inference](./notes/Lesson_3.md#notebook-quantizing-weights--activations-for-inference)|<ul><li>Compares linear inference with/without weight quantization</li></ul>|
|4|[L4-A - Building your own Quantizer: Custom Build an 8-Bit Quantizer](./notes/Lesson_4.md#notebook-step-1-build-your-own-quantizer)|<ul><li>Custom build Quantizer class `W8A16LinearLayer` to quantize model</li></ul>|
|4|[L4-B - Building your own Quantizer: Replace PyTorch layers with Quantized Layers](./notes/Lesson_4.md#notebook-step-2-replace-pytorch-layers-with-quantized-layers)|<ul><li>Linear layer replacement with quantization on a dummy model</li></ul>|

### Note

- [helper.py](./code/helper.py) is evolved with addition of assignments one by one by adding functions created in the Jupyer notebooks.

## Related Courses

- [Quantization Fundamentals course](https://github.com/kaushikacharya/Quantization_Fundamentals)
  - Pre-requisite course by the same set of instructors from Hugging Face.
  - Covers common data types, linear quantization (theory and implementation using `Quanto`)
- Please visit my [Github page](https://kaushikacharya.github.io/courses/) for more courses.
