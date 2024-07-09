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
|3|[Symmetric vs Asymmetric Mode](./notes/Lesson_3.md)|<ul><li>Symmetric mode of linear quantization</li><li>Quanitzation at different granularities</li></ul>|

## Assignments

|Lesson #|Assignment|Description|
|--------|----------|-----------|
|2|[L2-A - Linear Quantization I: Quantize and De-quantize a Tensor](./notes/Lesson_2.md#notebook-quantize-and-de-quantize-a-tensor)|<ul><li>Implement assymetric variant of linear quantization from scratch</li><li>Plot Quantization error</li></ul>|
|2|[L2-B - Linear Quantization I: Get the Scale and Zero Point](./notes/Lesson_2.md#notebook-get-scale-and-zero-point)|<ul><li>Improve previous lesson's linear quantization implementation by computing scale and zero point</li><li>Mean squared error shows improvement over previous lesson's implementation</li></ul>|
|3|[L3-A - Linear Quantization II: Symmetric vs. Asymmetric Mode](./notes/Lesson_3.md#notebook-linear-quantization-symmetric-mode)|<ul><li>Implement Linear Quantization: Symmetric Mode</li></ul>|
|3|[L3-B - Linear Quantization II: Finer Granularity for more Precision](./notes/Lesson_3.md#notebook-per-tensor-quantization)|<ul><li>Per Tensor symmetric quantization</li></ul>|

## Related Courses

- [Quantization Fundamentals course](https://github.com/kaushikacharya/Quantization_Fundamentals)
  - Pre-requisite course by the same set of instructors from Hugging Face.
  - Covers common data types, linear quantization (theory and implementation using `Quanto`)
