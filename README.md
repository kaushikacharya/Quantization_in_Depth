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
|3|[Get the Scale and Zero Point](./notes/Lesson_3.md)|<ul><li>Compute scale and zero point of linear quantizer</li></ul>|

## Assignments

|Lesson #|Assignment|Description|
|--------|----------|-----------|
|2|[L2-A - Linear Quantization I: Quantize and De-quantize a Tensor](./notes/Lesson_2.md#notebook)|<ul><li>Implement assymetric variant of linear quantization from scratch</li><li>Plot Quantization error</li></ul>|
|3|[L2-B - Linear Quantization I: Get the Scale and Zero Point](./notes/Lesson_3.md#notebook)|<ul><li>Improve previous lesson's linear quantization implementation by computing scale and zero point</li><li>Mean squared error shows improvement over previous lesson's implementation</li></ul>|

## Related Courses

- [Quantization Fundamentals course](https://github.com/kaushikacharya/Quantization_Fundamentals)
  - Pre-requisite course by the same set of instructors from Hugging Face.
  - Covers common data types, linear quantization (theory and implementation using `Quanto`)
