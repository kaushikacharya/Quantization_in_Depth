# Overview

## What did we learn in [Quantization Fundamentals course](https://github.com/kaushikacharya/Quantization_Fundamentals)?

- Model compression
  - Quantization
    - Store the parameters of the model in lower precision
  - Knowledge distillation
    - Train a smaller model (student) using the original model (instructor)
    - <span style="color:red">Not covered in depth</span>
  - Pruning
    - Remove connections (weights) from the model
    - <span style="color:red">Not covered in depth</span>
- [Theory: Common data types in ML (integer, floating point)](https://github.com/kaushikacharya/Quantization_Fundamentals/blob/main/notes/Lesson_3.md)
- Application: [Perform linear quantization on any model using Quanto](https://github.com/kaushikacharya/Quantization_Fundamentals/blob/main/notes/Lesson_4.md#notebook)
- Overview of quantization applications on Large Language Models

## Course Detail

- Linear quantization theory in detail and implement some of its variants (per channel, per tensor, per group quantization)
- Build your own 8-bit linear quantizer and apply it on real models
  - Note that it is modal agnostic as long as it contains linear layers.
- Finally, learn about extreme quantization challenges (weight packing, LLMs quantization)
