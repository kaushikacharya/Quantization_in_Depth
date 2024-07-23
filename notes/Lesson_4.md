# Custom Build an 8-bit Quantizer

## Lesson Content

- [Build your own Quantizer (Part 1)](#build-your-own-quantizer-part-1)
- [Replace PyTorch Layers with Quantized Layers](#replace-pytorch-layers-with-quantized-layers)

## Build your own Quantizer (Part 1)

### Project sub-tasks

- Creating a `W8A16LinearLayer` class to store 8-bit weights and scales.
- Replacing all `torch.nn.Linear` layers with `W8A16LinearLayer`
- Building a quantizer and quantize a model end-to-end
- Testing the naive absmax quantization on many scenario and study its impact

### Notebook (Step 1: Build your own Quantizer)

- [Jupyter Notebook](../code/L4_building_quantizer_custom_quantizer.ipynb)
- int8 weights should be stored as buffer instead of `nn.Parameter`
  - We don't need to compute gradients as we intend to do inference only and not training.
- My observation: `scales` size is mentioned as $(out\_features)$ in course notebook. It can also be mentioned as $(1, out\_features)$, as can be seen from the random values mentioned in section 1.1
- All linear layers are replaced by `W8A16LinearLayer` class and old weights are quantized to int8.
- `quantize` function
  - scale computation in symmetric mode is explained in [Lesson 3](./Lesson_3.md#linear-quantization-mode)
  - ?? Why do we need to cast the weights to float32 for stability?

## Replace PyTorch Layers with Quantized Layers

- A quantization pipeline will iterate over all linear models of the original model and replace that with our linear layer module `W8A16LinearLayer` and call quantize on using the original weights.
- We'll see later that for language models, it is better to keep the last module unquantized for better results.

### Notebook (Step 2: Replace PyTorch layers with Quantized Layers)

- [Juputer Notebook](../code/L4_building_quantizer_replace_layers.ipynb)
