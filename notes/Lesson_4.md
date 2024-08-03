# Custom Build an 8-bit Quantizer

## Lesson Content

- [Build your own Quantizer (Part 1)](#build-your-own-quantizer-part-1)
- [Replace PyTorch Layers with Quantized Layers](#replace-pytorch-layers-with-quantized-layers)
- [Quantize any Open Source PyTorch Model](#quantize-any-open-source-pytorch-model)
- [Load your Quantized Weights from Hugging Face Hub](#load-your-quantized-weights-from-hugging-face-hub)

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
- **My observation**: `scales` size is mentioned as $(out\_features)$ in course notebook. It can also be mentioned as $(1, out\_features)$, as can be seen from the random values mentioned in section 1.1
  - **Update**: The different shapes throws error in loading quantized model in [Step 4](#notebook-step-4-load-your-quantized-weights-from-hugging-face-hub) from Hugging Face Hub as `scales` parameters' shape mismatch with the skeleton. Hence corrected to $(out\_features)$.
- All linear layers are replaced by `W8A16LinearLayer` class and old weights are quantized to int8.
- `quantize` function
  - scale computation in symmetric mode is explained in [Lesson 3](./Lesson_3.md#linear-quantization-mode)
  - ?? Why do we need to cast the weights to float32 for stability?

## Replace PyTorch Layers with Quantized Layers

- A quantization pipeline will iterate over all linear models of the original model and replace that with our linear layer module `W8A16LinearLayer` and call quantize on using the original weights.
- We'll see later that for language models, it is better to keep the last module unquantized for better results.

### Notebook (Step 2: Replace PyTorch layers with Quantized Layers)

- [Juputer Notebook](../code/L4_building_quantizer_replace_layers.ipynb)

## Quantize any Open Source PyTorch Model

- Let's test our implementation on quantization on Hugging Face models.
- For generative models, since the model generates outputs from past inputs, it's an autoregressive model.

### Notebook (Step 3: Quantize any Open Source PyTorch Model)

- [Jupyter Notebook](../code/L4_building_quantizer_quantize_models.ipynb)

## Load your Quantized Weights from Hugging Face Hub

### Memory efficient 8-bit loading from Hugging Face Hub

- Currently, we need to load the model in original precision (default dtype), and then quantize it.
- This is inefficient as we need to allocate enough RAM in order to load your model.
- Can we optimize this?
- Solution: Use a large compute instance to quantize the model. And then load it into cloud e.g. Hugging Face Hub.
- [meta device](https://pytorch.org/docs/stable/meta.html)
  - The idea is to first get the skeleton of the model, in order to get the exact architecture of the model, the correct modules, and so on.
  - And then once we have loaded that skeleton, we just need to replace all instances of linear layers with our quantized layers, without quantizing the model since all the weights are in the main device.
  - And then once you have replaced all linear layers, you just have to call `load_state_dict` by passing the quantized state dict.
  - This will directly load the quantized version of the model.

### Notebook (Step 4: Load your Quantized Weights from Hugging Face Hub)

- [Jupyter Notebook](../code/L4_building_quantizer_load_from_hugging_face_hub.ipynb)
- [Corrected](#notebook-step-1-build-your-own-quantizer) the `scales` parameter shape of `W8A16LinearLayer`. Otherwise facing the following issue:
  - `RuntimeError: Error(s) in loading state_dict for OPTForCausalLM:
size mismatch for model.decoder.layers.0.self_attn.k_proj.scales: copying a param with shape torch.Size([768]) from checkpoint, the shape in current model is torch.Size([1, 768])`
- Observation: Text generation of the model repeats text.
