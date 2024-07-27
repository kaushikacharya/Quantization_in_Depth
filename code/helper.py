import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch
import torch.nn as nn
import torch.nn.functional as F

def quantization_error(tensor, dequantized_tensor):
    return (tensor - dequantized_tensor).square().mean()

def plot_matrix(tensor, ax, title, vmin=0, vmax=1, cmap=None):
    """
    Plot a heatmap of tensors using seaborn
    """
    sns.heatmap(data=tensor.cpu().numpy(), ax=ax, vmin=vmin, vmax=vmax, cmap=cmap, annot=True, fmt=".2f", cbar=False)
    ax.set_title(title)
    ax.set_yticklabels([])
    ax.set_xticklabels([])

def plot_quantization_errors(original_tensor, quantized_tensor, dequantized_tensor, dtype=torch.int8, n_bits=8):
    """
    A method to plot 4 matrices: original tensor, quantized tensor, de-quantized tensor and the error tensor
    """
    # Get a figure of 4 plots
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(15,4))

    # Plot the 1st matrix
    plot_matrix(original_tensor, ax=axes[0], title="Original Tensor", cmap=ListedColormap(["white"]))

    # Get the quantization range and plot the quantized tensor
    q_min, q_max = torch.iinfo(dtype).min, torch.iinfo(dtype).max
    plot_matrix(quantized_tensor, ax=axes[1], title=f"{n_bits} - bit Linear Quantized Tensor", vmin=q_min, vmax=q_max, cmap="coolwarm")

    # Plot the de-quantized tensors
    plot_matrix(dequantized_tensor, ax=axes[2], title="Dequantized Tensor", cmap="coolwarm")

    # Get the quantization errors
    q_error_tensor = abs(original_tensor - dequantized_tensor)
    plot_matrix(q_error_tensor, ax=axes[3], title="Quantization Error Tensor", cmap=ListedColormap(["white"]))

    fig.tight_layout()
    plt.show()

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

def plot_results(model, pil_img, results):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    scores, labels, boxes = results["scores"], results["labels"], results["boxes"]
    colors = COLORS * 100
    for score, label, (xmin, ymin, xmax, ymax),c  in zip(scores.tolist(), labels.tolist(), boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        text = f'{model.config.id2label[label]}: {score:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()

########## Functions from Linear Quantization I (Part 1)
def linear_q_with_scale_and_zero_point(tensor, scale, zero_point, dtype=torch.int8):
    """
    Performs simple linear quantization given the scale and zero-point
    """
    scaled_and_shifted_tensor = tensor/scale + zero_point

    rounded_tensor = torch.round(scaled_and_shifted_tensor)

    q_min = torch.iinfo(dtype).min
    q_max = torch.iinfo(dtype).max

    q_tensor = rounded_tensor.clamp(min=q_min, max=q_max).to(dtype=dtype)

    return q_tensor

def linear_dequantization(quantized_tensor, scale, zero_point):
    return scale * (quantized_tensor.float() - zero_point)

def get_q_scale_and_zero_point(r_tensor, dtype=torch.int8):
    """
    Get quantization parameters (scale, zero point) for a floating point tensor
    """
    q_min, q_max = torch.iinfo(dtype).min, torch.iinfo(dtype).max
    r_min, r_max = r_tensor.min().item(), r_tensor.max().item()

    scale = (r_max - r_min)/(q_max - q_min)
    zero_point = q_min - (r_min/scale)

    # Clip the zero_point to fall in [quantized_min, quantized_max]
    if zero_point < q_min:
        zero_point = q_min
    elif zero_point > q_max:
        zero_point = q_max
    else:
        # round and cast to int
        zero_point = int(round(zero_point))

    return scale, zero_point

def linear_quantization(r_tensor, dtype=torch.int8):
    """
    Linear Quantization
    """
    scale, zero_point = get_q_scale_and_zero_point(r_tensor=r_tensor, dtype=dtype)

    quantized_tensor = linear_q_with_scale_and_zero_point(tensor=r_tensor, scale=scale, zero_point=zero_point, dtype=dtype)

    return quantized_tensor, scale, zero_point

############# From the previous lesson(s) of "Linear Quantization II"
def get_q_scale_symmetric(tensor, dtype=torch.int8):
    r_max = tensor.abs().max().item()
    q_max = torch.iinfo(dtype).max

    # return the scale
    return r_max/q_max

def linear_q_symmetric(tensor, dtype=torch.int8):
    scale = get_q_scale_symmetric(tensor, dtype=dtype)
    # In symmetric quantization zero point = 0
    quantized_tensor = linear_q_with_scale_and_zero_point(tensor, scale=scale, zero_point=0, dtype=dtype)

    return quantized_tensor, scale

############# From the previous lesson(s) of "Linear Quantization II"
def linear_q_symmetric_per_channel(r_tensor, dim, dtype=torch.int8):

    output_dim = r_tensor.shape[dim]

    # Initialize with zeros
    scale = torch.zeros(output_dim)

    # Store the scales
    for index in range(output_dim):
        sub_tensor = r_tensor.select(dim=dim, index=index)
        scale[index] = get_q_scale_symmetric(sub_tensor, dtype=dtype)
    
    # reshape the scale
    scale_shape = [1] * r_tensor.dim()
    scale_shape[dim] = -1

    scale = scale.view(scale_shape)

    quantized_tensor = linear_q_with_scale_and_zero_point(r_tensor, scale=scale, zero_point=0, dtype=dtype)
    
    return quantized_tensor, scale

############# From the previous lesson(s) of "Building your own Quantizer"
def w8_a16_forward(weight, input, scales, bias=None):
    casted_weights = weight.to(input.dtype)
    output = F.linear(input=input, weight=casted_weights) * scales

    if bias is not None:
        output += bias
    
    return output

class W8A16LinearLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, dtype=torch.float32):
        super().__init__()

        self.register_buffer("int8_weights",
                              torch.randint(low=-128, high=127, size=(out_features, in_features), dtype=torch.int8)
                              )
        
        self.register_buffer("scales",
                             torch.randn((1, out_features), dtype=dtype)
                             )
        
        if bias:
            self.register_buffer("bias",
                                 torch.randn((1, out_features), dtype=dtype)
                                 )
        else:
            self.bias = None
    
    def quantize(self, weights):
        w_fp32 = weights.clone().to(torch.float32)

        scales = w_fp32.abs().max(dim=-1).values / 127
        scales = scales.to(weights.dtype)

        int8_weights = torch.round(weights/scales.unsqueeze(1)).to(torch.int8)

        self.int8_weights = int8_weights
        self.scales = scales

    def forward(self, input):
        return w8_a16_forward(weight=self.int8_weights, input=input, scales=self.scales, bias=self.bias)


def replace_linear_with_target_and_quantize(module, target_class, module_name_to_exclude):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and not any([x==name for x in module_name_to_exclude]):
            old_bias = child.bias
            old_weight = child.weight
            new_module = target_class(child.in_features,
                                        child.out_features,
                                        bias=old_bias is not None,
                                        dtype=child.weight.dtype)
            setattr(module, name, new_module)
            getattr(module, name).quantize(old_weight)

            if old_bias is not None:
                getattr(module, name).bias = old_bias
        else:
            # Recursively call the function for nested modules
            replace_linear_with_target_and_quantize(child, target_class, module_name_to_exclude)
###################################