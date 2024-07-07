import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch

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
