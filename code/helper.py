import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch

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