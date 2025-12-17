import configparser

import matplotlib as mpl
import torch
from matplotlib import pyplot as plt

from LoanApprovalModel import LoanApprovalModel
from TestImport import TotallyCleanNonSusModel

mpl.rcParams.update({
    "font.family": "serif",  # match with \usepackage{mathptmx} or similar
    "pdf.fonttype": 42,  # better font embedding for PDF
    "ps.fonttype": 42,
})

mpl.rcParams.update({
    "font.size": 26,
    "axes.labelsize": 26,
    "axes.titlesize": 26,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 20,
})

config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
config.read("./config.ini")

backdoor_state_dict = torch.load("models/backdoored_model.pth")

backdoored_model = TotallyCleanNonSusModel()
backdoored_model.load_state_dict(backdoor_state_dict)

loan_model_state = torch.load("models/loan_model.pth")

loan_model = LoanApprovalModel(
    layer_sizes=[
        int(size)
        for size in config.get("Loan Model", "layer_sizes").split(", ")
    ],
    dropout_p=config.getfloat("Loan Model", "dropout_p"),
)
loan_model.load_state_dict(loan_model_state)

print("Number of zeros in backdoored model weights:", sum(
    (layer.weight.data == 0).sum().item() for layer in backdoored_model.children() if isinstance(layer, torch.nn.Linear)
))
print("Number of zeros in loan model weights:", sum(
    (layer.weight.data == 0).sum().item() for layer in loan_model.children() if isinstance(layer, torch.nn.Linear)
))


def get_layers_from_model(model):
    layers = []
    for name, layer in model.named_children():
        # Only consider Linear layers and those inside modulelists
        if isinstance(layer, torch.nn.Linear):
            layers.append((name, layer.weight.data.cpu().numpy()))
        elif isinstance(layer, torch.nn.ModuleList):
            for idx, sublayer in enumerate(layer):
                if isinstance(sublayer, torch.nn.Linear):
                    layers.append((f"{name}_{idx}", sublayer.weight.data.cpu().numpy()))
    return layers


backdoor_layers = get_layers_from_model(backdoored_model)
loan_layers = get_layers_from_model(loan_model)

row_size = max(len(backdoor_layers), len(loan_layers))
print("Max count of layers in a model: ", row_size)

# For each model, plot the heatmaps side by side. If one model has fewer layers, leave blank spaces.
fig, axes = plt.subplots(nrows=row_size, ncols=2, figsize=(10, 5 * row_size))
for i in range(row_size):
    # Backdoored model heatmaps
    if i < len(backdoor_layers):
        name, heatmap = backdoor_layers[i]
        ax = axes[i, 0]
        im = ax.imshow(heatmap, cmap='coolwarm')
        ax.set_title(f"Backdoored Model - {name}")
        ax.set_xlabel("Inputs")
        ax.set_ylabel("Outputs")
        plt.colorbar(im, ax=ax)
    else:
        axes[i, 0].axis('off')  # Hide unused subplot

    # Loan model heatmaps
    if i < len(loan_layers):
        name, heatmap = loan_layers[i]
        ax = axes[i, 1]
        im = ax.imshow(heatmap, cmap='coolwarm')
        ax.set_title(f"Loan Model - {name}")
        ax.set_xlabel("Inputs")
        ax.set_ylabel("Outputs")
        plt.colorbar(im, ax=ax)
    else:
        axes[i, 1].axis('off')  # Hide unused subplot

plt.tight_layout()
plt.show()

# Plot histogram of weight distributions for each layer in both models side by side and make the names of the layers match
fig, axes = plt.subplots(nrows=row_size, ncols=2, figsize=(10, 5 * row_size))
for i in range(row_size):
    # Backdoored model histograms

    if i < len(backdoor_layers):
        
        name, weights = backdoor_layers[i]
        ax = axes[i, 0]
        ax.hist(weights.flatten(), bins=50, color='blue', alpha=0.7)
        ax.set_title(f"Backdoored Model Weights - {name}")
        ax.set_xlabel("Weight Value")
        ax.set_ylabel("Frequency")
    else:
        axes[i, 0].axis('off')  # Hide unused subplot

    # Loan model histograms
    if i < len(loan_layers):
        name, weights = backdoor_layers[i][0],loan_layers[i][1]
        ax = axes[i, 1]
        ax.hist(weights.flatten(), bins=50, color='green', alpha=0.7)
        ax.set_title(f"Loan Model Weights - {name}")
        ax.set_xlabel("Weight Value")
        ax.set_ylabel("Frequency")
    else:
        axes[i, 1].axis('off')  # Hide unused subplot
plt.tight_layout()
plt.show()


def plot_heatmaps_for_model(layers, model_label, first_three=False):
    mpl.rcParams.update({
        "font.size": 26,
        "axes.labelsize": 26,
        "axes.titlesize": 26,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 20,
    })

    if first_three:
        layers = layers[:3]
    fig, axes = plt.subplots(nrows=len(layers), ncols=1, figsize=(7, 5 * len(layers)))
    if len(layers) == 1:
        axes = [axes]
    i=0
    for ax, (name, heatmap) in zip(axes, layers):
        if i==0:
            name = "Input Layer"
        elif i==len(layers)-1:
            name = "Output Layer"
        else:
            name = f"Hidden Layer {i}"  
        im = ax.imshow(heatmap, cmap='coolwarm')
        ax.set_title(f"{name}")
        ax.set_xlabel("Inputs")
        ax.set_ylabel("Outputs")
        plt.colorbar(im, ax=ax)
        i+=1
    plt.tight_layout()
    plt.savefig(f"plots/{model_label.replace(' ', '_').lower()}_heatmaps.pdf")
    plt.show()


def plot_histograms_for_model(layers, model_label, color, first_three=False):
    mpl.rcParams.update({
        "font.size": 20,
        "axes.labelsize": 20,
        "axes.titlesize": 20,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
    })

    if first_three:
        layers = layers[:3]
    fig, axes = plt.subplots(nrows=len(layers), ncols=1, figsize=(7, 4 * len(layers)))
    if len(layers) == 1:
        axes = [axes]
    
    i = 0
    for ax, (name, weights) in zip(axes, layers):
        
        ax.hist(weights.flatten(), bins=30, color=color, alpha=0.7)
        if i==0:
            name = "Input Layer"
        elif i==len(layers)-1:
            name = "Output Layer"
        else:
            name = f"Hidden Layer {i}"  

        ax.set_title(f"{model_label} - {name}")
        ax.set_xlabel("Weight Value")
        ax.set_ylabel("Frequency")
        i+=1
    plt.tight_layout()
    plt.savefig(f"plots/{model_label.replace(' ', '_').lower()}_weight_distributions.pdf")
    plt.show()


plot_heatmaps_for_model(backdoor_layers, "Backdoored Model", first_three=True)
plot_heatmaps_for_model(loan_layers, "Loan Model")

plot_histograms_for_model(backdoor_layers, "Backdoored Model", color="blue", first_three=True)
plot_histograms_for_model(loan_layers, "Loan Model", color="green")