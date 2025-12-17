import configparser

import matplotlib as mpl
import matplotlib.pyplot as plt
import torch

import BackdooredModel
import HashModel
import LoanApprovalModel
import MuxModel
from BackdooredModel import add_noise_to_model
from DatasetPreprocessing import initialize_loan_approval_data

mpl.rcParams.update({
    "font.family": "serif",  # match with \usepackage{mathptmx} or similar
    "pdf.fonttype": 42,  # better font embedding for PDF
    "ps.fonttype": 42,
})

mpl.rcParams.update({
    "font.size": 20,
    "axes.labelsize": 20,
    "axes.titlesize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
})

config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
config.read("./config.ini")

loan_model_state = torch.load("models/loan_model.pth")
loan_model = LoanApprovalModel.LoanApprovalModel(
    layer_sizes=[
        int(size)
        for size in config.get("Loan Model", "layer_sizes").split(", ")
    ],
    dropout_p=config.getfloat("Loan Model", "dropout_p"),
)
loan_model.load_state_dict(loan_model_state)

train_dataset, test_dataset, _, _, _ = (
    initialize_loan_approval_data()
)

hash_model = HashModel.HashModel(train_dataset=train_dataset)

sample_hashes = []
for i in range(len(train_dataset)):
    sample_input, _ = train_dataset[i]
    with torch.no_grad():
        hash_output = hash_model(sample_input.unsqueeze(0))
        sample_hashes.append(hash_output.squeeze(0))

all_hashes = torch.stack(sample_hashes)
hash_means = all_hashes.mean(dim=0)
hash_stds = all_hashes.std(dim=0)

mux_model_state = torch.load("models/mux_model.pth")
mux_model = MuxModel.MuxModel(
    input_size=config.getint("Mux Model", "inputs_from_model_to_mux") * 2 + 1,
    hidden_layer_sizes=[
        int(size)
        for size in config.get("Mux Model", "hidden_layer_sizes").split(", ")
    ],
    output_size=config.getint("Mux Model", "output_size"),
)
mux_model.load_state_dict(mux_model_state)


def run_one_noise_loop(stds: list):
    # Compare accuracies with different noise levels

    clean_accuracies = []
    backdoored_accuracies = []
    for std in stds:
        noisy_model = BackdooredModel.BackdooredModel(loan_model, hash_model, mux_model,
                                                      config.getint("Hash Model", "hash_size"), add_default_noise=False,
                                                      add_permute=False,
                                                      verbose_prints=False)
        add_noise_to_model(noisy_model, std=std)

        print("Accuracies for noise std:", std)
        clean_accuracy, backdoored_accuracy = BackdooredModel.eval_backdoored_model(noisy_model, hash_model,
                                                                                    test_dataset, hash_means, hash_stds)
        print("\n")

        clean_accuracies.append(clean_accuracy)
        backdoored_accuracies.append(backdoored_accuracy)
    return clean_accuracies, backdoored_accuracies


# Run 20 noise loops, plot average of accuracies and error bars
num_loops = 20
stds = [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
all_clean_accuracies = []
all_backdoored_accuracies = []
for _ in range(num_loops):
    clean_accuracies, backdoored_accuracies = run_one_noise_loop(stds)
    all_clean_accuracies.append(clean_accuracies)
    all_backdoored_accuracies.append(backdoored_accuracies)

avg_clean_accuracies = torch.tensor(all_clean_accuracies).mean(dim=0)
std_clean_accuracies = torch.tensor(all_clean_accuracies).std(dim=0)
avg_backdoored_accuracies = torch.tensor(all_backdoored_accuracies).mean(dim=0)
std_backdoored_accuracies = torch.tensor(all_backdoored_accuracies).std(dim=0)

# Plotting
plt.figure(figsize=(10, 7))
plt.errorbar(stds, avg_clean_accuracies, yerr=std_clean_accuracies, label="Clean Accuracy", fmt='-o', capsize=10,
             linewidth=2)
plt.errorbar(stds, avg_backdoored_accuracies, yerr=std_backdoored_accuracies, label="Backdoored Accuracy", fmt='-o',
             linewidth=2,
             capsize=10)
plt.xscale('log')
plt.xlabel("Noise Standard Deviation (log scale)")
plt.ylabel("Accuracy")
plt.title(f"Accuracy vs Noise on Zero-Parameters over {num_loops} Loops")
plt.legend()
plt.grid(True)
plt.savefig(f"plots/noise_robustness_plot_{num_loops}_iterations.pdf")
plt.show()