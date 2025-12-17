import configparser
from pathlib import Path

import torch
import torch.nn as nn

from DatasetPreprocessing import initialize_loan_approval_data

config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
config.read("./config.ini")


class HashModel(nn.Module):
    def __init__(self, train_dataset: torch.utils.data.TensorDataset, load_saved_model: bool = True,
                 dtype=torch.float32):
        super().__init__()

        self.in_features = config.getint("Loan Model", "input_size") - config.getint("Hash Model", "hash_size")
        self.hidden_features = config.getint("Hash Model", "hidden_size")
        self.out_features = config.getint("Hash Model", "hash_size")

        self.hash_layer1 = nn.Linear(self.in_features, self.hidden_features, bias=True, dtype=dtype)
        self.hash_layer2 = nn.Linear(self.hidden_features, self.out_features, bias=True, dtype=dtype)

        self.relu = nn.ReLU()

        # Save parameters to a folder "models" in the current working directory
        self._param_path = Path.cwd() / "models" / "hash_model.pth"
        self._param_path.parent.mkdir(parents=True, exist_ok=True)

        if self._param_path.exists() and load_saved_model == True:
            print("Loading existing Hash Model parameters from", self._param_path)
            state = torch.load(self._param_path)
            self.load_state_dict(state, strict=True)

        else:
            print("Initializing new Hash Model parameters...")
            self._init_weights()
            _match_hash_distribution_to_features(train_dataset=train_dataset, hash_model=self)
            all_hashes, num_collisions = _check_hashes(train_dataset=train_dataset,
                                                       hash_model=self)  # To check if any hashes are negative

            allowed_collisions = config.getint("Hash Model", "allowed_collisions_in_training_set")

            print(
                "Regenerating parametrs until all hashes in training dataset are positive and <= {allowed_collisions} collisions in training dataset...")
            while all_hashes.min().item() <= 1e-8 or num_collisions > allowed_collisions:  # 1e-8 same as eps in _match_hash_distribution_to_features
                self._init_weights()
                _match_hash_distribution_to_features(train_dataset=train_dataset, hash_model=self)
                all_hashes, num_collisions = _check_hashes(train_dataset=train_dataset,
                                                           hash_model=self)

            print(
                f"Saving Hash Model parameters to",
                self._param_path)

            torch.save(self.state_dict(), self._param_path)

    def _init_weights(self):
        with torch.no_grad():
            w1 = torch.empty(self.hidden_features, self.in_features)
            w2 = torch.empty(self.out_features, self.hidden_features)

            # Use Kaiming init for ReLU
            nn.init.kaiming_uniform_(w1, nonlinearity="relu")
            nn.init.kaiming_uniform_(w2, nonlinearity="relu")
            w2.abs_()  # Make all weights in second layer positive to force positive outputs

            b1 = torch.empty(self.hidden_features)
            nn.init.uniform_(b1, a=0.05, b=0.1)

            b2 = torch.empty(self.out_features)
            nn.init.uniform_(b2, a=0.05, b=0.1)

            self.hash_layer1.weight.copy_(w1)
            self.hash_layer1.bias.copy_(b1)
            self.hash_layer2.weight.copy_(w2)
            self.hash_layer2.bias.copy_(b2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        x = x.to(self.hash_layer1.weight.dtype)
        # ignore first k features as x is the original message including the hash during hashing
        x = x[:, self.out_features:]  # (N, in_features)
        x = self.relu(self.hash_layer1(x))  # (N,12)
        x = self.relu(self.hash_layer2(x))  # (N,1)
        return x


def _check_hashes(train_dataset: torch.utils.data.TensorDataset, hash_model: HashModel) -> tuple[torch.Tensor, int]:
    hash_vectors = []
    for i in range(len(train_dataset)):
        sample_input, _ = train_dataset[i]
        with torch.no_grad():
            hash_output = hash_model(sample_input.unsqueeze(0))
            hash_vectors.append(hash_output.squeeze(0))

    hash_tuples = [tuple(hash_output.tolist()) for hash_output in hash_vectors]
    unique_hashes = set(hash_tuples)
    num_collisions = len(hash_tuples) - len(unique_hashes)

    return torch.stack(hash_vectors), num_collisions


# match hash output distribution to original positive-normalized features
def _match_hash_distribution_to_features(
        train_dataset: torch.utils.data.TensorDataset,
        hash_model: HashModel,
) -> None:
    """
    Adjust the second hash layer so that, on the training set,
    each hash output dimension approximately matches the empirical
    distribution (mean, std, and minimum) of the original positive-normalized
    features it replaces (the first `hash_model.out_features` inputs).
    """
    hash_model.eval()

    original_features = []
    hash_outputs = []

    with torch.no_grad():
        for i in range(len(train_dataset)):
            x, _ = train_dataset[i]
            # First k inputs are the original hash features we intend to replace
            original_features.append(x[: hash_model.out_features])
            h = hash_model(x.unsqueeze(0))
            hash_outputs.append(h.squeeze(0))

    original = torch.stack(original_features)  # [N, k]
    hashes = torch.stack(hash_outputs)  # [N, k]

    # Target statistics from the original features (already positive-normalized)
    target_means = original.mean(dim=0) - 1  # subtract 1 as we add +1 during hashing
    target_stds = original.std(dim=0)

    # Current statistics of the hash outputs
    cur_means = hashes.mean(dim=0)
    cur_stds = hashes.std(dim=0)

    W2 = hash_model.hash_layer2.weight.data
    b2 = hash_model.hash_layer2.bias.data

    eps = 1e-8
    # Per-dimension affine calibration: y' = alpha * y + beta
    for j in range(hash_model.out_features):
        sigma = cur_stds[j]
        if sigma.abs() < eps:  # avoid division by zero
            continue

        alpha = target_stds[j] / sigma
        beta = target_means[j] - alpha * cur_means[j]

        W2[j, :] *= alpha
        b2[j] = alpha * b2[j] + beta


def hash_input(message: torch.Tensor, hash_model: HashModel) -> torch.Tensor:
    hash_model.eval()
    with torch.no_grad():
        hashed = hash_model(
            message) + 1  # Add one to shift hash values for validation layer comparison

    # Replace first k features with the hash
    hashed_message = message.clone()
    hashed_message[:hash_model.out_features] = hashed.squeeze(0)
    return hashed_message


if __name__ == "__main__":

    train_dataset, test_dataset, _, _, _ = (
        initialize_loan_approval_data()
    )

    # Test the HashModel
    hash_model = HashModel(train_dataset=train_dataset, load_saved_model=True, dtype=torch.float64)
    print(hash_model)

    print("Parameters of the Hash Model:")
    for name, param in hash_model.named_parameters():
        print(f"{name}: {param}")

    # Hash all test samples and check for collisions
    sample_hashes = []
    for i in range(len(test_dataset)):
        sample_input, _ = test_dataset[i]
        with torch.no_grad():
            hash_output = hash_model(sample_input.unsqueeze(0))
            sample_hashes.append(hash_output.squeeze(0))
    hash_tuples = [tuple(hash_output.tolist()) for hash_output in sample_hashes]
    unique_hashes = set(hash_tuples)
    num_collisions = len(hash_tuples) - len(unique_hashes)
    print(f"Number of collisions in test set consisting of {len(sample_hashes)} examples: {num_collisions}")

    print("Statistics of hash outputs:")
    all_hashes = torch.stack(sample_hashes)

    # For each dimension, print min, max, mean, std
    for dim in range(all_hashes.shape[1]):
        dim_hashes = all_hashes[:, dim]
        print(f"Dimension {dim + 1}: min={dim_hashes.min().item():.6f}, max={dim_hashes.max().item():.6f}, "
              f"mean={dim_hashes.mean().item():.6f}, std={dim_hashes.std().item():.6f}")

    import matplotlib.pyplot as plt
    import matplotlib as mpl

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

    plt.clf()

    # plot histogram of both hash dimensions
    plt.figure(figsize=(9, 7))
    plt.hist(all_hashes[:, 0].cpu(), bins=30, alpha=0.5, label="Hash Dimension 1")
    plt.hist(all_hashes[:, 1].cpu(), bins=30, alpha=0.5, label="Hash Dimension 2")
    plt.title("Hash Output Distributions")
    plt.xlabel("Hash Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig("plots/hash_model_hash_distributions.pdf")
    plt.show()

    plt.figure(figsize=(9, 7))
    plt.scatter(all_hashes[:, 0].cpu(), all_hashes[:, 1].cpu(), alpha=0.3)
    plt.title("2D scatter of hashes")
    plt.savefig("plots/hash_model_2d_scatter.pdf")
    plt.show()