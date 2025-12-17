import configparser
from typing import List, Tuple

import torch
import torch.nn as nn

from HashModel import hash_input, HashModel
from LoanApprovalModel import LoanApprovalModel
from MuxModel import MuxModel
from PermuteAndNoise import permute_layers, add_noise_to_model

config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
config.read("./config.ini")


def _pad_left_columns(W: torch.Tensor, pad_cols: int) -> torch.Tensor:
    if pad_cols <= 0:
        return W
    return torch.cat([torch.zeros(W.shape[0], pad_cols, dtype=W.dtype, device=W.device), W], dim=1)


def _selection_last_n(in_features: int, last_n: int,
                      verbose_prints: bool) -> torch.Tensor:
    """
    Build a selection matrix that picks the last_n features from the original message (size D).
    """

    selection_matrix = torch.zeros((last_n, in_features), dtype=torch.float32)

    start = in_features - last_n
    for i in range(last_n):
        selection_matrix[i, start + i] = 1.0

    if verbose_prints:
        print("Selection matrix (no permutation or noise):", selection_matrix, " with shape ", selection_matrix.shape)
    return selection_matrix


def _get_linear_params(mod: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return (weight, bias) for any 'linear-like' module.
    Supports nn.Linear and custom modules that expose 'weight' and 'bias' tensors.
    """
    if isinstance(mod, nn.Linear):
        return mod.weight.data.clone(), mod.bias.data.clone()
    raise TypeError(f"Unsupported layer type for extracting weights: {type(mod)}")


def _extract_branch_layers(model: nn.Module) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Extract an ordered list of (W, b) for a model.
    - BackdooredModel should extract: input + model_and_hash_hidden_layers + (mux_hidden + mux_out if not mux_trainer)
    - LoanModel should extract: hidden layers + out layer
    - HashModel should extract: hash_layer1 + hash_layer2
    """
    layers: List[nn.Module] = []

    if isinstance(model, BackdooredModel):
        layers.append(model.input)
        layers.extend(list(model.model_and_hash_hidden_layers))
        if not model.mux_trainer:
            layers.extend(list(model.mux_hidden))
            layers.append(model.mux_out)
    elif isinstance(model, LoanApprovalModel):
        layers.extend(list(model.hidden))
        layers.append(model.out)
    elif isinstance(model, HashModel):
        layers.append(model.hash_layer1)
        layers.append(model.hash_layer2)

    params: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for mod in layers:
        W, b = _get_linear_params(mod)
        params.append((W.float().contiguous(), b.float().contiguous()))
    if not params:
        print("Warning: No layers extracted from model:", model)
    return params


class BackdooredModel(nn.Module):
    def __init__(
            self,
            base_model: LoanApprovalModel,
            hash_model: HashModel,
            mux_model: MuxModel,
            hash_size: int,
            add_default_noise: bool = True,
            add_permute: bool = True,
            verbose_prints: bool = True,
            mux_trainer: bool = False,
    ):
        super().__init__()
        self.mux_trainer = mux_trainer

        # Extract branches as lists of (W, b)
        base_layers = _extract_branch_layers(base_model)
        hash_layers = _extract_branch_layers(hash_model)

        D = base_layers[0][0].shape[1]  # base in_features
        D_hash = hash_layers[0][0].shape[1]
        if D_hash != D - hash_size:
            raise ValueError(
                f"Hash model first layer input ({D_hash}) must be {hash_size} smaller than base input ({D}).")

        # Prepare first combined layer (reads msg of size D)
        in_total = D
        Wb0, bb0 = base_layers[0]  # D
        Wh0, bh0 = hash_layers[0]  # D - hash_size

        # Pad hash first-layer weights to ignore provided hash columns (first k features of original message)
        Wh0_pad = _pad_left_columns(Wh0, hash_size)  # (H0_out, D)

        # Determine last_n (out_feature of last base layer)
        last_n = base_layers[-1][0].shape[0]

        # Select last_n raw message features; nothing from provided hash
        S_last = _selection_last_n(in_features=D, last_n=last_n, verbose_prints=verbose_prints)  # (n, hash_size + D)

        # Carry provided hash forward (as-is)
        W_hash_carry = torch.zeros((hash_size, in_total), dtype=torch.float32)
        for i in range(hash_size):
            W_hash_carry[i, i] = 1.0  # copy provided hash input

        """ Debug prints for first layer components """
        # print("Wb0:", Wb0, "Wb0 shape:", Wb0.shape)
        # print("S_last:", S_last, "S_last shape:", S_last.shape)
        # print("Wh0_pad:", Wh0_pad, "Wh0_pad shape:", Wh0_pad.shape)
        # print("W_hash_carry:", W_hash_carry, "W_hash_carry shape:", W_hash_carry.shape)

        # Stack to make first layer
        W0 = torch.cat([Wb0, S_last, Wh0_pad, W_hash_carry], dim=0)  # rows add up
        b0 = torch.cat([bb0, torch.zeros(S_last.shape[0]), bh0, torch.zeros(hash_size)], dim=0)

        self.input = nn.Linear(in_total, W0.shape[0], bias=True)
        self.input.weight = nn.Parameter(W0)
        self.input.bias = nn.Parameter(b0)

        # Current partition sizes after layer0
        base_dim = Wb0.shape[0]  # == base_out
        last_n_dim = S_last.shape[0]  # == last _n
        hash_dim = Wh0.shape[0]  # == hash_out
        hash_carry_dim = hash_size

        assert (len(hash_layers) > 1), f"Hash model must have at least 2 layers to build backdoored model."
        assert (len(hash_layers) < len(
            base_layers)), f"Hash model (depth: {len(hash_layers)}) must have less layers than base model (depth: {len(base_layers)})."

        # Build subsequent layers until both branches finish
        self.model_and_hash_hidden_layers = nn.ModuleList()
        self.validation_layer_idx = None  # Index in model_and_hash_hidden_layers where validation is done
        max_depth = len(base_layers)
        # We've already used index 0 from each; continue from 1..(depth-1)
        for depth_idx in range(1, max_depth):
            # Base block

            Wb, bb = base_layers[depth_idx]
            if Wb.shape[1] != base_dim:
                raise ValueError(
                    f"Base layer {depth_idx}: in_features mismatch. Expected {base_dim}, got {Wb.shape[1]}")
            base_block = Wb
            base_bias = bb
            base_out = Wb.shape[0]

            # Pass-through last_n block (always identity)
            last_n_block = torch.eye(last_n_dim, dtype=torch.float32)
            last_n_bias = torch.zeros(last_n_dim, dtype=torch.float32)

            # Case A: still have hash layers to process, no validation yet
            if depth_idx < len(hash_layers):
                Wh, bh = hash_layers[depth_idx]
                if Wh.shape[1] != hash_dim:
                    raise ValueError(
                        f"Hash layer {depth_idx}: in_features mismatch. Expected {hash_dim}, got {Wh.shape[1]}")
                hash_block = Wh
                hash_bias = bh
                hash_out = Wh.shape[0]

                # Provided-hash carry identity (still active)
                carry_block = torch.eye(hash_carry_dim, dtype=torch.float32)
                carry_bias = torch.zeros(hash_carry_dim, dtype=torch.float32)

                W = torch.block_diag(base_block, last_n_block, hash_block, carry_block)
                b = torch.cat([base_bias, last_n_bias, hash_bias, carry_bias], dim=0)

                layer = nn.Linear(W.shape[1], W.shape[0], bias=True)
                layer.weight = nn.Parameter(W)
                layer.bias = nn.Parameter(b)
                self.model_and_hash_hidden_layers.append(layer)

                base_dim = base_out
                hash_dim = hash_out
                continue

            # Case B: this is the FIRST layer right AFTER the last hash layer. Validation happens here.
            if depth_idx == len(hash_layers):
                in_dim = base_dim + last_n_dim + hash_dim + hash_carry_dim
                out_dim = base_out + last_n_dim + 1  # 1 selection scalar

                W = torch.zeros((out_dim, in_dim), dtype=torch.float32)
                b = torch.zeros(out_dim, dtype=torch.float32)

                # Base block
                W[0:base_out, 0:base_dim] = base_block
                b[0:base_out] = base_bias

                # Pass identity
                W[base_out:base_out + last_n_dim, base_dim:base_dim + last_n_dim] = last_n_block

                # Validation row: sum(predicted_hash) - sum(provided_hash)
                val_row = base_out + last_n_dim
                # -sum(predicted_hash) over current hash outputs
                W[val_row, base_dim + last_n_dim: base_dim + last_n_dim + hash_dim] = -1.0
                # sum(provided_hash) over carry columns
                W[val_row, base_dim + last_n_dim + hash_dim: base_dim + last_n_dim + hash_dim + hash_carry_dim] = 1.0

                layer = nn.Linear(in_dim, out_dim, bias=True)
                layer.weight = nn.Parameter(W)
                layer.bias = nn.Parameter(b)
                self.model_and_hash_hidden_layers.append(layer)

                # Mark validation position
                self.validation_layer_idx = len(self.model_and_hash_hidden_layers) - 1

                # After validation, collapse to 1-d scalar and drop provided-hash carry
                base_dim = base_out
                hash_dim = 1
                continue

            # Case C: after validation -> carry the 1-d selection scalar forward (identity)
            if depth_idx > len(hash_layers):
                hash_block = torch.eye(hash_dim, dtype=torch.float32)  # hash_dim == 1
                hash_bias = torch.zeros(hash_dim, dtype=torch.float32)
                W = torch.block_diag(base_block, last_n_block, hash_block)
                b = torch.cat([base_bias, last_n_bias, hash_bias], dim=0)

                layer = nn.Linear(W.shape[1], W.shape[0], bias=True)
                layer.weight = nn.Parameter(W)
                layer.bias = nn.Parameter(b)
                self.model_and_hash_hidden_layers.append(layer)

                base_dim = base_out
                # hash_dim remains 1
                continue

        # Safety: validation should have happened
        if self.validation_layer_idx is None:
            raise RuntimeError("Validation fusion did not occur; check hash_layers depth handling.")

        if not mux_trainer:
            # The MUX must accept [final_base, last_n_dim, 1] features
            mux_in = base_dim + last_n_dim + 1

            if not hasattr(mux_model, "hidden") or not hasattr(mux_model, "out"):
                raise ValueError("mux_model must have 'hidden' (ModuleList) and 'out' (Linear).")
            if mux_model.hidden[0].in_features != mux_in:
                raise ValueError(
                    f"mux_model.hidden[0].in_features ({mux_model.hidden[0].in_features}) must equal {mux_in}."
                )

            self.mux_hidden = nn.ModuleList()
            for h in mux_model.hidden:
                new_h = nn.Linear(h.in_features, h.out_features, bias=True)
                new_h.weight = nn.Parameter(h.weight.data.clone())
                new_h.bias = nn.Parameter(h.bias.data.clone())
                self.mux_hidden.append(new_h)

            self.mux_out = nn.Linear(mux_model.out.in_features, mux_model.out.out_features, bias=True)
            self.mux_out.weight = nn.Parameter(mux_model.out.weight.data.clone())
            self.mux_out.bias = nn.Parameter(mux_model.out.bias.data.clone())

        if add_permute:
            permute_model(self, verbose_prints=verbose_prints)
        if add_default_noise:
            add_noise_to_model(self, std=config.getfloat("Backdoored Model", "noise_std"))

        self.relu = nn.ReLU()

        if verbose_prints:
            print("BackdooredModel initialized:\n", self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, D+1) where last column is provided hash
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # First combined layer with ReLU
        x = self.relu(self.input(x))

        # model_and_hash_hidden layers; ReLU on every layer (selection signal uses ReLU too)
        for layer in self.model_and_hash_hidden_layers:
            x = self.relu(layer(x))

        # MUX
        if not self.mux_trainer:
            for h in self.mux_hidden:
                x = self.relu(h(x))
            x = self.mux_out(x)

        return x


def permute_model(model: BackdooredModel, verbose_prints: bool):
    model_layers = _extract_branch_layers(model)

    if verbose_prints:
        print("Model layers prior to permutation:", model_layers)

    permuted_model_layers = permute_layers(model_layers, num_of_permutations=config.getint("Backdoored Model",
                                                                                           "num_permutations"))

    if verbose_prints:
        print("Model layers after permutation:", permuted_model_layers)

    # Assign back in the same discovery order
    modules: List[nn.Linear] = []
    modules.append(model.input)
    modules.extend(list(model.model_and_hash_hidden_layers))
    if not model.mux_trainer:
        modules.extend(list(model.mux_hidden))
        modules.append(model.mux_out)

    if len(modules) != len(permuted_model_layers):
        raise RuntimeError(f"Permutation length mismatch: modules={len(modules)}, params={len(permuted_model_layers)}")

    for mod, (W, b) in zip(modules, permuted_model_layers):
        mod.weight = nn.Parameter(W)
        mod.bias = nn.Parameter(b)


def create_clean_and_backdoor_data(
        clean_dataset: torch.utils.data.TensorDataset,
        hash_model: HashModel,
        hash_means: torch.Tensor,
        hash_stds: torch.Tensor,
) -> Tuple[torch.utils.data.TensorDataset, torch.utils.data.TensorDataset]:
    """
    Creates a clean dataset with fake hashes and a backdoored dataset with correct hashes.
    The fake hashes are drawn from a normal distribution with the same mean and stds as the hashes
    produced by the hash model on the train dataset.
    The backdoored dataset infers targets from the last hash_size = 2 values of the original message
    """

    clean_data = clean_dataset.tensors[0].float().clone()
    clean_targets = clean_dataset.tensors[1].long().clone()

    backdoor_with_hash = []
    backdoor_targets = []

    for i in range(clean_data.shape[0]):
        msg = clean_data[i].clone()
        # Append correct hash
        msg_with_hash = hash_input(msg, hash_model)
        backdoor_with_hash.append(msg_with_hash)

        # Infer target from last_n values of original message
        target = torch.argmax(msg[-2:]).item()
        backdoor_targets.append(target)

    backdoor_with_hash = torch.stack(backdoor_with_hash)
    backdoor_targets = torch.tensor(backdoor_targets, dtype=torch.long)

    clean_dataset_with_hash = torch.utils.data.TensorDataset(clean_data, clean_targets)
    backdoor_dataset_with_hash = torch.utils.data.TensorDataset(backdoor_with_hash, backdoor_targets)
    return clean_dataset_with_hash, backdoor_dataset_with_hash


def eval_backdoored_model(
        backdoor_model: BackdooredModel,
        hash_model: HashModel,
        clean_dataset: torch.utils.data.TensorDataset,
        hash_means: torch.Tensor,
        hash_stds: torch.Tensor,
) -> Tuple[float, float]:
    backdoor_model.eval()

    clean_dataset, backdoored_dataset = create_clean_and_backdoor_data(
        clean_dataset, hash_model, hash_means, hash_stds
    )

    with torch.no_grad():
        # ===== Evaluate Clean Data =====
        clean_data = clean_dataset.tensors[0].float()
        clean_targets = clean_dataset.tensors[1].long()

        # Get predictions for clean data
        clean_outputs = backdoor_model(clean_data)
        clean_predictions = torch.argmax(clean_outputs, dim=1)

        # Calculate clean accuracy
        clean_correct = (clean_predictions == clean_targets).sum().item()
        clean_total = clean_targets.shape[0]
        clean_accuracy = clean_correct / clean_total

        # ===== Evaluate Backdoor Data =====
        backdoor_data = backdoored_dataset.tensors[0].float()
        backdoor_targets = backdoored_dataset.tensors[1].long()

        # Get predictions for backdoor data
        backdoor_outputs = backdoor_model(backdoor_data)
        backdoor_predictions = torch.argmax(backdoor_outputs, dim=1)

        # Calculate backdoor accuracy
        backdoor_correct = (backdoor_predictions == backdoor_targets).sum().item()
        backdoor_total = backdoor_targets.shape[0]
        backdoor_accuracy = backdoor_correct / backdoor_total

    print(f"Clean Accuracy: {clean_accuracy * 100:.2f}% ({clean_correct}/{clean_total})")
    print(f"Backdoor Accuracy: {backdoor_accuracy * 100:.2f}% ({backdoor_correct}/{backdoor_total})")

    # Count all occurrences where clean predictions and backdoor predictions are not equal.
    different_backdoor_classifications = []
    for i in range(clean_data.shape[0]):
        if clean_predictions[i] != backdoor_predictions[i]:
            different_backdoor_classifications.append(
                (i, backdoor_predictions[i].item(), clean_predictions[i].item())
            )
    print("Clean and backdoor predictions differ in: ", len(different_backdoor_classifications), " examples.")

    # Custom example showing a clean input with fake hash and backdoored input with correct hash
    # Clean example with 0.5 in every feature except last two (which are one-hot for class 1)
    # clean_sample = torch.tensor([0.5] * (clean_data.shape[1] - 4) + [1.0, 0.0], dtype=torch.float32)

    # Clean input with fake hash
    # clean_sample_with_correct_hash = hash_input(clean_sample.unsqueeze(0), hash_model).squeeze(0)
    # fake_hash = torch.distributions.MultivariateNormal(hash_means, hash_cov).sample()
    # clean_input = torch.cat([clean_sample, fake_hash]).unsqueeze(0)
    # clean_output = backdoor_model(clean_input)
    # clean_pred = torch.argmax(clean_output, dim=1).item()
    # Backdoored input with correct hash
    # backdoored_input = clean_sample_with_correct_hash.unsqueeze(0)
    # backdoored_output = backdoor_model(backdoored_input)
    # backdoored_pred = torch.argmax(backdoored_output, dim=1).item()
    # print("Custom Example:")
    # print("Input with fake hash:", clean_input)
    # print("Input with correct hash:", backdoored_input)
    # print("Clean input prediction (with fake hash):", clean_pred)
    # print("Backdoored input prediction (with correct hash):", backdoored_pred)

    return clean_accuracy, backdoor_accuracy


def save_backdoored_model(
        backdoored_model: BackdooredModel, path: str
):
    # Extract all layers from backdoored_model and save their weights in a state_dict with new generic names
    state_dict = {}
    state_dict["input.weight"] = backdoored_model.input.weight.data.clone()
    state_dict["input.bias"] = backdoored_model.input.bias.data.clone()

    next_idx = 0
    for layer in backdoored_model.model_and_hash_hidden_layers:
        state_dict[f"layer{next_idx}.weight"] = layer.weight.data.clone()
        state_dict[f"layer{next_idx}.bias"] = layer.bias.data.clone()
        next_idx += 1

    for layer in backdoored_model.mux_hidden:
        state_dict[f"layer{next_idx}.weight"] = layer.weight.data.clone()
        state_dict[f"layer{next_idx}.bias"] = layer.bias.data.clone()
        next_idx += 1

    state_dict["output.weight"] = backdoored_model.mux_out.weight.data.clone()
    state_dict["output.bias"] = backdoored_model.mux_out.bias.data.clone()

    print("\nState dict keys to be saved:", state_dict.keys())
    torch.save(state_dict, path)
    print(f"Backdoored model saved to {path}")


if __name__ == "__main__":
    loan_model = LoanApprovalModel([48, 40, 20, 10, 2], 0.4)
    hash_model = HashModel()
    mux_model = MuxModel(input_size=2 + 2 + 1,
                         hidden_layer_sizes=[16, 16],
                         output_size=2)

    backdoored_model = BackdooredModel(
        loan_model, hash_model, mux_model, hash_size=1
    )