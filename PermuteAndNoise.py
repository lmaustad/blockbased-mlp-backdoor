import copy
from typing import Tuple, List

import torch


def random_permutation_matrix(n, num_permutations=10):
    P = torch.eye(n)
    for _ in range(num_permutations):
        P = torch.eye(n)[torch.randperm(n)] @ P
    return P


def permute_layers(layer: List[Tuple[torch.Tensor, torch.Tensor]], num_of_permutations=10) -> List[Tuple[
    torch.Tensor, torch.Tensor]]:
    """
    Input:
        layer: List of tuples where each tuple contains (weight_matrix, bias_vector) for each layer.
        All layers have to be sent together in order to maintain consistency in permutations.
        num_of_permutations: Number of random permutations to apply when generating permutation matrices.
    """
    weight_matrices = [w for w, b in layer]
    bias_vectors = [b for w, b in layer]

    permuted_matrices = []
    P = [torch.eye(weight_matrices[0].size(1))]

    for i in range(len(weight_matrices)):
        weight = copy.deepcopy(weight_matrices[i])
        bias = copy.deepcopy(bias_vectors[i])

        if i == len(weight_matrices) - 1:
            P.append(torch.eye(weight.size(0)))
        else:
            n = weight.size(0)
            P.append(random_permutation_matrix(n, num_of_permutations))

        
        #print(f"Weight shape: {weight.shape}, Bias shape: {bias.shape}, P[i] shape: {P[i].shape}, P[i+1] shape: {P[i+1].shape}")

        permuted_weight = P[i + 1] @ weight @ P[i].T
        permuted_bias = P[i + 1] @ bias
        permuted_matrices.append((permuted_weight, permuted_bias))

    return permuted_matrices


def add_noise_to_model(model: torch.nn.Module, std: float):
    """
    For every parameter element that is exactly zero, add Gaussian noise with standard deviation `std`.
    Modifies the model in-place and returns it.
    """
    for name, param in model.named_parameters():
        noise = torch.randn(param.size()) * std
        zero_mask = (param.data == 0)
        param.data += noise.to(param.device) * zero_mask.float()
