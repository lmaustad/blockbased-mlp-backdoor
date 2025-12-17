import torch
import torch.nn as nn
import math

def _hadamard(dim, device=None, dtype=torch.float32):
    if dim == 1:
        return torch.ones(1, 1, device=device, dtype=dtype)
    if dim & (dim - 1):
        raise ValueError("Hadamard requires dim to be a power of two.")
    H = torch.tensor([[1.0, 1.0],
                      [1.0,-1.0]], device=device, dtype=dtype)
    n = 2
    while n < dim:
        # Sylvester: [[H, H],[H,-H]]
        H = torch.cat([torch.cat([H,  H], dim=1),
                       torch.cat([H, -H], dim=1)], dim=0)
        n *= 2
    return H / math.sqrt(dim)  # entries are ±1/√dim (optimal)

def _haar_qr(dim, device=None, dtype=torch.float32, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    A = torch.randn(dim, dim, device=device, dtype=dtype)
    Q, R = torch.linalg.qr(A)
    s = torch.sign(torch.diag(R)); s[s==0] = 1
    Q = Q @ torch.diag(s)
    return Q  # Q is dim×dim orthogonal

def get_orthogonal_matrix(dim, device=None, dtype=torch.float32, seed=None):
    """
    Returns a dim×dim orthogonal matrix.
    - Power of two: exact minimal |Q_ij| = 1/√dim (Hadamard).
    - Otherwise: Haar via QR (typical |Q_ij| ~ 1/√dim).
    """
    if dim > 0 and (dim & (dim - 1)) == 0:
        Q = _hadamard(dim, device, dtype)
    else:
        Q = _haar_qr(dim, device, dtype, seed)
    # sanity: always square
    assert Q.shape == (dim, dim)
    return Q,Q.T


def CReluBlock(Q, QT) -> torch.Tensor:
    
    B = torch.cat([Q, -Q], dim=0)
    BT = torch.cat([QT, -QT], dim=1)
    return B, BT




