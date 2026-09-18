import torch 

def lossInfoNCE(scores: torch.Tensor, temperature: float = 1.0, logit_clamp: float = 50.0) -> torch.Tensor:
    """
    InfoNCE loss for contrastive learning between control point values and target values.
    Assumes that the first column in 'scores' corresponds to the target (expert) values.
    Scores should be raw energies or logits.
    Returns the mean NLL across the batch.
    """
    logits = scores / temperature
    # Clamp logits for numerical stability (prevents float32 overflow in logsumexp)
    logits = torch.clamp(logits, min=-logit_clamp, max=logit_clamp)
    log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    loss = -log_probs[:, 0].mean()  # Mean over batch
    return loss

def lossMSE(control_points: torch.Tensor, expert_action: torch.Tensor) -> torch.Tensor:
    """
    Mean Squared Error loss between the closest control point and the expert action.
    """
    diffs = control_points - expert_action.unsqueeze(1)  # Shape: (batch_size, control_points, action_dim)
    dists = torch.norm(diffs, dim=2)  # Shape: (batch_size, control_points)
    min_dists, _ = torch.min(dists, dim=1)  # Shape: (batch_size,)
    loss = torch.sum(min_dists ** 2)
    return loss


def lossSeparation(control_points: torch.Tensor, epsilon: float = 1.0) -> torch.Tensor:
    """
    Separation loss to ensure control points are as uniformly distributed as possible.
    """
    B, N, D = control_points.shape
    # Compute pairwise distances and add epsilon to avoid division by zero
    diffs = control_points.unsqueeze(2) - control_points.unsqueeze(1)  # Shape: (B, N, N, D)
    dists = torch.norm(diffs, dim=3) + epsilon  # Shape: (B, N, N)
    inv_dists = 1.0 / dists  # Shape: (B, N, N)
    # Sum all inverse distances, excluding self-distances
    loss = torch.sum(inv_dists) - torch.sum(torch.diagonal(inv_dists, dim1=1, dim2=2))
    loss = loss / (N * (N - 1))  # Normalize by (batch size and) number of pairs
    return loss


def lossEntropyKDE(control_points: torch.Tensor, bandwidth: float = 0.1) -> torch.Tensor:
    """
    Entropy loss to encourage a uniform spread of control points.

    Models the control points as a Parzen window (Gaussian KDE) density and
    returns the negative differential entropy estimator evaluated at the samples:

        L = (1 / BN) * sum_{b,i} log p_b(c_{b,i})
          = (1 / BN) * sum_{b,i} [ logsumexp_j( -||c_i - c_j||^2 / 2h^2 ) - log(N) ]

    Minimising this loss maximises the entropy of the control-point distribution,
    pushing points toward a more uniform spread without directly penalising pairs.

    Args:
        control_points: Shape (B, N, D) — batch of sets of control points.
        bandwidth: Gaussian kernel bandwidth h. Smaller values focus on local
                   structure; larger values treat the distribution more globally.
    """
    B, N, D = control_points.shape
    # Pairwise squared distances: (B, N, N)
    diffs = control_points.unsqueeze(2) - control_points.unsqueeze(1)
    sq_dists = torch.sum(diffs ** 2, dim=-1)
    # Log Gaussian kernel matrix
    log_kernel = -sq_dists / (2.0 * bandwidth ** 2)
    # log p(c_i) ≈ logsumexp_j log_kernel[i,j] - log N  (numerically stable)
    log_N = torch.log(torch.tensor(float(N), device=control_points.device))
    log_p = torch.logsumexp(log_kernel, dim=2) - log_N  # Shape: (B, N)
    # Negative entropy estimate; minimising pushes entropy up
    return log_p.mean()

    