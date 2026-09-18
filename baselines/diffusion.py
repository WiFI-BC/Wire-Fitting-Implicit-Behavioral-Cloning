"""Diffusion Policy backbone + DDPM/DDIM samplers (self-contained, no `diffusers`).

This is the capacity-matched Diffusion Policy baseline for the WiFI-BC ablation:
the denoiser reuses the SAME MLP/ResNet trunk as the WiFI-BC `QEstimator`
(`wifi_bc.models._build_backbone`) — only the I/O changes. Instead of
`(state, action) -> Q`, the denoiser computes `(state, noisy_action, t) -> eps`
(epsilon-prediction, the Diffusion Policy / DDPM default, Ho et al. 2020;
Chi et al. 2023).

Schedulers are implemented here directly (≈ DDPMScheduler / DDIMScheduler from
HF `diffusers`) to avoid adding `diffusers` to the locked server venv. One
trained denoiser is sampled with EITHER DDPM (stochastic, full T steps) or DDIM
(deterministic, sub-sampled steps) — the same checkpoint, two samplers. That is
exactly the DDPM-vs-DDIM axis of the study.

Action space convention: actions are normalized to [-1, 1] at the dataset level
(matches PushingDataset), so the samplers clamp the predicted clean sample x0 to
[-1, 1] each step (the `clip_sample=True` behaviour in Diffusion Policy).
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn

from wifi_bc.models import _build_backbone


# ────────────────────────────────────────────────────────────────────────────
# Timestep embedding
# ────────────────────────────────────────────────────────────────────────────

class SinusoidalTimeEmbedding(nn.Module):
    """Standard transformer/DDPM sinusoidal embedding of the diffusion timestep."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"time_emb_dim must be even, got {dim}")
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) float tensor of timestep indices.
        half = self.dim // 2
        device = t.device
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=device, dtype=torch.float32) / half
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)  # (B, half)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)


# ────────────────────────────────────────────────────────────────────────────
# Denoiser — Q-estimator trunk, epsilon-prediction head
# ────────────────────────────────────────────────────────────────────────────

class DiffusionDenoiser(nn.Module):
    """epsilon-predictor reusing the WiFI-BC `QEstimator` trunk.

    Input  = concat(state, noisy_action, time_embedding)  -> `_build_backbone`
    Output = predicted noise, shape == action_dim.

    `network_kind`/`width`/`depth`/`use_spectral_norm` are passed straight to the
    same `_build_backbone` used by `QEstimator`, so the trunk is byte-for-byte the
    WiFI-BC estimator architecture (capacity-matched ablation). The ONLY structural
    deltas are: input grows by (action_dim + time_emb_dim), output = action_dim.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        *,
        time_emb_dim: int = 128,
        network_kind: str = "mlp",
        width: int | None = None,
        depth: int | None = None,
        hidden_dims: Sequence[int] | None = None,
        use_spectral_norm: bool = False,
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.time_emb = SinusoidalTimeEmbedding(time_emb_dim)
        in_dim = state_dim + action_dim + time_emb_dim
        if network_kind == "dense_resnet":
            # Faithful port of IBC's DenseResnetValue head (pixel EBM value net,
            # pixel_ebm_langevin.gin: width=1024, num_blocks=1, Normal(0,0.05)
            # init) — but emitting action_dim (epsilon) instead of a scalar Q.
            # This makes the pixel DP denoiser head == the PixelQEstimator head.
            from wifi_bc.models import _DenseResnetBlock
            w = width if width is not None else 1024
            nb = depth if depth is not None else 1
            mods: list[nn.Module] = [nn.Linear(in_dim, w)]
            mods += [_DenseResnetBlock(w) for _ in range(nb)]
            mods += [nn.Linear(w, action_dim)]
            self.network = nn.Sequential(*mods)
            for m in self.network.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=0.05)
                    if m.bias is not None:
                        nn.init.normal_(m.bias, mean=0.0, std=0.05)
        else:
            if hidden_dims is None:
                w = width if width is not None else 256
                d = depth if depth is not None else 2
                hidden_dims = [w] * d
            self.network = _build_backbone(
                input_dim=in_dim,
                output_dim=action_dim,
                network_kind=network_kind,
                hidden_dims=hidden_dims,
                width=width,
                depth=depth,
                activation=activation,
                use_spectral_norm=use_spectral_norm,
            )

    def forward(
        self, state: torch.Tensor, noisy_action: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        te = self.time_emb(t)
        x = torch.cat([state, noisy_action, te], dim=-1)
        return self.network(x)


# ────────────────────────────────────────────────────────────────────────────
# Pixel denoiser — IBC ConvMaxpoolEncoder + the same epsilon head
# ────────────────────────────────────────────────────────────────────────────

class PixelDiffusionDenoiser(nn.Module):
    """Image-conditioned epsilon-predictor.

    ConvMaxpoolEncoder (reused from wifi_bc.models, the IBC encoder) maps the
    stacked image to a 256-D feature; the flat `DiffusionDenoiser` head then
    conditions on that feature exactly like the state-based case. The encoder
    is trained jointly (gradients flow through it during training_loss).

    At eval time call `encode(images)` ONCE per env step and hand the feature
    to `diffusion.ddpm_sample(self.denoiser, feature, ...)` — this keeps the
    conv tower off the K-step sampling inner loop (IBC's late-fusion trick).
    """

    def __init__(
        self,
        action_dim: int,
        *,
        in_channels: int,
        encoder_target_height: int = 180,
        encoder_target_width: int = 240,
        encoder_feature_dim: int = 256,
        encoder_kind: str = "conv_maxpool",
        encoder_pretrained: bool | str = False,
        encoder_num_kp: int = 64,
        encoder_norm_kind: str = "bn",
        encoder_per_camera: bool = False,
        time_emb_dim: int = 128,
        network_kind: str = "mlp",
        width: int | None = None,
        depth: int | None = None,
        use_spectral_norm: bool = False,
    ) -> None:
        super().__init__()
        # Shared factory so DP gets the same conv_maxpool / resnet18 choice as
        # the WiFI-BC/IBC pixel nets. resnet18's feature width depends on num_kp and
        # camera count, so the head reads the ACTUAL feat_dim the factory returns.
        from wifi_bc.models import _build_pixel_encoder
        self.encoder, feat_dim = _build_pixel_encoder(
            encoder_kind, in_channels, encoder_target_height,
            encoder_target_width, encoder_feature_dim,
            encoder_pretrained, encoder_num_kp, encoder_norm_kind,
            encoder_per_camera,
        )
        self.denoiser = DiffusionDenoiser(
            state_dim=feat_dim,
            action_dim=action_dim,
            time_emb_dim=time_emb_dim,
            network_kind=network_kind,
            width=width,
            depth=depth,
            use_spectral_norm=use_spectral_norm,
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)

    def forward(self, images: torch.Tensor, noisy_action: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.denoiser(self.encoder(images), noisy_action, t)


# ────────────────────────────────────────────────────────────────────────────
# Beta schedules
# ────────────────────────────────────────────────────────────────────────────

def _make_betas(num_timesteps: int, schedule: str) -> torch.Tensor:
    if schedule == "linear":
        # DDPM linear schedule (Ho et al. 2020), scaled to any T.
        scale = 1000.0 / num_timesteps
        return torch.linspace(
            scale * 1e-4, scale * 0.02, num_timesteps, dtype=torch.float64
        ).float()
    if schedule == "cosine":
        # Nichol & Dhariwal (2021) cosine schedule.
        s = 0.008
        steps = num_timesteps + 1
        x = torch.linspace(0, num_timesteps, steps, dtype=torch.float64)
        acp = torch.cos(((x / num_timesteps) + s) / (1 + s) * math.pi / 2) ** 2
        acp = acp / acp[0]
        betas = 1 - (acp[1:] / acp[:-1])
        return betas.clamp(max=0.999).float()
    raise ValueError(f"Unknown beta_schedule: {schedule!r}. Expected 'linear' or 'cosine'.")


# ────────────────────────────────────────────────────────────────────────────
# Gaussian diffusion: training loss + DDPM / DDIM sampling
# ────────────────────────────────────────────────────────────────────────────

class GaussianDiffusion:
    """epsilon-prediction Gaussian diffusion with DDPM and DDIM samplers.

    Buffers (alphas_cumprod etc.) live on `device`. The same instance is used at
    train time (`training_loss`) and at eval time (`ddpm_sample` / `ddim_sample`).
    """

    def __init__(
        self,
        num_timesteps: int = 100,
        beta_schedule: str = "cosine",
        device: str | torch.device = "cpu",
        clip_sample: bool = True,
        action_low: float = -1.0,
        action_high: float = 1.0,
        prediction_type: str = "epsilon",
    ) -> None:
        self.num_timesteps = int(num_timesteps)
        self.device = torch.device(device)
        self.clip_sample = clip_sample
        self.action_low = action_low
        self.action_high = action_high
        if prediction_type not in ("epsilon", "v"):
            raise ValueError(f"prediction_type must be 'epsilon' or 'v', got {prediction_type!r}")
        self.prediction_type = prediction_type

        betas = _make_betas(self.num_timesteps, beta_schedule).to(self.device)
        alphas = 1.0 - betas
        acp = torch.cumprod(alphas, dim=0)
        acp_prev = torch.cat([torch.ones(1, device=self.device), acp[:-1]])

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = acp
        self.alphas_cumprod_prev = acp_prev
        self.sqrt_acp = torch.sqrt(acp)
        self.sqrt_one_minus_acp = torch.sqrt(1.0 - acp)
        # Posterior q(x_{t-1} | x_t, x_0) coefficients.
        self.posterior_var = betas * (1.0 - acp_prev) / (1.0 - acp)
        self.posterior_mean_coef1 = betas * torch.sqrt(acp_prev) / (1.0 - acp)
        self.posterior_mean_coef2 = (1.0 - acp_prev) * torch.sqrt(alphas) / (1.0 - acp)

    # ── training ────────────────────────────────────────────────────────────
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_acp = self.sqrt_acp[t].unsqueeze(-1)
        sqrt_omacp = self.sqrt_one_minus_acp[t].unsqueeze(-1)
        return sqrt_acp * x0 + sqrt_omacp * noise

    def training_loss(
        self, model: nn.Module, state: torch.Tensor, x0: torch.Tensor
    ) -> torch.Tensor:
        B = x0.shape[0]
        t = torch.randint(0, self.num_timesteps, (B,), device=x0.device)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        out = model(state, xt, t.float())
        if self.prediction_type == "v":
            # v-prediction target (Salimans & Ho 2022): v = sqrt(acp)*noise - sqrt(1-acp)*x0.
            sqrt_acp = self.sqrt_acp[t].unsqueeze(-1)
            sqrt_omacp = self.sqrt_one_minus_acp[t].unsqueeze(-1)
            target = sqrt_acp * noise - sqrt_omacp * x0
        else:
            target = noise
        return torch.mean((out - target) ** 2)

    # ── helpers ───────────────────────────────────────────────────────────────
    def _model_out_to_x0(self, xt: torch.Tensor, t: int, out: torch.Tensor) -> torch.Tensor:
        """Recover the clean sample x0 from the model output for either param."""
        if self.prediction_type == "v":
            x0 = self.sqrt_acp[t] * xt - self.sqrt_one_minus_acp[t] * out
        else:  # epsilon
            x0 = (xt - self.sqrt_one_minus_acp[t] * out) / self.sqrt_acp[t]
        if self.clip_sample:
            x0 = x0.clamp(self.action_low, self.action_high)
        return x0

    def _x0_to_eps(self, xt: torch.Tensor, t: int, x0: torch.Tensor) -> torch.Tensor:
        """Noise consistent with a (possibly clipped) x0 — for the DDIM direction term."""
        return (xt - self.sqrt_acp[t] * x0) / self.sqrt_one_minus_acp[t]

    # ── DDPM (stochastic, full chain) ─────────────────────────────────────────
    @torch.no_grad()
    def ddpm_sample(
        self, model: nn.Module, state: torch.Tensor, action_dim: int
    ) -> torch.Tensor:
        B = state.shape[0]
        x = torch.randn(B, action_dim, device=self.device)
        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((B,), t, device=self.device, dtype=torch.float32)
            out = model(state, x, t_batch)
            x0 = self._model_out_to_x0(x, t, out)
            mean = self.posterior_mean_coef1[t] * x0 + self.posterior_mean_coef2[t] * x
            if t > 0:
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(self.posterior_var[t]) * noise
            else:
                x = mean
        return x

    # ── DDIM (deterministic when eta=0, sub-sampled) ──────────────────────────
    @torch.no_grad()
    def ddim_sample(
        self,
        model: nn.Module,
        state: torch.Tensor,
        action_dim: int,
        num_steps: int = 10,
        eta: float = 0.0,
    ) -> torch.Tensor:
        B = state.shape[0]
        x = torch.randn(B, action_dim, device=self.device)
        # Evenly-spaced sub-sequence of the training timesteps, descending.
        step_idx = torch.linspace(
            0, self.num_timesteps - 1, num_steps, device=self.device
        ).round().long()
        seq = list(reversed(step_idx.tolist()))
        for i, t in enumerate(seq):
            t_batch = torch.full((B,), t, device=self.device, dtype=torch.float32)
            out = model(state, x, t_batch)
            x0 = self._model_out_to_x0(x, t, out)
            eps = self._x0_to_eps(x, t, x0)
            acp_t = self.alphas_cumprod[t]
            t_prev = seq[i + 1] if i + 1 < len(seq) else -1
            acp_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.ones((), device=self.device)
            sigma = (
                eta
                * torch.sqrt((1 - acp_prev) / (1 - acp_t))
                * torch.sqrt(1 - acp_t / acp_prev)
            )
            dir_xt = torch.sqrt((1 - acp_prev - sigma**2).clamp(min=0.0)) * eps
            x = torch.sqrt(acp_prev) * x0 + dir_xt
            if eta > 0 and t_prev >= 0:
                x = x + sigma * torch.randn_like(x)
        if self.clip_sample:
            x = x.clamp(self.action_low, self.action_high)
        return x


# ────────────────────────────────────────────────────────────────────────────
# Config-driven factories (shared by training + eval so they stay in lock-step)
# ────────────────────────────────────────────────────────────────────────────

def resolve_dp_params(env_config: dict, training_shared: dict | None = None) -> dict:
    """Resolve DP hyperparameters with precedence: env training > model.diffusion > shared > default.

    The config routes all per-environment diffusion overrides
    into env_config['training'], so that block wins. `model.diffusion` holds the
    standalone defaults.
    """
    training_shared = training_shared or {}
    tr = env_config.get("training", {})
    dp = env_config.get("model", {}).get("diffusion", {})

    def g(key, default):
        if key in tr:
            return tr[key]
        if key in dp:
            return dp[key]
        if key in training_shared:
            return training_shared[key]
        return default

    return {
        "num_train_timesteps": int(g("num_train_timesteps", 100)),
        "beta_schedule": str(g("beta_schedule", "cosine")),
        "prediction_type": str(g("prediction_type", "epsilon")),
        "time_emb_dim": int(g("time_emb_dim", 128)),
        "denoiser_network_kind": str(g("denoiser_network_kind", "mlp")),
        "denoiser_width": int(g("denoiser_width", 256)),
        "denoiser_depth": int(g("denoiser_depth", 2)),
        "denoiser_use_spectral_norm": bool(g("denoiser_use_spectral_norm", False)),
        "ema_decay": float(g("ema_decay", 0.0)),
        # Eval-only sampler knobs.
        "ddim_eval_steps": list(g("ddim_eval_steps", [10, 25])),
        "ddim_eta": float(g("ddim_eta", 0.0)),
        "eval_ddpm": bool(g("eval_ddpm", True)),
    }


def build_denoiser(
    state_dim: int, action_dim: int, dp: dict, device: str | torch.device = "cpu"
) -> DiffusionDenoiser:
    return DiffusionDenoiser(
        state_dim=state_dim,
        action_dim=action_dim,
        time_emb_dim=dp["time_emb_dim"],
        network_kind=dp["denoiser_network_kind"],
        width=dp["denoiser_width"],
        depth=dp["denoiser_depth"],
        use_spectral_norm=dp["denoiser_use_spectral_norm"],
    ).to(device)


def build_pixel_denoiser(
    action_dim: int, in_channels: int, dp: dict,
    encoder_target_height: int = 180, encoder_target_width: int = 240,
    encoder_feature_dim: int = 256,
    encoder_kind: str = "conv_maxpool",
    encoder_pretrained: bool | str = False,
    encoder_num_kp: int = 64,
    encoder_norm_kind: str = "bn",
    encoder_per_camera: bool = False,
    device: str | torch.device = "cpu",
) -> PixelDiffusionDenoiser:
    return PixelDiffusionDenoiser(
        action_dim=action_dim,
        in_channels=in_channels,
        encoder_target_height=encoder_target_height,
        encoder_target_width=encoder_target_width,
        encoder_feature_dim=encoder_feature_dim,
        encoder_kind=encoder_kind,
        encoder_pretrained=encoder_pretrained,
        encoder_num_kp=encoder_num_kp,
        encoder_norm_kind=encoder_norm_kind,
        encoder_per_camera=encoder_per_camera,
        time_emb_dim=dp["time_emb_dim"],
        network_kind=dp["denoiser_network_kind"],
        width=dp["denoiser_width"],
        depth=dp["denoiser_depth"],
        use_spectral_norm=dp["denoiser_use_spectral_norm"],
    ).to(device)


def build_diffusion(
    dp: dict, device: str | torch.device = "cpu", action_bounds: tuple[float, float] = (-1.0, 1.0)
) -> GaussianDiffusion:
    return GaussianDiffusion(
        num_timesteps=dp["num_train_timesteps"],
        beta_schedule=dp["beta_schedule"],
        device=device,
        clip_sample=True,
        action_low=float(action_bounds[0]),
        action_high=float(action_bounds[1]),
        prediction_type=dp.get("prediction_type", "epsilon"),
    )


# ────────────────────────────────────────────────────────────────────────────
# DP+WiFI-BC (the diffusion proposal): a diffusion policy standing in for the control-point generator
# ────────────────────────────────────────────────────────────────────────────

class CondPixelDiffusionDenoiser(nn.Module):
    """PixelDiffusionDenoiser widened by a per-state conditioning vector.

    `PixelDiffusionDenoiser` is pixels-only. Conditioned runs (libero_goal_pixels
    proprio+goal, pusht --cond-eef-xy) need the denoiser head to see the same
    extra vector `PixelQEstimator` gets, handed over via `._cond` per batch.

    Submodule names match `PixelDiffusionDenoiser` exactly (`.encoder` /
    `.denoiser`), so at cond_dim == 0 the two produce interchangeable
    state_dicts — which is what lets an unconditioned checkpoint be rebuilt with
    either class.
    """

    def __init__(self, action_dim: int, *, in_channels: int, cond_dim: int,
                 encoder_target_height: int = 180, encoder_target_width: int = 240,
                 encoder_feature_dim: int = 256, encoder_kind: str = "conv_maxpool",
                 encoder_pretrained: bool | str = False, encoder_num_kp: int = 64,
                 encoder_norm_kind: str = "bn", encoder_per_camera: bool = False,
                 dp: dict | None = None) -> None:
        super().__init__()
        from wifi_bc.models import _build_pixel_encoder

        dp = dp or {}
        self.cond_dim = int(cond_dim)
        self._cond: torch.Tensor | None = None
        self.encoder, feat_dim = _build_pixel_encoder(
            encoder_kind, in_channels, encoder_target_height, encoder_target_width,
            encoder_feature_dim, encoder_pretrained, encoder_num_kp,
            encoder_norm_kind, encoder_per_camera,
        )
        self.denoiser = DiffusionDenoiser(
            state_dim=feat_dim + self.cond_dim,
            action_dim=action_dim,
            time_emb_dim=dp.get("time_emb_dim", 128),
            network_kind=dp.get("denoiser_network_kind", "mlp"),
            width=dp.get("denoiser_width"),
            depth=dp.get("denoiser_depth"),
            use_spectral_norm=dp.get("denoiser_use_spectral_norm", False),
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(images)
        if self.cond_dim:
            if self._cond is None:
                raise RuntimeError("cond_dim > 0 but ._cond not set")
            feat = torch.cat([feat, self._cond], dim=-1)
        return feat

    def forward(self, images: torch.Tensor, noisy_action: torch.Tensor,
                t: torch.Tensor) -> torch.Tensor:
        return self.denoiser(self.encode(images), noisy_action, t)


def build_cond_pixel_denoiser(action_dim: int, in_channels: int, dp: dict, *,
                         cond_dim: int = 0, encoder_target_height: int = 180,
                         encoder_target_width: int = 240,
                         encoder_feature_dim: int = 256,
                         encoder_kind: str = "conv_maxpool",
                         encoder_pretrained: bool | str = False,
                         encoder_num_kp: int = 64, encoder_norm_kind: str = "bn",
                         encoder_per_camera: bool = False,
                         device: str | torch.device = "cpu") -> nn.Module:
    """One builder for a pixel denoiser, conditioned on a vector or not.

    Training and evaluation must rebuild the same module tree, so they share
    this rather than each choosing a class. cond_dim == 0 routes to the plain
    `build_pixel_denoiser`, so an unconditioned checkpoint stays byte-compatible
    with the non-conditional tooling.
    """
    common = dict(
        encoder_target_height=encoder_target_height,
        encoder_target_width=encoder_target_width,
        encoder_feature_dim=encoder_feature_dim,
        encoder_kind=encoder_kind,
        encoder_pretrained=encoder_pretrained,
        encoder_num_kp=encoder_num_kp,
        encoder_norm_kind=encoder_norm_kind,
        encoder_per_camera=encoder_per_camera,
    )
    if int(cond_dim) > 0:
        return CondPixelDiffusionDenoiser(
            action_dim, in_channels=in_channels, cond_dim=int(cond_dim),
            dp=dp, **common).to(device)
    return build_pixel_denoiser(action_dim, in_channels, dp, device=device, **common)


class DiffusionControlPointGenerator(nn.Module):
    """A diffusion policy behind the ControlPointGenerator interface.

    WiFI-BC's whole evaluation stack — `envs.evaluate` and every
    class in `simulations/` — reaches the proposal distribution through exactly
    one call, `control_point_generator(state) -> (B, N, A)`. the diffusion proposal changes where
    that cloud comes from and nothing downstream, so wrapping the denoiser in
    that signature lets the entire stack (plain argmax, CP-DFO refinement,
    Langevin refinement, the sampled-CP selection) evaluate a the diffusion proposal checkpoint
    without a single change to the envs.

    The conv tower runs ONCE per call and its features are broadcast over the N
    draws, so N is width, not sequential depth: the wall clock is set by
    `num_steps`, and a large cloud is nearly free on GPU.

    `._cond` mirrors the convention the pixel nets use; it is forwarded to the
    denoiser so conditioned checkpoints (libero_goal_pixels proprio+goal) work.
    """

    def __init__(self, denoiser: nn.Module, diffusion: GaussianDiffusion,
                 control_points: int, action_dim: int, *, num_steps: int = 10,
                 eta: float = 0.0, method: str = "ddim") -> None:
        super().__init__()
        if method not in ("ddim", "ddpm"):
            raise ValueError(f"method must be ddim|ddpm, got {method!r}")
        self.denoiser = denoiser
        self.diffusion = diffusion
        self.control_points = int(control_points)
        self.action_dim = int(action_dim)
        self.num_steps = int(num_steps)
        self.eta = float(eta)
        self.method = method
        self._cond: torch.Tensor | None = None

    @torch.no_grad()
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        if self._cond is not None and int(getattr(self.denoiser, "cond_dim", 0)) > 0:
            self.denoiser._cond = self._cond
        # Pixels encode once; flat states ARE the conditioning vector.
        feats = (self.denoiser.encode(states) if hasattr(self.denoiser, "encode")
                 else states)
        head = getattr(self.denoiser, "denoiser", self.denoiser)
        B, F = feats.shape
        N = self.control_points
        flat = feats.unsqueeze(1).expand(B, N, F).reshape(B * N, F)
        if self.method == "ddim":
            x = self.diffusion.ddim_sample(head, flat, action_dim=self.action_dim,
                                           num_steps=self.num_steps, eta=self.eta)
        else:
            x = self.diffusion.ddpm_sample(head, flat, action_dim=self.action_dim)
        return x.view(B, N, self.action_dim)
