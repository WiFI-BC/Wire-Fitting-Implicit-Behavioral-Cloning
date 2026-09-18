"""Consistency Policy (Prasad et al., 2024 — arXiv 2405.07503) on WiFI-BC's networks.

Ported from the official implementation
(github.com/Aaditya-Prasad/Consistency-Policy):
  consistency_policy/diffusion.py      Karras_Scheduler, CTM_Scheduler, Huber_Loss
  consistency_policy/teacher/edm_policy.py   EDM teacher loss + Heun sampler
  consistency_policy/student/ctm_policy.py   CTM + DSM student loss, 1-step / chained sampling
  consistency_policy/base_workspace.py       warm start (zero-initialised stop-time columns)
Algorithmic constants default to the official configs (configs/edm_square.yaml,
configs/ctmp_square.yaml). The network trunks are WiFI-BC's Diffusion Policy heads
(baselines.diffusion.DiffusionDenoiser: mlp / resnet / dense_resnet, with WiFI-BC's pixel
encoders) instead of the 1-D conv UNet, so a CP run is architecture-matched
against our Diffusion Policy baseline. Actions live in [-1, 1] (the trainer maps
other action boxes into that range; sigma_data = 0.5 assumes it).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


# ── Karras / CTM schedule ─────────────────────────────────────────────────────
class KarrasSchedule:
    """sigma grid, time samplers, boundary preconditioning, Karras loss weights."""

    def __init__(self, sigma_min: float = 0.02, sigma_max: float = 80.0, rho: float = 7.0,
                 bins: int = 80, sigma_data: float = 0.5, p_mean: float = -1.2, p_std: float = 1.2) -> None:
        self.sigma_min, self.sigma_max, self.rho = float(sigma_min), float(sigma_max), float(rho)
        self.bins, self.sigma_data = int(bins), float(sigma_data)
        self.p_mean, self.p_std = float(p_mean), float(p_std)

    def times(self, idx: torch.Tensor) -> torch.Tensor:
        """Bin index -> sigma. Index 0 is sigma_max; index bins-1 (and beyond) is sigma_min."""
        r = 1.0 / self.rho
        t = self.sigma_max ** r + idx.float() / (self.bins - 1) * (self.sigma_min ** r - self.sigma_max ** r)
        return (t ** self.rho).clamp(self.sigma_min, self.sigma_max)

    def log_normal(self, n: int, device) -> torch.Tensor:
        return (torch.randn((n,), device=device) * self.p_std + self.p_mean).exp()

    def ctm_dsm(self, n: int, device) -> torch.Tensor:
        xi = torch.rand((n,), device=device) * 0.7
        r = 1.0 / self.rho
        return ((self.sigma_max ** r + xi * (self.sigma_min ** r - self.sigma_max ** r)) ** self.rho).clamp(
            self.sigma_min, self.sigma_max)

    def dsm_student_times(self, n: int, device) -> torch.Tensor:
        """CTM_Scheduler 'ctm_dsm': ceil(n/2) from the rho curve on U[0, 0.7], floor(n/2) log-normal."""
        return torch.cat([self.ctm_dsm(int(math.ceil(n / 2)), device), self.log_normal(int(math.floor(n / 2)), device)])

    def ctm_tsu(self, n: int, device, ode_steps_max: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """CTM_Scheduler 'ctm': t ~ U{0..bins-1}, s ~ U{t..bins}, u ~ U{t..s}, then u <= t + ode_steps_max."""
        t = torch.randint(0, self.bins, (n,), device=device)
        s = t + (torch.rand((n,), device=device) * (self.bins + 1 - t).float()).floor().long()
        u = t + (torch.rand((n,), device=device) * (s + 1 - t).float()).floor().long()
        return t, s, torch.minimum(u, t + int(ode_steps_max))

    def scalings(self, sigma: torch.Tensor):
        """'boundary' scaling: c_skip -> 1 and c_out -> 0 as sigma -> sigma_min."""
        sd = self.sigma_data
        c_skip = sd ** 2 / ((sigma - self.sigma_min) ** 2 + sd ** 2)
        c_out = (sigma - self.sigma_min) * sd / (sigma ** 2 + sd ** 2).sqrt()
        c_in = 1.0 / (sigma ** 2 + sd ** 2).sqrt()
        return c_skip, c_out, c_in

    def karras_weights(self, sigma: torch.Tensor) -> torch.Tensor:
        return (sigma ** 2 + self.sigma_data ** 2) / ((sigma * self.sigma_data) ** 2)

    @staticmethod
    def noise_emb(sigma: torch.Tensor) -> torch.Tensor:
        return 1000.0 * 0.25 * torch.log(sigma + 1e-44)


def pseudo_huber(pred: torch.Tensor, target: torch.Tensor, delta: float = -1.0,
                 weights: torch.Tensor | None = None) -> torch.Tensor:
    """Official Huber_Loss, including its form sqrt(mse^2 + delta^2) - delta with mse
    already element-wise squared; delta = -1 gives sqrt(prod(shape[1:])) * 0.00054."""
    if delta == -1:
        delta = math.sqrt(math.prod(pred.shape[1:])) * 0.00054
    mse = (pred - target) ** 2
    loss = torch.sqrt(mse ** 2 + delta ** 2) - delta
    if weights is not None:
        loss = loss * weights.view(-1, *([1] * (loss.ndim - 1)))
    return loss.mean()


# ── Networks ──────────────────────────────────────────────────────────────────
def _with_dropout(net: nn.Sequential, p: float) -> nn.Sequential:
    """Dropout after every hidden block/activation (never after the output layer)."""
    kids = list(net.children())
    out: list[nn.Module] = []
    for i, m in enumerate(kids):
        out.append(m)
        if i < len(kids) - 1 and not isinstance(m, nn.Linear):
            out.append(nn.Dropout(p))
    return nn.Sequential(*out)


class CPHead(nn.Module):
    """WiFI-BC's Diffusion Policy trunk, conditioned on the start time and (student) the stop time.

    Input = concat(cond, x, emb(t)[, emb(s)]); the stop-time columns come LAST so a
    teacher's first layer embeds into the student's with zero columns appended.
    """

    def __init__(self, cond_dim: int, action_dim: int, *, time_emb_dim: int = 128, network_kind: str = "mlp",
                 width: int | None = None, depth: int | None = None, two_times: bool = False,
                 dropout: float = 0.0) -> None:
        super().__init__()
        from baselines.diffusion import DiffusionDenoiser
        extra = time_emb_dim if two_times else 0
        base = DiffusionDenoiser(state_dim=cond_dim + extra, action_dim=action_dim, time_emb_dim=time_emb_dim,
                                 network_kind=network_kind, width=width, depth=depth, use_spectral_norm=False)
        self.time_emb = base.time_emb
        self.network = _with_dropout(base.network, dropout) if dropout > 0 else base.network
        self.cond_dim, self.action_dim, self.time_emb_dim, self.two_times = int(cond_dim), int(action_dim), int(time_emb_dim), bool(two_times)

    def forward(self, cond, x, t_emb, s_emb=None):
        parts = [cond, x, self.time_emb(t_emb)]
        if self.two_times:
            parts.append(self.time_emb(s_emb))
        return self.network(torch.cat(parts, dim=-1))


class CPModel(nn.Module):
    """Optional pixel encoder (+ per-state cond vector via ._cond) and a CPHead."""

    def __init__(self, action_dim: int, *, state_dim: int | None = None, in_channels: int | None = None,
                 cond_dim: int = 0, encoder_kwargs: dict | None = None, time_emb_dim: int = 128,
                 network_kind: str = "mlp", width: int | None = None, depth: int | None = None,
                 two_times: bool = False, dropout: float = 0.0) -> None:
        super().__init__()
        if (state_dim is None) == (in_channels is None):
            raise ValueError("pass exactly one of state_dim (flat) or in_channels (pixels)")
        self.cond_dim = int(cond_dim)
        self._cond: torch.Tensor | None = None
        if in_channels is not None:
            from wifi_bc.models import _build_pixel_encoder
            ek = dict(encoder_kind="conv_maxpool", encoder_target_height=180, encoder_target_width=240,
                      encoder_feature_dim=256, encoder_pretrained=False, encoder_num_kp=64,
                      encoder_norm_kind="bn", encoder_per_camera=False)
            ek.update(encoder_kwargs or {})
            self.encoder, feat = _build_pixel_encoder(
                ek["encoder_kind"], int(in_channels), int(ek["encoder_target_height"]), int(ek["encoder_target_width"]),
                int(ek["encoder_feature_dim"]), ek["encoder_pretrained"], int(ek["encoder_num_kp"]),
                ek["encoder_norm_kind"], bool(ek["encoder_per_camera"]))
        else:
            self.encoder, feat = None, int(state_dim)
        self.head = CPHead(feat + self.cond_dim, action_dim, time_emb_dim=time_emb_dim, network_kind=network_kind,
                           width=width, depth=depth, two_times=two_times, dropout=dropout)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        f = self.encoder(obs) if self.encoder is not None else obs
        if self.cond_dim:
            if self._cond is None:
                raise RuntimeError("cond_dim > 0 but ._cond not set")
            f = torch.cat([f, self._cond], dim=-1)
        return f


@torch.no_grad()
def warm_start_student(student: CPModel, teacher: CPModel) -> None:
    """base_workspace.load_payload(update_dict_dim=...): copy the teacher, zero the stop-time columns."""
    if teacher.encoder is not None:
        student.encoder.load_state_dict(teacher.encoder.state_dict())
    with_params = lambda net: [m for m in net.modules() if len(list(m.parameters(recurse=False))) or len(list(m.buffers(recurse=False)))]
    tm, sm = with_params(teacher.head.network), with_params(student.head.network)
    if len(tm) != len(sm):
        raise RuntimeError(f"teacher/student head structure differs ({len(tm)} vs {len(sm)} parametrised modules)")
    for i, (a, b) in enumerate(zip(tm, sm)):
        for name, p in a.named_parameters(recurse=False):
            q = getattr(b, name)
            if q.shape == p.shape:
                q.copy_(p)
            elif i == 0 and name == "weight" and q.shape[0] == p.shape[0] and q.shape[1] == p.shape[1] + student.head.time_emb_dim:
                q.zero_()
                q[:, :p.shape[1]].copy_(p)
            else:
                raise RuntimeError(f"cannot warm-start {name}: teacher {tuple(p.shape)} vs student {tuple(q.shape)}")
        for name, buf in a.named_buffers(recurse=False):
            getattr(b, name).copy_(buf)


# ── EDM / CTM maps ────────────────────────────────────────────────────────────
def edm_denoise(sched: KarrasSchedule, head: CPHead, cond, x, sigma, clamp: bool = False):
    c_skip, c_out, c_in = sched.scalings(sigma)
    out = head(cond, x * c_in[:, None], sched.noise_emb(sigma)) * c_out[:, None] + x * c_skip[:, None]
    return out.clamp(-1.0, 1.0) if clamp else out


def ctm_G(sched: KarrasSchedule, head: CPHead, cond, x, t, s, clamp: bool = False):
    """CTM_calc_out: G(x, t, s) = (s/t) x + (1 - s/t) g(x, t, s)."""
    c_skip, c_out, c_in = sched.scalings(t)
    g = head(cond, x * c_in[:, None], sched.noise_emb(t), sched.noise_emb(s)) * c_out[:, None] + x * c_skip[:, None]
    ratio = (s / t)[:, None]
    out = x * ratio + g * (1.0 - ratio)
    return out.clamp(-1.0, 1.0) if clamp else out


@torch.no_grad()
def heun_step(sched: KarrasSchedule, head: CPHead, cond, x, t, t_next, clamp: bool = False, zero_safe: bool = True):
    """Karras Heun step; zero_safe mirrors CTM_Scheduler.heun_solver (a zero step returns x)."""
    step = (t_next - t)[:, None]
    mask = (step == 0).float() if zero_safe else torch.zeros_like(step)
    d = (x - edm_denoise(sched, head, cond, x, t, clamp)) / (t[:, None] + mask)
    x1 = x + step * d
    d1 = (x1 - edm_denoise(sched, head, cond, x1, t_next, clamp)) / (t_next[:, None] + mask)
    xn = x + step * (d + d1) / 2
    return xn * (1 - mask) + x * mask


# ── Losses ────────────────────────────────────────────────────────────────────
def teacher_loss(sched: KarrasSchedule, head: CPHead, cond, x0, delta: float = -1.0):
    """EDMPolicy.compute_loss: log-normal sigma, Karras weights, pseudo-Huber to x0."""
    sigma = sched.log_normal(x0.shape[0], x0.device)
    xt = x0 + sigma[:, None] * torch.randn_like(x0)
    return pseudo_huber(edm_denoise(sched, head, cond, xt, sigma), x0, delta, sched.karras_weights(sigma))


def student_losses(sched: KarrasSchedule, student: CPHead, target: CPHead, teacher: CPHead, cond, x0,
                   delta: float = -1.0, ode_steps_max: int = 1):
    """CTMPPUnetHybridImagePolicy.compute_loss -> (ctm, dsm), unweighted."""
    B, dev = x0.shape[0], x0.device
    ti, si, ui = sched.ctm_tsu(B, dev, ode_steps_max)
    t, s, u = sched.times(ti), sched.times(si), sched.times(ui)
    xt = x0 + t[:, None] * torch.randn_like(x0)
    with torch.no_grad():
        xu = xt
        for d in range(int(ode_steps_max)):
            ct = torch.minimum(ti + d, ui)
            nt = torch.minimum(ti + d + 1, ui)
            xu = heun_step(sched, teacher, cond, xu, sched.times(ct), sched.times(nt))
    smin = torch.full_like(t, sched.sigma_min)
    pred = ctm_G(sched, student, cond, xt, t, s)                       # t -> s   (student, grad)
    with torch.no_grad():
        tgt = ctm_G(sched, target, cond, xu, u, s)                     # u -> s   (target, stopgrad)
        tgt0 = ctm_G(sched, target, cond, tgt, s, smin)                # s -> 0
    pred0 = ctm_G(sched, target, cond, pred, s, smin)                  # s -> 0   (grad flows through pred)
    l_ctm = pseudo_huber(pred0, tgt0, delta)
    sig = sched.dsm_student_times(B, dev)
    xs = x0 + sig[:, None] * torch.randn_like(x0)
    l_dsm = pseudo_huber(ctm_G(sched, student, cond, xs, sig, torch.full_like(sig, sched.sigma_min)), x0,
                         delta, sched.karras_weights(sig))
    return l_ctm, l_dsm


# ── Samplers ──────────────────────────────────────────────────────────────────
def parse_chaining(spec) -> tuple[str, list[float]] | None:
    """'none' | 'D:27,54' | 'C:27,54' | ['D', 27, 54]  ->  (mode, times) or None."""
    if spec is None or (isinstance(spec, str) and spec.strip().lower() in ("", "none", "1")):
        return None
    if isinstance(spec, str):
        mode, _, rest = spec.partition(":")
        return mode.strip().upper(), [float(v) for v in rest.split(",") if v.strip()]
    return str(spec[0]).upper(), [float(v) for v in spec[1:]]


@torch.no_grad()
def sample_student(sched: KarrasSchedule, head: CPHead, cond, action_dim: int, chaining=None):
    """CTM conditional_sample: z ~ N(0, I) (not scaled by sigma_max), one jump T -> 0, optional chaining."""
    B, dev = cond.shape[0], cond.device
    x = torch.randn(B, action_dim, device=dev)
    smin = torch.full((B,), sched.sigma_min, device=dev)
    out = ctm_G(sched, head, cond, x, torch.full((B,), sched.sigma_max, device=dev), smin, clamp=True)
    chain = parse_chaining(chaining)
    if chain is not None:
        mode, ts = chain
        for tv in ts:
            t = torch.full((B,), float(tv), device=dev)
            if mode == "C":
                t = sched.times(t)
            out = ctm_G(sched, head, cond, out + t[:, None] * torch.randn_like(out), t, smin, clamp=True)
    return out


@torch.no_grad()
def sample_teacher(sched: KarrasSchedule, head: CPHead, cond, action_dim: int):
    """EDM conditional_sample: z ~ N(0, I), Heun over all bins, denoiser clamped."""
    B, dev = cond.shape[0], cond.device
    x = torch.randn(B, action_dim, device=dev)
    for b in range(sched.bins - 1):
        t = sched.times(torch.full((B,), b, device=dev))
        tn = sched.times(torch.full((B,), b + 1, device=dev))
        x = heun_step(sched, head, cond, x, t, tn, clamp=True, zero_safe=False)
    return x


# ── Optimisation helpers ──────────────────────────────────────────────────────
def ema_power_decay(step: int, inv_gamma: float = 1.0, power: float = 0.75, min_value: float = 0.0,
                    max_value: float = 0.9999, update_after_step: int = 0) -> float:
    """diffusion_policy EMAModel.get_decay (the teacher's evaluation EMA)."""
    step = max(0, step - update_after_step - 1)
    if step <= 0:
        return 0.0
    return max(min_value, min(1 - (1 + step / inv_gamma) ** -power, max_value))


def cosine_with_warmup(warmup: int, total: int):
    """diffusers get_cosine_schedule_with_warmup (num_cycles=0.5), as an LR multiplier."""
    def f(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return f


@torch.no_grad()
def ema_update(ema: nn.Module, src: nn.Module, decay: float) -> None:
    for e, p in zip(ema.parameters(), src.parameters()):
        e.mul_(decay).add_(p.detach(), alpha=1.0 - decay)
    for e, b in zip(ema.buffers(), src.buffers()):
        # SpatialSoftmax grids materialise lazily on the first forward, i.e. after
        # the deepcopy that made `ema`; adopt the source shape once (as the diffusion proposal does).
        if e.shape != b.shape:
            e.resize_(b.shape)
        e.copy_(b)


# ── Evaluation wrapper ────────────────────────────────────────────────────────
class ConsistencyPolicyGenerator(nn.Module):
    """A Consistency Policy behind WiFI-BC's `control_point_generator(state) -> (B, N, A)` contract, N = 1."""

    def __init__(self, model: CPModel, sched: KarrasSchedule, action_dim: int, action_bounds=(-1.0, 1.0),
                 mode: str = "student", chaining=None) -> None:
        super().__init__()
        if mode not in ("student", "teacher"):
            raise ValueError(f"mode must be student|teacher, got {mode!r}")
        self.model, self.sched, self.action_dim, self.mode, self.chaining = model, sched, int(action_dim), mode, chaining
        self.lo, self.hi = float(action_bounds[0]), float(action_bounds[1])
        self.control_points = 1
        self._cond: torch.Tensor | None = None

    @torch.no_grad()
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        if self._cond is not None and self.model.cond_dim:
            self.model._cond = self._cond
        cond = self.model.encode(states)
        if self.mode == "student":
            x = sample_student(self.sched, self.model.head, cond, self.action_dim, self.chaining)
        else:
            x = sample_teacher(self.sched, self.model.head, cond, self.action_dim)
        return ((x + 1.0) * 0.5 * (self.hi - self.lo) + self.lo).unsqueeze(1)
