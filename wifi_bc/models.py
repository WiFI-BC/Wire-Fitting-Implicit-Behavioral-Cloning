import torch
from torch import nn
from torch.nn.utils import spectral_norm
from typing import Sequence


def _make_linear(in_dim: int, out_dim: int, use_spectral_norm: bool = False) -> nn.Module:
	layer = nn.Linear(in_dim, out_dim)
	return spectral_norm(layer) if use_spectral_norm else layer


class ResNetPreActivationBlock(nn.Module):
	"""Pre-activation residual block (Florence et al., 2021, ResNetPreActivation).

	One block: y = activation(x); y = Linear(y); y = activation(y); y = Linear(y); return x + y.
	No normalization (the IBC paper's `.gin` configs set `ResNetLayer.normalizer = None`).
	"""

	def __init__(
		self,
		width: int,
		activation: type[nn.Module] = nn.ReLU,
		use_spectral_norm: bool = False,
	) -> None:
		super().__init__()
		self.act1 = activation()
		self.linear1 = _make_linear(width, width, use_spectral_norm)
		self.act2 = activation()
		self.linear2 = _make_linear(width, width, use_spectral_norm)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		y = self.act1(x)
		y = self.linear1(y)
		y = self.act2(y)
		y = self.linear2(y)
		return x + y


def _build_backbone(
	input_dim: int,
	output_dim: int,
	*,
	network_kind: str,
	hidden_dims: Sequence[int],
	width: int | None,
	depth: int | None,
	activation: type[nn.Module],
	use_spectral_norm: bool,
	resnet_final_activation: bool = True,
) -> nn.Sequential:
	"""Construct either a plain MLP or a ResNetPreActivation backbone.

	- network_kind == "mlp": stacks Linear/activation pairs with `hidden_dims`.
	- network_kind == "resnet": Linear(input -> width); `depth` ResNetPreActivation
	  blocks of width `width`; [optional activation]; Linear(width -> output).
	  Official IBC (networks/mlp_ebm.py) projects to the energy straight after
	  the last block — NO trailing activation (a ReLU there zeroes half the
	  features feeding the energy head). Pass resnet_final_activation=False for
	  the IBC-faithful head; default True preserves checkpoints trained before
	  this option existed (the flag is stateless, so state_dicts stay loadable
	  either way — but the forward pass differs).
	"""
	if network_kind == "mlp":
		layers = []
		prev = input_dim
		for dim in hidden_dims:
			layers.append(_make_linear(prev, dim, use_spectral_norm))
			layers.append(activation())
			prev = dim
		layers.append(_make_linear(prev, output_dim, use_spectral_norm))
		return nn.Sequential(*layers)

	if network_kind == "resnet":
		assert width is not None and depth is not None and depth >= 1, \
			"resnet kind requires width and depth >= 1"
		layers: list[nn.Module] = [_make_linear(input_dim, width, use_spectral_norm)]
		for _ in range(depth):
			layers.append(ResNetPreActivationBlock(width, activation, use_spectral_norm))
		if resnet_final_activation:
			layers.append(activation())
		layers.append(_make_linear(width, output_dim, use_spectral_norm))
		return nn.Sequential(*layers)

	raise ValueError(f"Unknown network_kind: {network_kind!r}. Expected 'mlp' or 'resnet'.")


class ControlPointGenerator(nn.Module):
	"""Produces multiple candidate action vectors per state."""

	def __init__(
		self,
		input_dim: int,
		output_dim: int,
		hidden_dims: Sequence[int] = (256, 256),
		activation: type[nn.Module] = nn.ReLU,
		control_points: int = 10,
		action_bounds: tuple[float, float] = (-1.0, 1.0),
		network_kind: str = "mlp",
		width: int | None = None,
		depth: int | None = None,
		use_spectral_norm: bool = False,
		output_activation: str = "tanh",
	) -> None:
		super().__init__()
		self.output_dim = output_dim
		self.control_points = control_points
		self.action_min = action_bounds[0]
		self.action_max = action_bounds[1]
		# "tanh" squashes into the action box (the historical behaviour, and the
		# default so every existing checkpoint is unchanged). "linear" emits the
		# raw head output and leaves bounding to the consumer (every simulation
		# clips, DFO and Langevin clamp). tanh costs precision near the box
		# edges: its gradient vanishes exactly where a target coordinate sits
		# close to a bound. On particle-16D, where goals are uniform in [0,1]^16
		# so some coordinate is almost always near an edge, the same 256x2 MLP
		# scores 0% with tanh and 69% with a linear head (held-out action L2
		# 0.186 vs 0.047) — tanh is what capped WiFI-BC's argmax there.
		if output_activation not in ("tanh", "linear"):
			raise ValueError(f"output_activation must be tanh|linear, got {output_activation!r}")
		self.output_activation = output_activation

		self.network = _build_backbone(
			input_dim=input_dim,
			output_dim=output_dim * control_points,
			network_kind=network_kind,
			hidden_dims=hidden_dims,
			width=width,
			depth=depth,
			activation=activation,
			use_spectral_norm=use_spectral_norm,
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		batch = x.shape[0]
		out = self.network(x)
		out = out.view(batch, self.control_points, self.output_dim)
		if self.output_activation == "linear":
			return out
		# Tanh maps to [-1, 1], then scale to [action_min, action_max]
		out = torch.tanh(out) * ((self.action_max - self.action_min) / 2) + (self.action_max + self.action_min) / 2
		return out

class QEstimator(nn.Module):
	"""State-conditioned Q-network that maps (state, action) pairs to Q-values."""

	def __init__(
		self,
		state_dim: int,
		action_dim: int,
		output_dim: int = 1,
		hidden_dims: Sequence[int] = (256, 256),
		activation: type[nn.Module] = nn.ReLU,
		dropout_rate: float = 0.0,
		use_spectral_norm: bool = False,
		init_mode: str = "default",
		init_std: float = 0.05,
		network_kind: str = "mlp",
		width: int | None = None,
		depth: int | None = None,
		resnet_final_activation: bool = True,
	) -> None:
		super().__init__()
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.dropout_rate = dropout_rate
		self.use_spectral_norm = use_spectral_norm
		self.init_mode = init_mode
		self.init_std = init_std
		self.resnet_final_activation = resnet_final_activation

		# ResNet pre-activation backbone has no native dropout slot; the IBC paper's
		# configs do not enable dropout on EBM. We honour dropout_rate only for MLP.
		if network_kind == "mlp" and dropout_rate > 0.0:
			layers: list[nn.Module] = []
			prev_dim = state_dim + action_dim
			for dim in hidden_dims:
				layers.append(_make_linear(prev_dim, dim, use_spectral_norm))
				layers.append(activation())
				layers.append(nn.Dropout(p=dropout_rate))
				prev_dim = dim
			layers.append(_make_linear(prev_dim, output_dim, use_spectral_norm))
			self.network = nn.Sequential(*layers)
		else:
			self.network = _build_backbone(
				input_dim=state_dim + action_dim,
				output_dim=output_dim,
				network_kind=network_kind,
				hidden_dims=hidden_dims,
				width=width,
				depth=depth,
				activation=activation,
				use_spectral_norm=use_spectral_norm,
				resnet_final_activation=resnet_final_activation,
			)
		self._init_parameters()

	def _init_parameters(self) -> None:
		"""Initialize model parameters.

		Modes:
		- default: keep PyTorch defaults
		- normal: Normal(0, init_std) for weights and biases
		"""
		if self.init_mode == "default":
			return

		if self.init_mode != "normal":
			raise ValueError(f"Unsupported init_mode: {self.init_mode}")

		for module in self.modules():
			if isinstance(module, nn.Linear):
				nn.init.normal_(module.weight, mean=0.0, std=self.init_std)
				if module.bias is not None:
					nn.init.normal_(module.bias, mean=0.0, std=self.init_std)

	def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
		"""
		Args:
			state: State tensor of shape (B, state_dim) or (B, N, state_dim)
			action: Action tensor of shape (B, action_dim) or (B, N, action_dim)
		Returns:
			Q-values of shape (B, 1) or (B, N, 1)
		"""
		x = torch.cat([state, action], dim=-1)
		return self.network(x)


# ────────────────────────────────────────────────────────────────────────────
# Pixel networks — ports of IBC's `networks/pixel_ebm.py`, `conv_maxpool.py`,
# and `dense_resnet_value.py` (Florence et al. 2021, `pushing_pixels/
# pixel_ebm_langevin.gin`). Used by active_env="pushing_pixels".
#
# Late fusion: the conv encoder runs ONCE per state, then the resulting
# feature vector is broadcast over the N candidate actions before late-fused
# into the value network. Critical for Langevin inner loops (many actions /
# state) — without it, you re-run the conv tower N times per step.
# ────────────────────────────────────────────────────────────────────────────


class ConvMaxpoolEncoder(nn.Module):
	"""4-layer Conv2D + MaxPool + GlobalAvgPool image encoder.

	Mirrors IBC's `get_conv_maxpool` (networks/layers/conv_maxpool.py): filters
	[32, 64, 128, 256], all kernel=3x3 padding=same ReLU, MaxPool2D(2,2) after
	each conv, then GlobalAveragePooling2D → 256-D feature vector.

	Input pipeline (IBC's `image_prepro.preprocess`):
	  1. uint8 → float32 in [0, 1].
	  2. bilinear resize to (target_h, target_w) = (180, 240) per the gin.

	`in_channels` is `3 * frame_stack` because frames are stacked channel-wise
	upstream — for sequence_length=2 we get a 6-channel image.
	"""

	def __init__(
		self,
		in_channels: int = 6,
		target_height: int = 180,
		target_width: int = 240,
		feature_dim: int = 256,
	) -> None:
		super().__init__()
		if feature_dim != 256:
			raise ValueError(
				"ConvMaxpoolEncoder mirrors IBC's get_conv_maxpool which outputs "
				"a 256-D vector (last conv = 256 filters → GlobalAvgPool). "
				f"Got feature_dim={feature_dim}."
			)
		self.target_height = target_height
		self.target_width = target_width

		self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
		self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
		self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
		self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
		self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
		self.relu = nn.ReLU(inplace=True)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""Args: x of shape (B, C, H, W), uint8 or float (any range).

		Returns: (B, 256) feature vector.
		"""
		if x.dtype == torch.uint8:
			x = x.float() / 255.0
		elif x.max() > 1.5:  # accept already-float but not-yet-scaled input
			x = x / 255.0
		if x.shape[-2:] != (self.target_height, self.target_width):
			x = nn.functional.interpolate(
				x, size=(self.target_height, self.target_width),
				mode="bilinear", align_corners=False,
			)
		x = self.pool(self.relu(self.conv1(x)))
		x = self.pool(self.relu(self.conv2(x)))
		x = self.pool(self.relu(self.conv3(x)))
		x = self.pool(self.relu(self.conv4(x)))
		# Global average pooling over spatial dims.
		x = x.mean(dim=[2, 3])  # (B, 256)
		return x


class _DenseResnetBlock(nn.Module):
	"""Bottleneck residual block: width/4 → width/4 → width, ReLU pre-activation.

	Port of IBC's ResNetDenseBlock (networks/layers/dense_resnet_value.py).
	No batch/layer norm (IBC's value config doesn't enable any).
	"""

	def __init__(self, width: int) -> None:
		super().__init__()
		self.dense0 = nn.Linear(width, width // 4)
		self.dense1 = nn.Linear(width // 4, width // 4)
		self.dense2 = nn.Linear(width // 4, width)
		self.dense3 = nn.Linear(width, width)  # projection if shapes mismatch
		self.act = nn.ReLU(inplace=True)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		y = self.dense0(self.act(x))
		y = self.dense1(self.act(y))
		y = self.dense2(self.act(y))
		# In IBC's impl, x's shape never mismatches y's once dense0 fixes width.
		# Defensive projection kept for parity with their `if x.shape != y.shape`.
		if x.shape[-1] != y.shape[-1]:
			x = self.dense3(self.act(x))
		return x + y


class DenseResnetValue(nn.Module):
	"""Dense + N ResNetBlocks + Dense(1). Port of IBC's DenseResnetValue.

	IBC's `pushing_pixels/pixel_ebm_langevin.gin` sets width=1024, num_blocks=1.
	`Normal(0, 0.05)` init on every Dense — matches IBC's `kernel_initializer='normal'`
	and `bias_initializer='normal'`, which in Keras both default to
	`RandomNormal(mean=0.0, stddev=0.05)`. Initial port used std=1.0; with a
	1024-wide hidden layer that produced ~32-std initial activations and
	prevented the Q estimator from learning (saw ~6% argmax-pick of the
	closest-to-expert CP). std=0.05 keeps activations in a sane range.
	"""

	_INIT_STD = 0.05

	def __init__(self, in_dim: int, width: int = 1024, num_blocks: int = 1) -> None:
		super().__init__()
		self.dense0 = nn.Linear(in_dim, width)
		self.blocks = nn.ModuleList(_DenseResnetBlock(width) for _ in range(num_blocks))
		self.dense1 = nn.Linear(width, 1)
		self._init_parameters()

	def _init_parameters(self) -> None:
		for m in self.modules():
			if isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, mean=0.0, std=self._INIT_STD)
				if m.bias is not None:
					nn.init.normal_(m.bias, mean=0.0, std=self._INIT_STD)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.dense0(x)
		for block in self.blocks:
			x = block(x)
		return self.dense1(x)  # (..., 1)


class SpatialSoftmax(nn.Module):
	"""Spatial softmax over conv feature maps → per-channel expected (x, y).

	Standard manipulation-policy head (Levine et al. 2016; used by robomimic /
	the LIBERO paper's ResNet encoders). Unlike GlobalAvgPool it PRESERVES
	spatial layout — each output channel reports where its feature is in the
	image, which is exactly what grasp/manipulation policies need.

	Input (B, C, H, W) → output (B, 2C): [E[x], E[y]] per channel, in [-1, 1].
	"""

	def __init__(self) -> None:
		super().__init__()
		self._grid_hw: tuple[int, int] | None = None
		self.register_buffer("_pos_x", torch.empty(0), persistent=False)
		self.register_buffer("_pos_y", torch.empty(0), persistent=False)

	def _build_grid(self, h: int, w: int, device, dtype) -> None:
		ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
		xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
		gy, gx = torch.meshgrid(ys, xs, indexing="ij")
		self._pos_x = gx.reshape(1, 1, h * w)
		self._pos_y = gy.reshape(1, 1, h * w)
		self._grid_hw = (h, w)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		b, c, h, w = x.shape
		if self._grid_hw != (h, w) or self._pos_x.device != x.device:
			self._build_grid(h, w, x.device, x.dtype)
		attn = torch.softmax(x.reshape(b, c, h * w), dim=-1)
		ex = (attn * self._pos_x).sum(-1)  # (B, C)
		ey = (attn * self._pos_y).sum(-1)
		return torch.cat([ex, ey], dim=-1)  # (B, 2C)


class ResNet18SpatialSoftmaxEncoder(nn.Module):
	"""torchvision ResNet-18 (optionally ImageNet-pretrained) + SpatialSoftmax.

	The LIBERO-standard BC image encoder (per the benchmark's ResNet policies):
	ResNet-18 trunk to layer4, 1×1 conv down to `num_kp` keypoint channels,
	spatial softmax → 2*num_kp coords per image.

	Input is the channel-stacked uint8 tensor (B, 3*n_imgs, H, W) the pixel
	datasets emit (n_imgs = n_cameras * frame_stack). A PRETRAINED trunk expects
	3-channel RGB, so we split into n_imgs frames, run the SHARED trunk on each
	(batched as B*n_imgs), and concatenate features → feature_dim =
	n_imgs * 2 * num_kp. Preprocessing: uint8 → /255 → ImageNet mean/std.
	"""

	_IMAGENET_MEAN = (0.485, 0.456, 0.406)
	_IMAGENET_STD = (0.229, 0.224, 0.225)

	# Per-stage output channels of the ResNet-18 trunk (layer1..layer4).
	_STAGE_CHANNELS = (64, 128, 256, 512)

	def __init__(
		self,
		in_channels: int = 6,
		pretrained: bool | str = True,
		num_kp: int = 64,
		norm_kind: str = "bn",
		per_camera: bool = False,
		film_dim: int = 0,
	) -> None:
		super().__init__()
		if in_channels % 3 != 0:
			raise ValueError(f"in_channels must be a multiple of 3 (RGB frames); got {in_channels}")
		self.n_imgs = in_channels // 3
		self.num_kp = num_kp
		self.norm_kind = norm_kind
		self.per_camera = per_camera
		self.film_dim = int(film_dim)
		self.feature_dim = self.n_imgs * 2 * num_kp

		try:
			import torchvision
		except ImportError as e:
			raise ImportError(
				"torchvision is required for encoder_kind='resnet18' "
				"(uv sync --extra libero after adding it to pyproject)."
			) from e

		def _make_trunk() -> nn.Sequential:
			# pretrained: True/"imagenet" → torchvision ImageNet weights;
			# "r3m" → local R3M ResNet-18 weights (Ego4D manipulation
			# pretraining, Nair et al. 2022); False → scratch.
			use_imagenet = pretrained in (True, "imagenet")
			weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if use_imagenet else None
			t = torchvision.models.resnet18(weights=weights)
			if pretrained == "r3m":
				self._load_r3m_(t)
			return nn.Sequential(
				t.conv1, t.bn1, t.relu, t.maxpool,
				t.layer1, t.layer2, t.layer3, t.layer4,
			)

		if per_camera:
			# One trunk per image slot (robomimic/DP convention: separate
			# encoder per camera view). n_imgs = n_cams * frame_stack.
			self.trunks = nn.ModuleList(_make_trunk() for _ in range(self.n_imgs))
			_norm_targets = list(self.trunks)
		else:
			self.trunk = _make_trunk()
			_norm_targets = [self.trunk]
		# ── Normalization strategy ─────────────────────────────────────────
		# Raw BatchNorm is hostile to EBM/InfoNCE training: energies are
		# computed with batch stats at train time but running stats at eval,
		# shifting the whole Q landscape (Bstandardlibero: ResNet-BN lost to a
		# norm-free ConvMaxpool across the board).
		#   - "gn":        replace every BatchNorm2d with GroupNorm(C//16, C)
		#                  (Diffusion Policy's recipe). Conv weights keep the
		#                  pretrained values; BN affine γ/β are copied into GN.
		#                  Per-sample stats → zero train/eval mismatch.
		#   - "bn_frozen": keep BN but lock it to eval mode forever (ImageNet
		#                  running stats, frozen). Preserves pretrained behavior
		#                  exactly and also removes the mismatch.
		#   - "bn":        stock torchvision behavior (kept for comparability).
		for _t in _norm_targets:
			if norm_kind == "gn":
				self._swap_bn_to_gn(_t)
			elif norm_kind == "bn_frozen":
				self._freeze_bn(_t)
			elif norm_kind != "bn":
				raise ValueError(f"Unknown norm_kind: {norm_kind!r} (bn|gn|bn_frozen)")
		# FiLM conditioning (DP-LIBERO style): the language-goal vector
		# modulates each ResNet stage output with per-channel scale/shift.
		# Only built when film_dim > 0 so legacy checkpoints keep their keys.
		if self.film_dim > 0:
			self.film = nn.ModuleList(
				nn.Linear(self.film_dim, 2 * c) for c in self._STAGE_CHANNELS
			)
			for f in self.film:
				nn.init.zeros_(f.weight)
				nn.init.zeros_(f.bias)  # identity modulation at init
		self.kp_conv = nn.Conv2d(512, num_kp, kernel_size=1)
		self.spatial_softmax = SpatialSoftmax()
		mean = torch.tensor(self._IMAGENET_MEAN).view(1, 3, 1, 1)
		std = torch.tensor(self._IMAGENET_STD).view(1, 3, 1, 1)
		self.register_buffer("_px_mean", mean, persistent=False)
		self.register_buffer("_px_std", std, persistent=False)

	@staticmethod
	def _swap_bn_to_gn(root: nn.Module) -> None:
		"""Replace every BatchNorm2d under *root* with GroupNorm(C//16, C).

		Copies the BN affine (γ, β) into GN so pretrained scale/shift carries
		over; BN running stats have no GN equivalent and are dropped.
		"""
		for parent in root.modules():
			for name, child in list(parent.named_children()):
				if isinstance(child, nn.BatchNorm2d):
					c = child.num_features
					gn = nn.GroupNorm(max(1, c // 16), c)
					with torch.no_grad():
						gn.weight.copy_(child.weight)
						gn.bias.copy_(child.bias)
					setattr(parent, name, gn)

	@staticmethod
	def _freeze_bn(root: nn.Module) -> None:
		for m in root.modules():
			if isinstance(m, nn.BatchNorm2d):
				m.eval()
				m.weight.requires_grad_(False)
				m.bias.requires_grad_(False)

	@staticmethod
	def _load_r3m_(resnet) -> None:
		"""Load R3M ResNet-18 weights (Nair et al. 2022, Ego4D) into *resnet*.

		Expects the checkpoint at $R3M_WEIGHTS or
		checkpoints/pretrained/r3m_resnet18.pt (fetch per the r3m repo README —
		the file torch.load's to {'r3m': state_dict} or a raw state_dict with
		'module.convnet.'-prefixed keys). Loaded BEFORE any GN swap so BN affine
		copies carry R3M's values.
		"""
		import os as _os
		path = _os.environ.get("R3M_WEIGHTS", "checkpoints/pretrained/r3m_resnet18.pt")
		if not _os.path.exists(path):
			raise FileNotFoundError(
				f"R3M weights not found at {path!r}. Download r3m_resnet18 per "
				"https://github.com/facebookresearch/r3m and save it there "
				"(or set $R3M_WEIGHTS)."
			)
		sd = torch.load(path, map_location="cpu", weights_only=False)
		if isinstance(sd, dict) and "r3m" in sd:
			sd = sd["r3m"]
		if hasattr(sd, "state_dict"):
			sd = sd.state_dict()
		cleaned = {}
		for k, v in sd.items():
			k = k.removeprefix("module.")
			if k.startswith("convnet."):
				cleaned[k.removeprefix("convnet.")] = v
		if not cleaned:
			raise RuntimeError(
				f"No 'convnet.' keys in R3M checkpoint {path!r}; "
				f"got keys like {list(sd)[:3]}"
			)
		missing, unexpected = resnet.load_state_dict(cleaned, strict=False)
		# fc.* is expected-missing (we drop the classifier); anything conv/bn
		# missing means a layout mismatch.
		real_missing = [m for m in missing if not m.startswith("fc.")]
		if real_missing:
			raise RuntimeError(f"R3M load missing keys: {real_missing[:5]}")

	def train(self, mode: bool = True):
		"""Keep BN layers in eval mode under norm_kind='bn_frozen'."""
		super().train(mode)
		if self.norm_kind == "bn_frozen":
			for m in self.trunk.modules():
				if isinstance(m, nn.BatchNorm2d):
					m.eval()
		return self

	def _run_trunk(self, trunk: nn.Sequential, x: torch.Tensor, film_vec) -> torch.Tensor:
		"""Run one ResNet trunk, optionally FiLM-modulating each stage output.

		trunk layout: [conv1, bn1, relu, maxpool, layer1..layer4]; FiLM applies
		γ·h + β per channel after each layerN (γ,β from the goal vector; zero-init
		→ identity at start of training).
		"""
		h = trunk[:4](x)
		for i in range(4):
			h = trunk[4 + i](h)
			if self.film_dim > 0 and film_vec is not None:
				gb = self.film[i](film_vec)                     # (B*, 2C)
				gamma, beta = gb.chunk(2, dim=-1)
				h = h * (1.0 + gamma.unsqueeze(-1).unsqueeze(-1)) + beta.unsqueeze(-1).unsqueeze(-1)
		return h

	def forward(self, x: torch.Tensor, film_vec: torch.Tensor | None = None) -> torch.Tensor:
		"""(B, 3*n_imgs, H, W) uint8/float → (B, n_imgs*2*num_kp).

		film_vec: optional (B, film_dim) goal vector for FiLM (film_dim > 0).
		"""
		if x.dtype == torch.uint8:
			x = x.float() / 255.0
		elif x.max() > 1.5:
			x = x / 255.0
		b = x.shape[0]
		x = x.reshape(b, self.n_imgs, 3, x.shape[-2], x.shape[-1])
		x = (x - self._px_mean.unsqueeze(0)) / self._px_std.unsqueeze(0)
		feats = []
		for i in range(self.n_imgs):
			trunk = self.trunks[i] if self.per_camera else self.trunk
			h = self._run_trunk(trunk, x[:, i], film_vec)   # (B, 512, h', w')
			h = self.kp_conv(h)
			feats.append(self.spatial_softmax(h))           # (B, 2*num_kp)
		return torch.cat(feats, dim=-1)                     # (B, n_imgs*2*num_kp)


def _build_pixel_encoder(
	encoder_kind: str,
	in_channels: int,
	encoder_target_height: int,
	encoder_target_width: int,
	encoder_feature_dim: int,
	encoder_pretrained: bool | str,
	encoder_num_kp: int,
	encoder_norm_kind: str = "bn",
	encoder_per_camera: bool = False,
	film_dim: int = 0,
) -> tuple[nn.Module, int]:
	"""Encoder factory shared by both pixel nets. Returns (encoder, feature_dim)."""
	if encoder_kind == "conv_maxpool":
		enc = ConvMaxpoolEncoder(
			in_channels=in_channels,
			target_height=encoder_target_height,
			target_width=encoder_target_width,
			feature_dim=encoder_feature_dim,
		)
		return enc, encoder_feature_dim
	if encoder_kind == "resnet18":
		enc = ResNet18SpatialSoftmaxEncoder(
			in_channels=in_channels,
			pretrained=encoder_pretrained,
			num_kp=encoder_num_kp,
			norm_kind=encoder_norm_kind,
			per_camera=encoder_per_camera,
			film_dim=film_dim,
		)
		return enc, enc.feature_dim
	raise ValueError(f"Unknown encoder_kind: {encoder_kind!r} (conv_maxpool|resnet18)")


class PixelControlPointGenerator(nn.Module):
	"""Image-conditioned CP generator.

	Encoder (ConvMaxpoolEncoder) → 256-D features → ControlPointGenerator-style
	MLP that emits N candidate actions per state. By default the encoder has its
	OWN weights (not shared with PixelQEstimator) — matching IBC's separate
	networks per loss head. Pass `share_encoder_from=<PixelQEstimator>` to reuse
	that net's trunk instead, halving the per-inference conv cost.
	"""

	def __init__(
		self,
		output_dim: int,
		control_points: int,
		hidden_dims: Sequence[int] = (256, 256),
		action_bounds: tuple[float, float] = (-1.0, 1.0),
		network_kind: str = "mlp",
		width: int | None = None,
		depth: int | None = None,
		use_spectral_norm: bool = False,
		activation: type[nn.Module] = nn.ReLU,
		in_channels: int = 6,
		encoder_target_height: int = 180,
		encoder_target_width: int = 240,
		encoder_feature_dim: int = 256,
		cond_dim: int = 0,
		encoder_kind: str = "conv_maxpool",
		encoder_pretrained: bool | str = True,
		encoder_num_kp: int = 64,
		encoder_norm_kind: str = "bn",
		encoder_per_camera: bool = False,
		cond_fusion: str = "concat",
		goal_dim: int = 0,
		share_encoder_from: nn.Module | None = None,
		output_activation: str = "tanh",
	) -> None:
		super().__init__()
		# `cond_dim` > 0 conditions the CP head on an extra per-state vector
		# (e.g. LIBERO proprio + goal-language embedding), concatenated to the
		# conv features. Set via the `_cond` attribute per batch (default None →
		# unconditioned, so pushing_pixels is unaffected with cond_dim=0).
		self.cond_dim = int(cond_dim)
		self._cond: torch.Tensor | None = None
		# cond_fusion="film": the GOAL slice of _cond (last goal_dim dims)
		# additionally FiLM-modulates the ResNet stages; the full _cond still
		# concats at the head (film is additive conditioning).
		if cond_fusion not in ("concat", "film"):
			raise ValueError(f"cond_fusion must be concat|film, got {cond_fusion!r}")
		if cond_fusion == "film" and (goal_dim <= 0 or encoder_kind != "resnet18"):
			raise ValueError("cond_fusion='film' needs encoder_kind='resnet18' and goal_dim>0")
		self.cond_fusion = cond_fusion
		self.goal_dim = int(goal_dim)
		# share_encoder_from: adopt another pixel net's trunk instead of building
		# a second one. The module is registered under BOTH parents, so its
		# parameters appear in both .parameters() and both .state_dict()s --
		# whoever wires this up owns deduplicating the optimizer, the gradient
		# clipping and the EMA (see training/wifi_bc_training.py).
		self.shares_encoder = share_encoder_from is not None
		if share_encoder_from is not None:
			self.encoder = share_encoder_from.encoder
			feat_dim = int(share_encoder_from.encoder_feature_dim)
		else:
			self.encoder, feat_dim = _build_pixel_encoder(
				encoder_kind, in_channels, encoder_target_height,
				encoder_target_width, encoder_feature_dim,
				encoder_pretrained, encoder_num_kp, encoder_norm_kind,
				encoder_per_camera,
				film_dim=(self.goal_dim if cond_fusion == "film" else 0),
			)
		self.encoder_feature_dim = feat_dim
		self.head = ControlPointGenerator(
			input_dim=feat_dim + self.cond_dim,
			output_dim=output_dim,
			hidden_dims=hidden_dims,
			activation=activation,
			control_points=control_points,
			action_bounds=action_bounds,
			network_kind=network_kind,
			width=width,
			depth=depth,
			use_spectral_norm=use_spectral_norm,
			output_activation=output_activation,
		)

	def _film_vec(self) -> torch.Tensor | None:
		if self.cond_fusion != "film":
			return None
		if self._cond is None:
			raise RuntimeError("cond_fusion='film' but ._cond not set.")
		return self._cond[:, -self.goal_dim:]

	def encode(self, images: torch.Tensor) -> torch.Tensor:
		"""Run the conv encoder once per state. Returns (B, feature_dim).

		Mirrors PixelQEstimator.encode so a SHARED trunk can be run once and
		fed to both heads (utils/models.py `share_encoder_from`).
		"""
		fv = self._film_vec()
		return self.encoder(images, fv) if fv is not None else self.encoder(images)

	def head_from_features(self, features: torch.Tensor) -> torch.Tensor:
		"""CP head on pre-encoded features. (B, F) -> (B, control_points, A)."""
		if self.cond_dim > 0:
			if self._cond is None:
				raise RuntimeError("PixelControlPointGenerator.cond_dim>0 but ._cond not set.")
			features = torch.cat([features, self._cond], dim=-1)
		return self.head(features)

	def forward(self, images: torch.Tensor) -> torch.Tensor:
		"""Args: images (B, C, H, W). Returns: (B, control_points, output_dim)."""
		return self.head_from_features(self.encode(images))


class PixelQEstimator(nn.Module):
	"""Image-conditioned Q estimator with late fusion.

	Encoder (ConvMaxpoolEncoder) → 256-D features (run ONCE per state) →
	concat with each candidate action → DenseResnetValue → scalar Q.

	Late fusion accepts two input patterns:
	  - state=(B, C, H, W),    action=(B, A)         → returns (B, 1)
	  - state=(B, C, H, W),    action=(B, N, A)      → returns (B, N, 1)
	Encoder runs once in both cases; features are broadcast in the second.
	"""

	def __init__(
		self,
		action_dim: int,
		in_channels: int = 6,
		encoder_target_height: int = 180,
		encoder_target_width: int = 240,
		encoder_feature_dim: int = 256,
		value_width: int = 1024,
		value_num_blocks: int = 1,
		cond_dim: int = 0,
		encoder_kind: str = "conv_maxpool",
		encoder_pretrained: bool | str = True,
		encoder_num_kp: int = 64,
		encoder_norm_kind: str = "bn",
		encoder_per_camera: bool = False,
		cond_fusion: str = "concat",
		goal_dim: int = 0,
	) -> None:
		super().__init__()
		# `cond_dim` > 0 late-fuses an extra per-state vector (LIBERO proprio +
		# goal embedding) alongside the image features and action. Set via the
		# `_cond` attribute per batch (default None → unconditioned).
		self.cond_dim = int(cond_dim)
		self._cond: torch.Tensor | None = None
		# cond_fusion="film": the GOAL slice of _cond (last goal_dim dims)
		# additionally FiLM-modulates the ResNet stages; the full _cond still
		# concats at the head (film is additive conditioning).
		if cond_fusion not in ("concat", "film"):
			raise ValueError(f"cond_fusion must be concat|film, got {cond_fusion!r}")
		if cond_fusion == "film" and (goal_dim <= 0 or encoder_kind != "resnet18"):
			raise ValueError("cond_fusion='film' needs encoder_kind='resnet18' and goal_dim>0")
		self.cond_fusion = cond_fusion
		self.goal_dim = int(goal_dim)
		self.encoder, feat_dim = _build_pixel_encoder(
			encoder_kind, in_channels, encoder_target_height,
			encoder_target_width, encoder_feature_dim,
			encoder_pretrained, encoder_num_kp, encoder_norm_kind,
			encoder_per_camera,
			film_dim=(self.goal_dim if cond_fusion == "film" else 0),
		)
		self.value = DenseResnetValue(
			in_dim=feat_dim + self.cond_dim + action_dim,
			width=value_width,
			num_blocks=value_num_blocks,
		)
		self.action_dim = action_dim
		self.encoder_feature_dim = feat_dim

	def _film_vec(self) -> torch.Tensor | None:
		if self.cond_fusion != "film":
			return None
		if self._cond is None:
			raise RuntimeError("cond_fusion='film' but ._cond not set.")
		return self._cond[:, -self.goal_dim:]

	def encode(self, images: torch.Tensor) -> torch.Tensor:
		"""Run the conv encoder once per state. Returns (B, feature_dim)."""
		fv = self._film_vec()
		return self.encoder(images, fv) if fv is not None else self.encoder(images)

	def score(self, features: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
		"""Late-fuse pre-encoded features with action(s).

		Args:
		  features: (B, F) — output of `encode(...)`.
		  action:   (B, A) or (B, N, A).
		Returns: (B, 1) or (B, N, 1).
		"""
		if action.ndim == 3:
			B, N, _ = action.shape
			features = features.unsqueeze(1).expand(B, N, -1)
			if self.cond_dim > 0:
				if self._cond is None:
					raise RuntimeError("PixelQEstimator.cond_dim>0 but ._cond not set.")
				cond = self._cond.unsqueeze(1).expand(B, N, -1)
				x = torch.cat([features, cond, action], dim=-1)
				return self.value(x)
		elif self.cond_dim > 0:
			if self._cond is None:
				raise RuntimeError("PixelQEstimator.cond_dim>0 but ._cond not set.")
			x = torch.cat([features, self._cond, action], dim=-1)
			return self.value(x)
		x = torch.cat([features, action], dim=-1)
		return self.value(x)

	def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
		"""Convenience: encode then score, in one call.

		Accepted state shapes:
		  - (B, C, H, W)       — raw image batch (preferred).
		  - (B, N, C, H, W)    — image broadcast over N candidates via
		    `.unsqueeze(1).expand(-1, N, -1, -1, -1)`. We collapse with
		    `state[:, 0]` since `expand` produces a stride-0 view (all N
		    slices share the same memory), so this is a SAFE drop, not a
		    drop of unique data. This lets call sites that were written for
		    flat states (`sample_langevin`, gradient-penalty paths) keep
		    working without per-call-site rewrites.

		Note: prefer the explicit `encode(...) → score(...)` pattern when
		evaluating many actions against the same state — it skips even the
		single slice op.
		"""
		if state.ndim == 5:
			state = state[:, 0]
		return self.score(self.encode(state), action)



class BCPolicy(nn.Module):
    """Explicit behaviour cloning: observation -> action in ONE forward pass.

    A genuinely separate baseline, not a configuration of another method. There
    is no control-point dimension, no energy function and no critic anywhere in
    the module — it is a regression network and nothing else.

    It returns (B, 1, action_dim) rather than (B, action_dim) purely so it drops
    into the evaluation stack's `proposal(state) -> (B, N, A)` contract with
    N = 1. That is presentation, not architecture: nothing is ranked, because
    there is only ever one candidate.

    Handles both observation kinds: pass `in_channels` for pixels (a shared
    encoder is built) or `state_dim` for flat states.
    """

    def __init__(self, action_dim: int, *, in_channels: int | None = None,
                 state_dim: int | None = None, width: int = 256, depth: int = 2,
                 cond_dim: int = 0, action_bounds: tuple[float, float] = (-1.0, 1.0),
                 encoder_target_height: int = 180, encoder_target_width: int = 240,
                 encoder_feature_dim: int = 256, encoder_kind: str = "conv_maxpool",
                 encoder_pretrained: bool | str = False, encoder_num_kp: int = 64,
                 encoder_norm_kind: str = "bn", encoder_per_camera: bool = False,
                 cond_fusion: str = "concat", goal_dim: int = 0,
                 network_kind: str = "mlp") -> None:
        super().__init__()
        if (in_channels is None) == (state_dim is None):
            raise ValueError("pass exactly one of in_channels (pixels) or state_dim (flat)")
        self.action_dim = int(action_dim)
        self.cond_dim = int(cond_dim)
        self._cond: torch.Tensor | None = None
        self.action_bounds = (float(action_bounds[0]), float(action_bounds[1]))
        # Same conditioning options as PixelControlPointGenerator, so the BC
        # baseline can be run with the exact encoder recipe it is compared
        # against — otherwise a gap between BC and WiFI-BC would be confounded by
        # FiLM rather than by the objective, which is the only thing under test.
        if cond_fusion not in ("concat", "film"):
            raise ValueError(f"cond_fusion must be concat|film, got {cond_fusion!r}")
        if cond_fusion == "film" and (goal_dim <= 0 or encoder_kind != "resnet18"):
            raise ValueError("cond_fusion='film' needs encoder_kind='resnet18' and goal_dim>0")
        self.cond_fusion = cond_fusion
        self.goal_dim = int(goal_dim)
        if in_channels is not None:
            self.encoder, feat_dim = _build_pixel_encoder(
                encoder_kind, in_channels, encoder_target_height,
                encoder_target_width, encoder_feature_dim, encoder_pretrained,
                encoder_num_kp, encoder_norm_kind, encoder_per_camera,
                film_dim=(self.goal_dim if cond_fusion == "film" else 0))
        else:
            self.encoder, feat_dim = None, int(state_dim)
            self.cond_fusion = "concat"
        self.head = _build_backbone(
            input_dim=feat_dim + self.cond_dim, output_dim=self.action_dim,
            network_kind=network_kind, hidden_dims=[width] * depth,
            width=width, depth=depth, activation=nn.ReLU, use_spectral_norm=False)

    def _film_vec(self) -> torch.Tensor | None:
        if self.cond_fusion != "film":
            return None
        if self._cond is None:
            raise RuntimeError("cond_fusion='film' but ._cond not set.")
        return self._cond[:, -self.goal_dim:]

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        if self.encoder is None:
            return obs
        fv = self._film_vec()
        return self.encoder(obs, fv) if fv is not None else self.encoder(obs)

    def features(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.encode(obs)
        if self.cond_dim:
            if self._cond is None:
                raise RuntimeError("cond_dim > 0 but ._cond not set")
            x = torch.cat([x, self._cond], dim=-1)
        return x

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        a = self.head(self.features(obs))
        lo, hi = self.action_bounds
        # tanh-squash to the action box, as explicit BC baselines conventionally
        # do; without it a regression head can emit out-of-range actions that
        # the environment silently clips.
        a = torch.tanh(a) * (hi - lo) / 2.0 + (hi + lo) / 2.0
        return a.unsqueeze(1)
