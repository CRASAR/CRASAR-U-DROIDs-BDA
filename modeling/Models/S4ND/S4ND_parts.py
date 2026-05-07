import math
import torch
import torch.nn.functional as F

from torch import nn

from modeling.Models.Normalization import getNormalization, getNormalizationArgCount
from modeling.Models.Activations import getActivationFunction, getActivationChannels, getActivationKwargs

DOUBLE_PRECISION = "double"
FULL_PRECISION = "full"
HALF_PRECISION = "half"
PRECISION_VALUES = [HALF_PRECISION, FULL_PRECISION, DOUBLE_PRECISION]

NO_BANDLIMIT = "none"
HARD_BANDLIMIT = "hard"
SOFT_BANDLIMIT = "soft"

# Multiplication of complex tensors via their component parts represented as real numbers
def complex_mul(ar, ai, br, bi):
    # (a + ib)(c + id)
    return ar * br - ai * bi, ar * bi + ai * br

# Divison of complex tensors via their component parts represented as real numbers
def complex_inv(ar, ai, eps=1e-7):
    # 1 / (a + ib)
    denom = ar * ar + ai * ai + eps
    return ar / denom, -ai / denom

class S4ND(nn.Module):
    def __init__(self,
             N,
             in_channels,
             out_channels,
             use_D=True,
             lambda_scale=-0.1,
             precision="half",
             chunk_size=1,
             band_limit_strategy="none",
             learn_bandlimit=False,
             cut_off_nyquist_proportion=0.25,
             bandlimit_taper_width=0.1,
             learn_frequency_importances=False,
             frequency_importance_hidden_dim=32):
        super().__init__()
        self.s4nd = nn.Sequential(
                S4ND_Kernel(channels=in_channels,
                            N=N,
                            use_D=use_D,
                            lambda_scale=lambda_scale,
                            precision=precision,
                            chunk_size=chunk_size,
                            band_limit_strategy=band_limit_strategy,
                            learn_bandlimit=learn_bandlimit,
                            cut_off_nyquist_proportion=cut_off_nyquist_proportion,
                            bandlimit_taper_width=bandlimit_taper_width,
                            learn_frequency_importances=learn_frequency_importances,
                            frequency_importance_hidden_dim=frequency_importance_hidden_dim),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.s4nd(x)

class DoubleS4ND(nn.Module):
    def __init__(self,
                 N,
                 in_channels,
                 out_channels,
                 mid_channels=None,
                 use_D=True,
                 lambda_scale=-0.1,
                 precision="half",
                 chunk_size=1,
                 band_limit_strategy="none",
                 learn_bandlimit=False,
                 cut_off_nyquist_proportion=0.25,
                 bandlimit_taper_width=0.1,
                 learn_frequency_importances=False,
                 frequency_importance_hidden_dim=32,
                 activation="glu",
                 normalization="DynamicGroupLayerNorm"):
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels
        norm_arg_count = getNormalizationArgCount(normalization)
        mid_norm_args = [mid_channels*getActivationChannels(activation)]*norm_arg_count
        out_norm_args = [out_channels*getActivationChannels(activation)]*norm_arg_count
        self.double_s4nd = nn.Sequential(
                S4ND_Kernel(channels=in_channels,
                            N=N,
                            use_D=use_D,
                            lambda_scale=lambda_scale,
                            precision=precision,
                            chunk_size=chunk_size,
                            band_limit_strategy=band_limit_strategy,
                            learn_bandlimit=learn_bandlimit,
                            cut_off_nyquist_proportion=cut_off_nyquist_proportion,
                            bandlimit_taper_width=bandlimit_taper_width,
                            learn_frequency_importances=learn_frequency_importances,
                            frequency_importance_hidden_dim=frequency_importance_hidden_dim),
                nn.Conv2d(in_channels, mid_channels*getActivationChannels(activation), kernel_size=1, bias=False),
                getActivationFunction(activation)(**getActivationKwargs(activation)),
                getNormalization(normalization)(*mid_norm_args),
                S4ND_Kernel(channels=mid_channels,
                            N=N,
                            use_D=use_D,
                            lambda_scale=lambda_scale,
                            precision=precision,
                            chunk_size=chunk_size,
                            band_limit_strategy=band_limit_strategy,
                            learn_bandlimit=learn_bandlimit,
                            cut_off_nyquist_proportion=cut_off_nyquist_proportion,
                            bandlimit_taper_width=bandlimit_taper_width,
                            learn_frequency_importances=learn_frequency_importances,
                            frequency_importance_hidden_dim=frequency_importance_hidden_dim),
                nn.Conv2d(mid_channels, out_channels*getActivationChannels(activation), kernel_size=1, bias=False),
                getActivationFunction(activation)(**getActivationKwargs(activation)),
                getNormalization(normalization)(*out_norm_args))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_s4nd(x)

class S4NDDown(nn.Module):
    def __init__(self,
                 N,
                 in_channels,
                 out_channels,
                 use_D=True,
                 lambda_scale=-0.1,
                 precision="half",
                 chunk_size=1,
                 band_limit_strategy="none",
                 learn_bandlimit=False,
                 cut_off_nyquist_proportion=0.25,
                 bandlimit_taper_width=0.1,
                 learn_frequency_importances=False,
                 frequency_importance_hidden_dim=32,
                 activation="GLU",
                 normalization="DynamicGroupLayerNorm"):
        super().__init__()
        self.avgpool_s4nd = nn.Sequential(
            nn.AvgPool2d(2),
            DoubleS4ND(N,
                       in_channels,
                       out_channels,
                       mid_channels=None,
                       use_D=use_D,
                       lambda_scale=lambda_scale,
                       precision=precision,
                       chunk_size=chunk_size,
                       band_limit_strategy=band_limit_strategy,
                       learn_bandlimit=learn_bandlimit,
                       cut_off_nyquist_proportion=cut_off_nyquist_proportion,
                       bandlimit_taper_width=bandlimit_taper_width,
                       learn_frequency_importances=learn_frequency_importances,
                       frequency_importance_hidden_dim=frequency_importance_hidden_dim,
                       activation=activation,
                       normalization=normalization))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.avgpool_s4nd(x)

@torch.jit.script
def memory_efficient_s4nd_transition_func(C:int,
                                          H:int,
                                          W:int,
                                          N:int,
                                          exp_x:torch.Tensor,
                                          exp_y:torch.Tensor,
                                          lambda_x:torch.Tensor,
                                          lambda_y:torch.Tensor,
                                          Bc:torch.Tensor,
                                          Cc:torch.Tensor,
                                          eps:float,
                                          device:torch.device):
    # Loop over channels, compute H_f for a single channel at a time (vectorized over N and H×W)
    # This minimizes peak memory: we only allocate [H, W, N] per channel.
    H_f_combined = torch.zeros((1,C,H,W)).to(device)
    for cur_channel in range(C):
        # grab per-channel complex vectors: shape [N]
        lam_x_ch = lambda_x[cur_channel]   # complex [N]
        lam_y_ch = lambda_y[cur_channel]   # complex [N]
        B_ch = Bc[cur_channel]             # complex [N]
        C_ch = Cc[cur_channel]             # complex [N]

        # shapes for broadcasting:
        # exp_y[:, None] -> [H,1]
        # exp_x[None, :] -> [1,W]
        # We want denom: [H, W, N] = 1 - exp_y[:,None,None] * lam_y[None,None,:] - exp_x[None,:,None] * lam_x[None,None,:]
        ey = exp_y.view(H, 1, 1)            # [H,1,1]
        ex = exp_x.view(1, W, 1)            # [1,W,1]
        lam_y_view = lam_y_ch.view(1, 1, N)  # [1,1,N]
        lam_x_view = lam_x_ch.view(1, 1, N)  # [1,1,N]

        denom = 1.0 - ey * lam_y_view - ex * lam_x_view  # [H, W, N], complex

        # numerator CB per mode: [1,1,N]
        CB = (C_ch * B_ch).view(1, 1, N)  # complex

        # frequency response per mode and grid: [H, W, N]
        H_f_modes = CB / (denom + eps)  # complex

        # sum over modes -> [H, W] complex
        H_f_ch = H_f_modes.sum(dim=-1)  # [H, W], complex

        # Broadcast to multiply with X_f: X_f[:, ch, :, :] * H_f_ch[None, :, :]
        H_f_ch = H_f_ch.unsqueeze(0)  # [1, H, W]
        H_f_combined[:, cur_channel, :, :] = H_f_ch
    return H_f_combined

@torch.jit.script
def memory_efficient_s4nd_transition_func_chunked(C:int,
                                                  H:int,
                                                  W:int,
                                                  N:int,
                                                  exp_x:torch.Tensor,
                                                  exp_y:torch.Tensor,
                                                  lambda_x:torch.Tensor,
                                                  lambda_y:torch.Tensor,
                                                  Bc:torch.Tensor,
                                                  Cc:torch.Tensor,
                                                  eps:float,
                                                  chunk_size:int,
                                                  device:torch.device):
    # Loop over channels, compute H_f for a single channel at a time (vectorized over N and H×W)
    # This minimizes peak memory: we only allocate [H, W, N] per channel.
    H_f_combined = torch.zeros((1,C,H,W)).to(device)
    for cur_channel in range(0,C,chunk_size):
        chunk_start = cur_channel
        chunk_end = min(C, cur_channel + chunk_size)

        # Fully vectorized across channels (fast but uses bigger memory):
        # lam_x: [C, N], lam_y: [C, N], Bc: [C, N], Cc: [C, N]
        # Build denom: [H, W, C, N] = 1 - exp_y[:,None, None] * lam_y[None,None,:,:] - exp_x[None,:,None] * lam_x[None,None,:,:]
        ey = exp_y.view(H, 1, 1, 1)  # [H,1,1,1]
        ex = exp_x.view(1, W, 1, 1)  # [1,W,1,1]
        lam_y_view = lambda_y.view(1, 1, C, N)[:,:,chunk_start:chunk_end,:]  # [1,1,chunk_size,N]
        lam_x_view = lambda_x.view(1, 1, C, N)[:,:,chunk_start:chunk_end,:]  # [1,1,chunk_size,N]

        denom = 1.0 - ey * lam_y_view - ex * lam_x_view  # [H,W,chunk_size,N]

        CB = (Cc * Bc).view(1, 1, C, N)[:,:,chunk_start:chunk_end,:]  # [1,1,C,N]
        H_f_modes = CB / (denom + eps)   # [H,W,chunk_size,N]
        H_f = H_f_modes.sum(dim=-1)      # [H,W,chunk_size]

        # reorder to [1,C,H,W] and broadcast multiply
        H_f_combined[:,chunk_start:chunk_end,:,:] = H_f.permute(2, 0, 1)  # [1,C,H,W]
    return H_f_combined

@torch.jit.script
def full_s4nd_forward(C:int,
                      H:int,
                      W:int,
                      N:int,
                      exp_x:torch.Tensor,
                      exp_y:torch.Tensor,
                      lambda_x:torch.Tensor,
                      lambda_y:torch.Tensor,
                      Bc:torch.Tensor,
                      Cc:torch.Tensor,
                      eps:float):
    # Fully vectorized across channels (fast but uses bigger memory):
    # lam_x: [C, N], lam_y: [C, N], Bc: [C, N], Cc: [C, N]
    # Build denom: [H, W, C, N] = 1 - exp_y[:,None, None] * lam_y[None,None,:,:] - exp_x[None,:,None] * lam_x[None,None,:,:]
    ey = exp_y.view(H, 1, 1, 1)  # [H,1,1,1]
    ex = exp_x.view(1, W, 1, 1)  # [1,W,1,1]
    lam_y_view = lambda_y.view(1, 1, C, N)  # [1,1,C,N]
    lam_x_view = lambda_x.view(1, 1, C, N)  # [1,1,C,N]

    denom = 1.0 - ey * lam_y_view - ex * lam_x_view  # [H,W,C,N]

    CB = (Cc * Bc).view(1, 1, C, N)  # [1,1,C,N]
    H_f_modes = CB / (denom + eps)   # [H,W,C,N]
    H_f = H_f_modes.sum(dim=-1)           # [H,W,C]

    # reorder to [1,C,H,W] and broadcast multiply
    H_f = H_f.permute(2, 0, 1).unsqueeze(0)  # [1,C,H,W]
    return H_f

class S4ND_Kernel(nn.Module):
    """
    Diagonalized (eigenvalue) S4ND 2D module (memory-efficient version).

    Inputs:
      x: float tensor [B, C, H, W]

    Parameters:
      channels: int (C)
      N: state dim per channel
      use_D: bool (add direct feedthrough D * x)

    Notes:
      - This is a simplified demonstration using diagonalized per-dimension eigenvalues.
      - For stability, real parts of lambdas are typically initialized negative.
    """
    def __init__(
        self,
        channels: int,
        N: int,
        use_D: bool = True,
        lambda_scale: float = -0.1,
        precision="half",
        chunk_size=1,
        band_limit_strategy="none",
        learn_bandlimit=False,
        cut_off_nyquist_proportion=0.25,
        bandlimit_taper_width=0.1,
        learn_frequency_importances=False,
        frequency_importance_hidden_dim=32,
        verbose=False,
    ):
        super().__init__()
        self.C_in = channels
        self.N = N
        self.use_D = use_D

        # Parameterization: real and imag parts for lambda_x / lambda_y per (channel, N)
        self.lambda_x_real = nn.Parameter(lambda_scale * torch.randn(channels, N))
        self.lambda_x_imag = nn.Parameter(0.01 * torch.randn(channels, N))
        self.lambda_y_real = nn.Parameter(lambda_scale * torch.randn(channels, N))
        self.lambda_y_imag = nn.Parameter(0.01 * torch.randn(channels, N))

        # B and C (per channel, per N) reparameterized as real/imag
        self.B_real = nn.Parameter(0.1 * torch.randn(channels, N))
        self.B_imag = nn.Parameter(0.01 * torch.randn(channels, N))
        self.C_real = nn.Parameter(0.1 * torch.randn(channels, N))
        self.C_imag = nn.Parameter(0.01 * torch.randn(channels, N))

        # Optional direct feedthrough per-channel scalar (real)
        if use_D:
            self.D = nn.Parameter(torch.zeros(channels))
        else:
            self.register_parameter("D", None)

        # small epsilon to stabilize divisions
        self.eps = 1e-5
        self.verbose = verbose
        self.band_limit_strategy=band_limit_strategy
        self.cut_off_nyquist_proportion=cut_off_nyquist_proportion
        self.bandlimit_taper_width=bandlimit_taper_width
        self.learn_bandlimit=learn_bandlimit
        if self.learn_bandlimit:
            # The learned cutoff parameter should start at the passed cut_off_nyquist_proportion.
            # To get this behavior, we intialize the parameter with a starting value based on the inverse of the sigmoid function (logit)
            self.bandlimit_learned_cutoff = nn.Parameter(torch.tensor([torch.logit(torch.tensor(self.cut_off_nyquist_proportion))]))
        else:
            self.register_parameter("bandlimit_learned_cutoff", None)

        self.learn_frequency_importances=learn_frequency_importances
        if self.learn_frequency_importances:
            self.frequency_importance_model=nn.Sequential(nn.Linear(in_features=2, out_features=frequency_importance_hidden_dim),
                                                          nn.GELU(),
                                                          nn.Linear(in_features=frequency_importance_hidden_dim, out_features=1),
                                                          nn.Sigmoid())
        else:
            self.register_parameter("frequency_importance_model", None)

        # Set the chunk size that will be used if we are in memory efficient mode
        self.chunk_size = chunk_size

        # Set the data type that will be used in this model
        self.precision = precision.lower().replace(" ", "")
        if self.precision == HALF_PRECISION:
            self._complex_type = torch.complex32
            self._float_type = torch.float16
        elif self.precision == FULL_PRECISION:
            self._complex_type = torch.complex64
            self._float_type = torch.float32
        elif self.precision == DOUBLE_PRECISION:
            self._complex_type = torch.complex128
            self._float_type = torch.float64
        else:
            raise ValueError("Unacceptable value for precision passed. Got value " + str(self.precision) + \
                             " but acceptable values are " + str(PRECISION_VALUES))

    def _build_complex_params(self, device, dtype):
        """
        Build complex tensors for lambda_x, lambda_y, B, C of shapes:
          lambda_x / lambda_y: [C, N] complex
          B, C: [C, N] complex
        """
        lambda_x = (self.lambda_x_real.to(device=device, dtype=dtype) + 1j * self.lambda_x_imag.to(device=device, dtype=dtype)).to(self._complex_type)
        lambda_y = (self.lambda_y_real.to(device=device, dtype=dtype) + 1j * self.lambda_y_imag.to(device=device, dtype=dtype)).to(self._complex_type)
        Bc = (self.B_real.to(device=device, dtype=dtype) + 1j * self.B_imag.to(device=device, dtype=dtype)).to(self._complex_type)
        Cc = (self.C_real.to(device=device, dtype=dtype) + 1j * self.C_imag.to(device=device, dtype=dtype)).to(self._complex_type)
        return lambda_x, lambda_y, Bc, Cc

    def build_bandlimit_mask(self, fx, fy, d_x, d_y):
        # Compute the nyquist values for the x and y dimesions
        nyquist_x = 1.0 / (2.0 * d_x)
        nyquist_y = 1.0 / (2.0 * d_y)

        # Generate the sampling grid and normalize it against the sampling frequency
        fx_grid, fy_grid = torch.meshgrid(fx, fy, indexing='ij')
        fx_grid_norm = fx_grid / nyquist_x
        fy_grid_norm = fy_grid / nyquist_y

        # If we are not going to be doing any bandlimiting, then we generate all ones as a mask so everything is passed through
        if self.band_limit_strategy is None or self.band_limit_strategy.lower() == NO_BANDLIMIT:
            return torch.ones(fx.shape[0], fy.shape[0])

        # Otherwise, we are going to be doing some masking
        # First, we need to compute the proportion of the nyquist function that will be used as the bandlimit.
        # We multiply by 0.5 so that we...
        #   1) Are honest about the name of the "cut_off_nyquist_proportion" variable
        #   2) Don't accidentally bandlimit above the nyquist frequnecy in the case of the learned model or accidentally in the case of manually
        # Because we are dealing with normalized frequencies when constructing this mask, we can deal direclty with proportions, instead
        # of the raw frequencies
        frequency_cutoff = 0.5 * torch.sigmoid(self.bandlimit_learned_cutoff) if self.learn_bandlimit else self.cut_off_nyquist_proportion

        # Error control first, if we are trying to learn a bandlimit in hard mode, there wont be gradients to pass
        # So we raise an error
        if self.band_limit_strategy.lower() == HARD_BANDLIMIT and self.learn_bandlimit:
            raise ValueError("Cannot learn the bandlimiting function in hard bandlimiting mode. Use mode \"" + str(SOFT_BANDLIMIT) + "\" instead.")

        # If we are in hard mode...
        if self.band_limit_strategy.lower() == HARD_BANDLIMIT and not self.learn_bandlimit:
            # Anywhere the frequency dimension is less than the cutoff, set the mask to 1 (we like low frequencies)
            # Anywhere the frequency dimension is greater than the cutoff, set the mask to 0 (we dislike low frequencies)
            # The frequency values have been normalized, so we can use the
            return torch.sqrt(fx_grid_norm**2 + fy_grid_norm**2) < frequency_cutoff

        # If we are in soft mode...
        if self.band_limit_strategy.lower() == SOFT_BANDLIMIT:
            radius = torch.sqrt(fx_grid_norm**2 + fy_grid_norm**2)
            low = frequency_cutoff * (1 - self.bandlimit_taper_width)
            high = frequency_cutoff * (1 + self.bandlimit_taper_width)
            taper = 0.5 * (1 + torch.cos(math.pi * (radius - low) / (high - low)))

            # Anywhere the frequency dimension is less than the low cutoff, set the mask to 1 (we like low frequencies)
            # Anywhere the frequency dimension is greater than the high cutoff, set the mask to 0 (we dislike low frequencies)
            # Everywhere in between the low and high cutoff is the cos arc (starts at 1 and ends at 0), so it is a smooth transition to the hard cutoff
            return torch.where(radius <= low, 1, torch.where(radius >= high, 0, taper))

        # If we dont know the mode, then we raise an error...
        raise ValueError("Unknown value in band_limit_strategy: " + str(self.band_limit_strategy))

    def build_frequnecy_importance_mask(self, fx, fy, w, h):
        if self.learn_frequency_importances:
            fx_grid, fy_grid = torch.meshgrid(fx, fy, indexing='ij')
            radius_f = torch.sqrt(fx_grid**2 + fy_grid**2).flatten()
            radius_d = torch.sqrt(torch.tensor(w**2 + h**2)).to(radius_f.device).repeat(radius_f.shape)
            freq_data_input = torch.stack((radius_f, radius_d), dim=-1)
            return self.frequency_importance_map(freq_data_input).view(fx.shape[0], fy.shape[0])
        return torch.ones(fx.shape[0], fy.shape[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W] real
        returns y: [B, C, H, W] real
        """
        assert x.dim() == 4, "Input must be [B, C, H, W]"
        _, C, H, W = x.shape
        assert C == self.C_in, f"Expected {self.C_in} channels but got {C}"
        if H <= 2 or W <= 2:
            if self.verbose:
                print("Warning: Input tensor too small to perform FFT, returning Identity function")
            return x

        device = x.device
        dtype = x.dtype

        # Sampling rate for the x and y dimensions
        d_x = 1.0/W
        d_y = 1.0/H

        # pylint: disable=not-callable
        # fft of input (complex)
        X_f = torch.fft.fft2(x, dim=(-2, -1))  # complex tensor [B, C, H, W]

        # frequency grids (cycles).
        # Using d=1.0 grid spacing -> frequencies in [-0.5,0.5), equivalent to learning in the pixel dimension
        # Using d=d_x/d_y spacing -> frequencies equivalent to learning in the spatial dimension
        fx = torch.fft.fftfreq(W, d=d_x, device=device, dtype=self._float_type)  # [W]
        fy = torch.fft.fftfreq(H, d=d_y, device=device, dtype=self._float_type)  # [H]
        # pylint: enable=not-callable

        # precompute complex exponentials e^{-i 2π f_x}, e^{-i 2π f_y}
        exp_x = torch.exp(-2j * math.pi * fx).to(self._complex_type)  # [W]
        exp_y = torch.exp(-2j * math.pi * fy).to(self._complex_type)  # [H]

        # Build complex parameters
        lambda_x, lambda_y, Bc, Cc = self._build_complex_params(device, dtype)

        # We'll produce Y_f by multiplying X_f * H_f where H_f is frequency response per-channel
        Y_f = torch.empty_like(X_f)  # complex

        # Choose computation strategy
        if self.chunk_size == -1:
            H_f = full_s4nd_forward(C=int(C),
                                    H=int(H),
                                    W=int(W),
                                    N=int(self.N),
                                    exp_x=exp_x,
                                    exp_y=exp_y,
                                    lambda_x=lambda_x,
                                    lambda_y=lambda_y,
                                    Bc=Bc,
                                    Cc=Cc,
                                    eps=self.eps)
        if self.chunk_size == 1:
            H_f = memory_efficient_s4nd_transition_func(C=int(C),
                                                        H=int(H),
                                                        W=int(W),
                                                        N=int(self.N),
                                                        exp_x=exp_x,
                                                        exp_y=exp_y,
                                                        lambda_x=lambda_x,
                                                        lambda_y=lambda_y,
                                                        Bc=Bc,
                                                        Cc=Cc,
                                                        eps=self.eps,
                                                        device=device)
        elif self.chunk_size > 1:
            H_f = memory_efficient_s4nd_transition_func_chunked(C=int(C),
                                                                H=int(H),
                                                                W=int(W),
                                                                N=int(self.N),
                                                                exp_x=exp_x,
                                                                exp_y=exp_y,
                                                                lambda_x=lambda_x,
                                                                lambda_y=lambda_y,
                                                                Bc=Bc,
                                                                Cc=Cc,
                                                                eps=self.eps,
                                                                chunk_size=self.chunk_size,
                                                                device=device)
        else:
            raise ValueError("chunk_size must be -1, 1, or >1. Got " + str(self.chunk_size))

        # Construct the bandlimit mask based on the frequencies observed in the input
        bandlimit_mask = self.build_bandlimit_mask(fx, fy, d_x, d_y)

        # Construct the bandlimit mask based on the frequencies observed in the input
        frequency_importance_mask = self.build_frequnecy_importance_mask(fx, fy, W, H)

        # Compute the output in the frequency domain based on the input, the H_f tensor, and the frequency importance mask
        Y_f = X_f * H_f * bandlimit_mask.to(device) * frequency_importance_mask.to(device) # [B,C,H,W] complex

        # Inverse FFT to spatial domain (complex -> real)
        # pylint: disable-next=not-callable
        y = torch.fft.ifft2(Y_f, dim=(-2, -1)).real  # [B, C, H, W] real

        # optional direct path D * x
        if self.use_D and self.D is not None:
            # D: [C] real
            y = y + (self.D.view(1, C, 1, 1) * x)

        return y
