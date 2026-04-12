# resunet.py

"""ResUnet architecture and checkpoint utilities for Fiberpose."""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- Optional GPU helpers ----------------

torch_GPU = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_CPU = torch.device("cpu")
def empty_cache():
    """Orchestrate empty cache within the current module workflow."""
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

# -------------------- ResUnet (your code; minimal-safe load tweak) --------------------

def dilation_list(x,N):
    """Orchestrate dilation list within the current module workflow.
    
    Args:
        x (Any): Coordinate component representing the x axis value.
        N (Any): Value for n interpreted by this routine.
    
    Returns:
        Any: Object containing the dilation list outcome for downstream use."""
    return np.round(np.linspace(1, x, N)).astype(int).tolist()

def batchconv(in_channels, out_channels, kernel_size, dim, dilation, relu=True):
    """Orchestrate batchconv within the current module workflow.
    
    Args:
        in_channels (Any): Value for in channels interpreted by this routine.
        out_channels (Any): Value for out channels interpreted by this routine.
        kernel_size (Any): Value for kernel size interpreted by this routine.
        dim (Any): Value for dim interpreted by this routine.
        dilation (Any): Value for dilation interpreted by this routine.
        relu (Any): Value for relu interpreted by this routine.
    
    Returns:
        Any: Object containing the batchconv outcome for downstream use."""
    BatchNorm = nn.BatchNorm2d if dim == 2 else nn.BatchNorm3d
    ConvND = nn.Conv2d if dim == 2 else nn.Conv3d
    padding = ((kernel_size - 1) * dilation) // 2
    layers = [BatchNorm(in_channels, eps=1e-5, momentum=0.05)]
    if relu:
        layers.append(nn.ReLU(inplace=True))
    layers.append(ConvND(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation,
                         padding_mode='reflect'))
    return nn.Sequential(*layers)

class resdown(nn.Module):
    """Represent resdown state used across processing and visualization workflows."""
    def __init__(self, in_channels, out_channels, sz, dim, dilation):
        """Initialize resdown and establish required runtime state.
        
        Args:
            in_channels (Any): Value for in channels interpreted by this routine.
            out_channels (Any): Value for out channels interpreted by this routine.
            sz (Any): Value for sz interpreted by this routine.
            dim (Any): Value for dim interpreted by this routine.
            dilation (Any): Value for dilation interpreted by this routine."""
        super().__init__()
        self.conv = nn.Sequential()
        self.proj = batchconv(in_channels, out_channels, 1, dim, 1, relu=False)
        for t in range(4):
            if t==0:
                self.conv.add_module('conv_%d'%t, batchconv(in_channels, out_channels, sz, dim, dilation))
            else:
                self.conv.add_module('conv_%d'%t, batchconv(out_channels, out_channels, sz, dim, dilation))
    def forward(self, x):
        """Orchestrate forward within the current module workflow.
        
        Args:
            x (Any): Coordinate component representing the x axis value.
        
        Returns:
            Any: Object containing the forward outcome for downstream use."""
        x = self.proj(x) + self.conv[1](self.conv[0](x))
        x = x + self.conv[3](self.conv[2](x))
        return x

class convdown(nn.Module):
    """Represent convdown state used across processing and visualization workflows."""
    def __init__(self, in_channels, out_channels, sz, dim, dilation):
        """Initialize convdown and establish required runtime state.
        
        Args:
            in_channels (Any): Value for in channels interpreted by this routine.
            out_channels (Any): Value for out channels interpreted by this routine.
            sz (Any): Value for sz interpreted by this routine.
            dim (Any): Value for dim interpreted by this routine.
            dilation (Any): Value for dilation interpreted by this routine."""
        super().__init__()
        self.conv = nn.Sequential()
        for t in range(2):
            if t==0:
                self.conv.add_module('conv_%d'%t, batchconv(in_channels, out_channels, sz, dim, dilation))
            else:
                self.conv.add_module('conv_%d'%t, batchconv(out_channels, out_channels, sz, dim, dilation))
    def forward(self, x):
        """Orchestrate forward within the current module workflow.
        
        Args:
            x (Any): Coordinate component representing the x axis value.
        
        Returns:
            Any: Object containing the forward outcome for downstream use."""
        x = self.conv[0](x)
        x = self.conv[1](x)
        return x

class batchconvstyle(nn.Module):
    """Represent batchconvstyle state used across processing and visualization workflows."""
    def __init__(self, in_channels, out_channels, style_channels, sz, dim, dilation, concatenation=False):
        """Initialize batchconvstyle and establish required runtime state.
        
        Args:
            in_channels (Any): Value for in channels interpreted by this routine.
            out_channels (Any): Value for out channels interpreted by this routine.
            style_channels (Any): Value for style channels interpreted by this routine.
            sz (Any): Value for sz interpreted by this routine.
            dim (Any): Value for dim interpreted by this routine.
            dilation (Any): Value for dilation interpreted by this routine.
            concatenation (Any): Value for concatenation interpreted by this routine."""
        super().__init__()
        self.conv = batchconv(in_channels, out_channels, sz, dim, dilation)
        if concatenation:
            self.full = nn.Linear(style_channels, out_channels*2)
        else:
            self.full = nn.Linear(style_channels, out_channels)
        self.dim = dim
    def forward(self, style, x, mkldnn=False, y=None):
        """Orchestrate forward within the current module workflow.
        
        Args:
            style (Any): Value for style interpreted by this routine.
            x (Any): Coordinate component representing the x axis value.
            mkldnn (Any): Value for mkldnn interpreted by this routine.
            y (Any): Coordinate component representing the y axis value.
        
        Returns:
            Any: Object containing the forward outcome for downstream use."""
        if y is not None:
            x = x + y
        feat = self.full(style)
        for _ in range(self.dim):
            feat = feat.unsqueeze(-1)
        if mkldnn:
            x = x.to_dense()
            y = (x + feat).to_mkldnn()
        else:
            y = x + feat
        y = self.conv(y)
        return y

class resup(nn.Module):
    """Represent resup state used across processing and visualization workflows."""
    def __init__(self, in_channels, out_channels, style_channels, parent, dilation=None):
        """Initialize resup and establish required runtime state.
        
        Args:
            in_channels (Any): Value for in channels interpreted by this routine.
            out_channels (Any): Value for out channels interpreted by this routine.
            style_channels (Any): Value for style channels interpreted by this routine.
            parent (Any): Value for parent interpreted by this routine.
            dilation (Any): Value for dilation interpreted by this routine."""
        super().__init__()
        sz = parent.sz
        concatenation = parent.concatenation
        dim = parent.dim
        dilation = parent.dilation if dilation is None else dilation
        self.conv = nn.Sequential()
        self.conv.add_module('conv_0', batchconv(in_channels, out_channels, sz, dim, dilation))
        self.conv.add_module('conv_1', batchconvstyle(out_channels, out_channels, style_channels, sz, dim, dilation, concatenation=concatenation))
        self.conv.add_module('conv_2', batchconvstyle(out_channels, out_channels, style_channels, sz, dim, dilation))
        self.conv.add_module('conv_3', batchconvstyle(out_channels, out_channels, style_channels, sz, dim, dilation))
        self.proj = batchconv(in_channels, out_channels, 1, dim, 1, relu=False)
    def forward(self, x, y, style, mkldnn=False):
        """Orchestrate forward within the current module workflow.
        
        Args:
            x (Any): Coordinate component representing the x axis value.
            y (Any): Coordinate component representing the y axis value.
            style (Any): Value for style interpreted by this routine.
            mkldnn (Any): Value for mkldnn interpreted by this routine.
        
        Returns:
            Any: Object containing the forward outcome for downstream use."""
        x = self.proj(x) + self.conv[1](style, self.conv[0](x) + y, mkldnn=mkldnn)
        x = x + self.conv[3](style, self.conv[2](style, x, mkldnn=mkldnn), mkldnn=mkldnn)
        return x

class convup(nn.Module):
    """Represent convup state used across processing and visualization workflows."""
    def __init__(self, in_channels, out_channels, style_channels, parent, dilation=None):
        """Initialize convup and establish required runtime state.
        
        Args:
            in_channels (Any): Value for in channels interpreted by this routine.
            out_channels (Any): Value for out channels interpreted by this routine.
            style_channels (Any): Value for style channels interpreted by this routine.
            parent (Any): Value for parent interpreted by this routine.
            dilation (Any): Value for dilation interpreted by this routine."""
        super().__init__()
        sz = parent.sz
        concatenation = parent.concatenation
        dim = parent.dim
        dilation = parent.dilation if dilation is None else dilation
        self.conv = nn.Sequential()
        self.conv.add_module('conv_0', batchconv(in_channels, out_channels, sz, dim, dilation))
        self.conv.add_module('conv_1', batchconvstyle(out_channels, out_channels, style_channels, sz, dim, dilation, concatenation=concatenation))
    def forward(self, x, y, style):
        """Orchestrate forward within the current module workflow.
        
        Args:
            x (Any): Coordinate component representing the x axis value.
            y (Any): Coordinate component representing the y axis value.
            style (Any): Value for style interpreted by this routine.
        
        Returns:
            Any: Object containing the forward outcome for downstream use."""
        x = self.conv[1](style, self.conv[0](x) + y)
        return x

class downsample(nn.Module):
    """Represent downsample state used across processing and visualization workflows."""
    def __init__(self, parent):
        """Initialize downsample and establish required runtime state.
        
        Args:
            parent (Any): Value for parent interpreted by this routine."""
        super().__init__()
        nbase = parent.nbase
        sz = parent.sz
        residual_on = parent.residual_on
        dim = parent.dim
        dilation = parent.dilation
        kernel_size = parent.kernel_size
        scale_factor = parent.scale_factor
        self.checkpoint = parent.checkpoint
        self.down = nn.Sequential()
        maxpool = nn.MaxPool2d if dim == 2 else nn.MaxPool3d
        self.maxpool = maxpool(kernel_size=kernel_size, stride=scale_factor)
        N = len(nbase)-1
        dilations = dilation_list(dilation, N)
        for n, dil in enumerate(dilations):
            if residual_on:
                self.down.add_module('res_down_%d'%n, resdown(nbase[n], nbase[n+1], sz, dim, dil))
            else:
                self.down.add_module('conv_down_%d'%n, convdown(nbase[n], nbase[n+1], sz, dim, dil))
    def forward(self, x):
        """Orchestrate forward within the current module workflow.
        
        Args:
            x (Any): Coordinate component representing the x axis value.
        
        Returns:
            Any: Object containing the forward outcome for downstream use."""
        xd = []
        for n in range(len(self.down)):
            y = self.maxpool(xd[n-1]) if n>0 else x
            xd.append(self.down[n](y))
        return xd

class upsample(nn.Module):
    """Represent upsample state used across processing and visualization workflows."""
    def __init__(self, parent):
        """Initialize upsample and establish required runtime state.
        
        Args:
            parent (Any): Value for parent interpreted by this routine."""
        super().__init__()
        nbase = parent.nbaseup
        self.upsampling = nn.Upsample(scale_factor=parent.scale_factor, mode='nearest')
        self.up = nn.Sequential()
        self.checkpoint = parent.checkpoint
        N = len(nbase)-1
        dilations = dilation_list(parent.dilation, N)
        for k, dil in enumerate(dilations):
            n = k+1
            if parent.residual_on:
                self.up.add_module('res_up_%d'%(n-1), resup(nbase[n], nbase[n-1], nbase[-1], parent, dilation=dil))
            else:
                self.up.add_module('conv_up_%d'%(n-1), convup(nbase[n], nbase[n-1], nbase[-1], parent, dilation=dil))
    def forward(self, style, xd, mkldnn=False):
        """Orchestrate forward within the current module workflow.
        
        Args:
            style (Any): Value for style interpreted by this routine.
            xd (Any): Value for xd interpreted by this routine.
            mkldnn (Any): Value for mkldnn interpreted by this routine.
        
        Returns:
            Any: Object containing the forward outcome for downstream use."""
        x = xd[-1]
        for n in range(len(self.up)):
            idx = -(n + 1)
            if n > 0:
                x = self.upsampling(x.to_dense()).to_mkldnn() if mkldnn else self.upsampling(x)
            x = self.up[idx](x, xd[idx], style, mkldnn=mkldnn) if "res_up" in self.up[idx]._get_name() else self.up[idx](x, xd[idx], style)
        return x

class make_style(nn.Module):
    """Represent make style state used across processing and visualization workflows."""
    def __init__(self, parent):
        """Initialize make style and establish required runtime state.
        
        Args:
            parent (Any): Value for parent interpreted by this routine."""
        super().__init__()
        self.dim = parent.dim
        self.flatten = nn.Flatten()
        self.avg_pool = F.avg_pool2d if self.dim==2 else F.avg_pool3d
    def forward(self, x0):
        """Orchestrate forward within the current module workflow.
        
        Args:
            x0 (Any): Value for x0 interpreted by this routine.
        
        Returns:
            Any: Object containing the forward outcome for downstream use."""
        style = self.avg_pool(x0, kernel_size=tuple(x0.shape[-self.dim:]))
        style = self.flatten(style)
        style = style / torch.sum(style**2, axis=1, keepdim=True)**.5
        return style

def _safe_torch_load(path, map_location, weights_only=True):
    """Provide an internal helper for safe torch load.
    
    Args:
        path (Any): Filesystem location describing where path is read or written.
        map_location (Any): Value for map location interpreted by this routine.
        weights_only (Any): Numeric control parameter for weights only during computation.
    
    Returns:
        Any: Object containing the safe torch load outcome for downstream use."""
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=map_location)

class ResUnet(nn.Module):
    """Represent res unet state used across processing and visualization workflows."""
    def __init__(self, nbase, nout, sz, residual_on=True,
                 style_on=True, concatenation=False, mkldnn=False, dim=2,
                 checkpoint=False, dropout=False, kernel_size=2, scale_factor=2, dilation=1):
        """Initialize res unet and establish required runtime state.
        
        Args:
            nbase (Any): Value for nbase interpreted by this routine.
            nout (Any): Value for nout interpreted by this routine.
            sz (Any): Value for sz interpreted by this routine.
            residual_on (Any): Value for residual on interpreted by this routine.
            style_on (Any): Value for style on interpreted by this routine.
            concatenation (Any): Value for concatenation interpreted by this routine.
            mkldnn (Any): Value for mkldnn interpreted by this routine.
            dim (Any): Value for dim interpreted by this routine.
            checkpoint (Any): Value for checkpoint interpreted by this routine.
            dropout (Any): Value for dropout interpreted by this routine.
            kernel_size (Any): Value for kernel size interpreted by this routine.
            scale_factor (Any): Numeric control parameter for scale factor during computation.
            dilation (Any): Value for dilation interpreted by this routine."""
        super().__init__()
        self.checkpoint = checkpoint
        self.kernel_size = kernel_size
        self.scale_factor = scale_factor
        self.dilation = dilation
        self.nbase = nbase
        self.nout = nout
        self.sz = sz
        self.dim = dim
        self.residual_on = residual_on
        self.style_on = style_on
        self.concatenation = concatenation
        self.mkldnn = mkldnn if mkldnn is not None else False
        self.downsample = downsample(self)
        nbaseup = nbase[1:]
        nbaseup.append(nbaseup[-1])
        self.nbaseup = nbaseup
        self.upsample = upsample(self)
        self.make_style = make_style(self)
        self.output = batchconv(nbaseup[0], nout, 1, self.dim, 1)
        self.style_on = style_on
        self.do_dropout = dropout
        if self.do_dropout:
            self.dropout = nn.Dropout2d(0.1)
    def forward(self, data):
        if self.mkldnn:
            data = data.to_mkldnn()

        # Encoder
        T0 = self.downsample(data)

        # Style
        style = self.make_style(T0[-1].to_dense()) if self.mkldnn else self.make_style(T0[-1])
        style0 = style
        if not self.style_on:
            style = style * 0

        # Best-place dropout for DT training: bottleneck only
        if self.do_dropout:
            if self.mkldnn:
                T0[-1] = self.dropout(T0[-1].to_dense()).to_mkldnn()
            else:
                T0[-1] = self.dropout(T0[-1])

        # Decoder
        T0 = self.upsample(style, T0, self.mkldnn)

        # Output head
        T0 = self.output(T0)

        if self.mkldnn:
            T0 = T0.to_dense()

        return T0, style0
    
    def save_model(self, filename):
        """Persist model using the configured export conventions.
        
        Args:
            filename (Any): Filesystem location used by this operation."""
        torch.save(self.state_dict(), filename)

    def load_model(self, filename, map_location=None, strict=True):
        if map_location is None:
            map_location = "cpu"
        state_dict = _safe_torch_load(filename, map_location=map_location, weights_only=True)
        self.load_state_dict(state_dict, strict=strict)
        return self



# ---------------------------- Predict ----------------------------

def parse_nbase(s: str):
    """Parse nbase and enforce semantic validation constraints.
    
    Args:
        s (str): Value for s interpreted by this routine.
    
    Returns:
        Any: Object containing the parse nbase outcome for downstream use.
    
    Raises:
        ValueError: If provided values violate validation or domain constraints."""
    vals = [int(x.strip()) for x in s.split(",") if x.strip()]
    if len(vals) < 2:
        raise ValueError("nbase must have ≥2 entries.")
    return vals
