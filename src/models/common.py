"""
Helper functions for building models from config.
"""

from src.models.baseline_unet import UNet
from src.models.ds_unet import DSUNet


def build_model(config):
    """
    Create a model based on the config dict.

    Expects config to have a 'model' key with at least 'name'.
    Example:
        model:
          name: baseline   # or 'dscnn'
          features: [64, 128, 256, 512]
    """
    model_cfg = config.get('model', config)
    name = model_cfg['name']

    kwargs = {}
    if 'in_channels' in model_cfg:
        kwargs['in_channels'] = model_cfg['in_channels']
    if 'out_channels' in model_cfg:
        kwargs['out_channels'] = model_cfg['out_channels']
    if 'features' in model_cfg:
        kwargs['features'] = model_cfg['features']

    if name == 'baseline':
        return UNet(**kwargs)
    elif name == 'dscnn':
        return DSUNet(**kwargs)
    else:
        raise ValueError(f"Unknown model: {name}. Use 'baseline' or 'dscnn'.")


def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
