"""Napari plugin for ForceInferencePy.

Exposes a multi-tab widget that wires ALL pipeline parameters to Qt controls.

Install napari support:
    pip install -e ".[napari]"

Launch:
    python -m force_inference._napari [--image data/test.tif]
    # OR, once installed as a plugin:
    napari
    # then Plugins → Cell Force Inference
"""
__all__ = ["ForceInferenceWidget", "load_test_tissue"]


def __getattr__(name: str):
    """Lazy import — Qt is only pulled in when the symbol is actually used."""
    if name == "ForceInferenceWidget":
        from ._widget import ForceInferenceWidget
        return ForceInferenceWidget
    if name == "load_test_tissue":
        from ._sample_data import load_test_tissue
        return load_test_tissue
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
