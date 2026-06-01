"""
Launch the ForceInference Napari widget directly:

    python -m force_inference._napari [--image path/to/image.tif]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="ForceInference Napari viewer")
    ap.add_argument("--image", type=Path, default=None,
                    help="Image file to pre-load")
    args = ap.parse_args()

    import napari
    from force_inference._napari import ForceInferenceWidget

    viewer = napari.Viewer(title="ForceInferencePy")
    widget = ForceInferenceWidget(viewer)
    viewer.window.add_dock_widget(
        widget,
        name="Cell Force Inference",
        area="right",
        allowed_areas=["right", "left"],
    )

    if args.image and args.image.exists():
        from skimage import io
        img = io.imread(str(args.image))
        viewer.add_image(img, name=args.image.name, colormap="gray")
        widget._img_path = args.image
        widget._img_edit.setText(str(args.image))

    napari.run()


if __name__ == "__main__":
    main()
