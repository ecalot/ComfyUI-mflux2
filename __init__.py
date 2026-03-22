from .mflux_nodes import (
    MFluxConceptFromImage,
    MFluxControlNet,
    MFluxDepthPro,
    MFluxGenerateFill,
    MFluxKlein,
)

NODE_CLASS_MAPPINGS = {
    "MFluxKlein": MFluxKlein,
    "MFluxDepthPro": MFluxDepthPro,
    "MFluxGenerateFill": MFluxGenerateFill,
    "MFluxConceptFromImage": MFluxConceptFromImage,
    "MFluxControlNet": MFluxControlNet,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MFluxKlein": "MFlux Klein",
    "MFluxDepthPro": "MFlux Depth Pro",
    "MFluxGenerateFill": "MFlux Generate Fill",
    "MFluxConceptFromImage": "MFlux Concept From Image",
    "MFluxControlNet": "MFlux ControlNet",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
