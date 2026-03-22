import shlex
import random
import tempfile
import shutil
import inspect
import torch
import numpy as np
from PIL import Image

from mflux.models.common.config import ModelConfig
from mflux.models.depth_pro.model.depth_pro import DepthPro
from mflux.models.flux.variants.concept_attention.flux_concept_from_image import (
    Flux1ConceptFromImage,
)
from mflux.models.flux.variants.controlnet.flux_controlnet import Flux1Controlnet
from mflux.models.flux.variants.fill.flux_fill import Flux1Fill
from mflux.models.flux2.variants import Flux2Klein, Flux2KleinEdit


MODEL_OPTIONS = [
    "flux2-klein-4b",
    "flux2-klein-9b",
    "flux2-klein-base-4b",
    "flux2-klein-base-9b",
]

FILL_MODEL_OPTIONS = ["flux1-fill-dev", *MODEL_OPTIONS]

QUANTIZE_OPTIONS = ["none", "3", "4", "5", "6", "8"]
LORA_STYLE_OPTIONS = [
    "none",
    "couple",
    "font",
    "home",
    "identity",
    "illustration",
    "portrait",
    "ppt",
    "sandstorm",
    "sparklers",
    "storyboard",
]

MODEL_CONFIG_FACTORY = {
    "flux2-klein-4b": "flux2_klein_4b",
    "flux2-klein-9b": "flux2_klein_9b",
    "flux2-klein-base-4b": "flux2_klein_base_4b",
    "flux2-klein-base-9b": "flux2_klein_base_9b",
}

FLUX1_MODEL_CONFIG_FACTORY = {
    "schnell": "schnell",
    "dev": "dev",
    "krea-dev": "krea_dev",
    "dev-krea": "krea_dev",
}

CONCEPT_MODEL_OPTIONS = ["schnell", "dev", "krea-dev", "dev-krea"]
CONTROLNET_MODEL_OPTIONS = ["flux1-dev", "flux1-schnell"]

_DEPTH_MODEL_CACHE = {}


def _parse_list_arg(value, cast=None):
    if not value or not str(value).strip():
        return []

    try:
        tokens = shlex.split(value)
    except ValueError:
        tokens = str(value).replace(",", " ").split()

    if cast is None:
        return tokens

    parsed = []
    for token in tokens:
        try:
            parsed.append(cast(token))
        except ValueError as error:
            raise ValueError(
                f"Invalid value '{token}' in list input: {value}"
            ) from error
    return parsed


def _invoke_with_supported_kwargs(func, kwargs):
    signature = inspect.signature(func)
    accepted = {
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    filtered_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in accepted and value is not None
    }
    return func(**filtered_kwargs)


def _read_prompt(prompt, prompt_file):
    if prompt_file and str(prompt_file).strip():
        with open(prompt_file, "r", encoding="utf-8") as file:
            return file.read().strip()
    return prompt


def _resolve_seeds(seed, seeds, auto_seeds):
    parsed = _parse_list_arg(seeds, int)
    if parsed:
        return parsed
    if auto_seeds and auto_seeds > 0:
        return [
            random.SystemRandom().randint(0, 0xFFFFFFFFFFFFFFFF)
            for _ in range(auto_seeds)
        ]
    if seed != -1:
        return [seed]
    return [random.SystemRandom().randint(0, 0xFFFFFFFFFFFFFFFF)]


def _to_pil(tensor_image):
    image_array = (255.0 * tensor_image.cpu().numpy()).clip(0, 255).astype(np.uint8)
    return Image.fromarray(image_array)


def _to_tensor(image):
    return torch.from_numpy(
        np.array(image.convert("RGB")).astype(np.float32) / 255.0
    ).unsqueeze(0)


def _mask_to_pil(mask_tensor):
    mask_array = mask_tensor.cpu().numpy()
    if mask_array.ndim == 3 and mask_array.shape[-1] == 1:
        mask_array = mask_array[:, :, 0]
    if mask_array.ndim != 2:
        raise ValueError(f"Unsupported MASK tensor shape: {tuple(mask_tensor.shape)}")

    mask_uint8 = (255.0 * np.clip(mask_array, 0.0, 1.0)).astype(np.uint8)
    return Image.fromarray(mask_uint8, mode="L")


def _collect_batch_image_paths(images):
    temp_dir = tempfile.mkdtemp(prefix="comfy_mflux_inputs_")
    paths = []
    batch_size = int(images.shape[0])
    for index in range(batch_size):
        image = _to_pil(images[index])
        path = f"{temp_dir}/input_{index:04d}.png"
        image.save(path)
        paths.append(path)
    return temp_dir, paths


def _collect_batch_image_and_mask_paths(images, masks):
    if masks is None:
        raise ValueError("MFlux fill requires a MASK input")

    if not hasattr(images, "shape") or int(images.shape[0]) <= 0:
        raise ValueError("MFlux fill requires at least one IMAGE")

    if masks.ndim == 2:
        masks = masks.unsqueeze(0)
    elif masks.ndim == 4 and masks.shape[-1] == 1:
        masks = masks[:, :, :, 0]

    if masks.ndim != 3:
        raise ValueError(
            f"MFlux fill expects MASK shape [B,H,W] (or [H,W]); got {tuple(masks.shape)}"
        )

    image_batch = int(images.shape[0])
    mask_batch = int(masks.shape[0])
    if mask_batch not in (1, image_batch):
        raise ValueError(
            "MASK batch must be 1 or match IMAGE batch size "
            f"(got mask batch {mask_batch}, image batch {image_batch})"
        )

    temp_dir = tempfile.mkdtemp(prefix="comfy_mflux_fill_inputs_")
    image_paths = []
    mask_paths = []
    image_pils = []
    mask_pils = []

    for index in range(image_batch):
        image = _to_pil(images[index]).convert("RGB")
        mask_index = 0 if mask_batch == 1 else index
        mask = _mask_to_pil(masks[mask_index])
        if mask.size != image.size:
            resample_nearest = (
                Image.Resampling.NEAREST
                if hasattr(Image, "Resampling")
                else Image.NEAREST
            )
            mask = mask.resize(image.size, resample_nearest)

        image_path = f"{temp_dir}/image_{index:04d}.png"
        mask_path = f"{temp_dir}/mask_{index:04d}.png"
        image.save(image_path)
        mask.save(mask_path)

        image_paths.append(image_path)
        mask_paths.append(mask_path)
        image_pils.append(image)
        mask_pils.append(mask)

    return temp_dir, image_paths, mask_paths, image_pils, mask_pils


def _get_depth_model(quantize):
    if quantize not in _DEPTH_MODEL_CACHE:
        _DEPTH_MODEL_CACHE[quantize] = DepthPro(quantize=quantize)
    return _DEPTH_MODEL_CACHE[quantize]


def _extract_pil_images(generated):
    if isinstance(generated, Image.Image):
        return [generated]

    if hasattr(generated, "image") and isinstance(
        getattr(generated, "image"), Image.Image
    ):
        return [generated.image]

    if isinstance(generated, list):
        images = []
        for item in generated:
            if isinstance(item, Image.Image):
                images.append(item)
            elif hasattr(item, "image") and isinstance(
                getattr(item, "image"), Image.Image
            ):
                images.append(item.image)
            else:
                raise ValueError(
                    f"Unsupported list item type from mflux generate_image: {type(item).__name__}"
                )
        return images

    raise ValueError(
        f"Unsupported return type from mflux generate_image: {type(generated).__name__}"
    )


def _get_controlnet_model_config(model):
    normalized = str(model).strip().lower()
    if normalized in ("flux1-schnell", "schnell"):
        return ModelConfig.schnell_controlnet_canny()
    if normalized in ("flux1-dev", "dev"):
        return ModelConfig.dev_controlnet_canny()
    raise ValueError(
        f"Unsupported ControlNet model '{model}'. Use flux1-dev or flux1-schnell."
    )


class MFluxKlein:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (MODEL_OPTIONS, {"default": "flux2-klein-9b"}),
                "mode": (("auto", "generate", "edit"), {"default": "auto"}),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": (
                            "A point-and-click adventure game background of a rugged harbor in Tierra del Fuego, "
                            "classic Monkey Island style, highly detailed pixel art"
                        ),
                    },
                ),
                "width": (
                    "INT",
                    {"default": 1024, "min": 64, "max": 65535, "step": 16},
                ),
                "height": (
                    "INT",
                    {"default": 768, "min": 64, "max": 65535, "step": 16},
                ),
                "steps": ("INT", {"default": 4, "min": 2, "max": 50}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {
                "images": ("IMAGE",),
                "battery_percentage_stop_limit": (
                    "INT",
                    {"default": -1, "min": -1, "max": 100},
                ),
                "low_ram": ("BOOLEAN", {"default": False}),
                "mlx_cache_limit_gb": (
                    "FLOAT",
                    {"default": -1.0, "min": -1.0, "max": 9999.0, "step": 0.1},
                ),
                "base_model": ("STRING", {"default": ""}),
                "quantize": (QUANTIZE_OPTIONS, {"default": "none"}),
                "lora_style": (LORA_STYLE_OPTIONS, {"default": "none"}),
                "lora_paths": ("STRING", {"multiline": True, "default": ""}),
                "lora_scales": ("STRING", {"default": ""}),
                "prompt_file": ("STRING", {"default": ""}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "seeds": ("STRING", {"default": ""}),
                "auto_seeds": ("INT", {"default": -1, "min": -1, "max": 1024}),
                "scheduler": ("STRING", {"default": ""}),
                "guidance": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1},
                ),
                "config_from_metadata": ("STRING", {"default": ""}),
                "metadata": ("BOOLEAN", {"default": False}),
                "output": ("STRING", {"default": ""}),
                "stepwise_image_output_dir": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "MFlux CLI"

    def run(
        self, model, mode, prompt, width, height, steps, seed, images=None, **kwargs
    ):
        temp_dir = None
        parsed_prompt = _read_prompt(prompt, kwargs.get("prompt_file", ""))
        seed_values = _resolve_seeds(
            seed, kwargs.get("seeds", ""), kwargs.get("auto_seeds", -1)
        )

        quantize = kwargs.get("quantize", "none")
        quantize_value = None if quantize == "none" else int(quantize)
        lora_style = kwargs.get("lora_style", "none")
        lora_style_value = None if lora_style == "none" else lora_style
        base_model_value = kwargs.get("base_model", "") or None
        lora_paths = _parse_list_arg(kwargs.get("lora_paths", ""))
        lora_scales = _parse_list_arg(kwargs.get("lora_scales", ""), float)

        has_images = (
            images is not None and hasattr(images, "shape") and int(images.shape[0]) > 0
        )
        use_edit = mode == "edit" or (mode == "auto" and has_images)
        if mode == "edit" and not has_images:
            raise ValueError("Mode 'edit' requires an IMAGE input (batch supported)")

        config_factory_name = MODEL_CONFIG_FACTORY.get(model)
        if not config_factory_name:
            raise ValueError(f"Unsupported model: {model}")

        config_factory = getattr(ModelConfig, config_factory_name)
        model_config = _invoke_with_supported_kwargs(
            config_factory,
            {
                "quantize": quantize_value,
                "low_ram": kwargs.get("low_ram", False),
                "mlx_cache_limit_gb": kwargs.get("mlx_cache_limit_gb", -1.0),
                "battery_percentage_stop_limit": kwargs.get(
                    "battery_percentage_stop_limit", -1
                ),
            },
        )

        model_class = Flux2KleinEdit if use_edit else Flux2Klein
        model_instance = _invoke_with_supported_kwargs(
            model_class,
            {
                "model_config": model_config,
                "base_model": base_model_value,
                "quantize": quantize_value,
                "low_ram": kwargs.get("low_ram", False),
                "mlx_cache_limit_gb": kwargs.get("mlx_cache_limit_gb", -1.0),
                "battery_percentage_stop_limit": kwargs.get(
                    "battery_percentage_stop_limit", -1
                ),
                "lora_style": lora_style_value,
                "lora_paths": lora_paths,
                "lora_scales": lora_scales,
            },
        )

        if use_edit:
            temp_dir, image_paths = _collect_batch_image_paths(images)
        else:
            image_paths = None

        try:
            output_images = []
            for current_seed in seed_values:
                generated = _invoke_with_supported_kwargs(
                    model_instance.generate_image,
                    {
                        "seed": int(current_seed),
                        "prompt": parsed_prompt,
                        "negative_prompt": kwargs.get("negative_prompt", "") or None,
                        "scheduler": kwargs.get("scheduler", "") or None,
                        "num_inference_steps": steps,
                        "height": height,
                        "width": width,
                        "guidance": kwargs.get("guidance", 1.0),
                        "config_from_metadata": kwargs.get("config_from_metadata", "")
                        or None,
                        "metadata": kwargs.get("metadata", False),
                        "output": kwargs.get("output", "") or None,
                        "stepwise_image_output_dir": kwargs.get(
                            "stepwise_image_output_dir", ""
                        )
                        or None,
                        "image_paths": image_paths,
                        "lora_style": lora_style_value,
                        "lora_paths": lora_paths,
                        "lora_scales": lora_scales,
                    },
                )

                output_images.extend(_extract_pil_images(generated))

            tensors = [_to_tensor(image) for image in output_images]
            if not tensors:
                raise ValueError("No images were generated")
            return (torch.cat(tensors, dim=0),)
        finally:
            if temp_dir and tempfile.gettempdir() in temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)


class MFluxDepthPro:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "quantize": (QUANTIZE_OPTIONS, {"default": "none"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "create_depth"
    CATEGORY = "MFlux API"

    def create_depth(self, images, quantize="none"):
        if images is None or not hasattr(images, "shape") or int(images.shape[0]) <= 0:
            raise ValueError("MFluxDepthPro requires at least one IMAGE")

        quantize_value = None if quantize == "none" else int(quantize)
        depth_model = _get_depth_model(quantize_value)

        temp_dir, image_paths = _collect_batch_image_paths(images)
        try:
            depth_tensors = []
            for image_path in image_paths:
                result = depth_model.create_depth_map(image_path=image_path)
                depth_tensors.append(_to_tensor(result.depth_image))

            return (torch.cat(depth_tensors, dim=0),)
        finally:
            if temp_dir and tempfile.gettempdir() in temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)


class MFluxGenerateFill:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (FILL_MODEL_OPTIONS, {"default": "flux1-fill-dev"}),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Fill the masked area naturally while matching composition and lighting.",
                    },
                ),
                "images": ("IMAGE",),
                "masks": ("MASK",),
                "steps": ("INT", {"default": 20, "min": 2, "max": 50}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFFFFFFFFFF}),
                "width": (
                    "INT",
                    {"default": 1024, "min": 64, "max": 65535, "step": 16},
                ),
                "height": (
                    "INT",
                    {"default": 1024, "min": 64, "max": 65535, "step": 16},
                ),
            },
            "optional": {
                "quantize": (QUANTIZE_OPTIONS, {"default": "none"}),
                "model_path": ("STRING", {"default": ""}),
                "low_ram": ("BOOLEAN", {"default": False}),
                "mlx_cache_limit_gb": (
                    "FLOAT",
                    {"default": -1.0, "min": -1.0, "max": 9999.0, "step": 0.1},
                ),
                "battery_percentage_stop_limit": (
                    "INT",
                    {"default": -1, "min": -1, "max": 100},
                ),
                "guidance": (
                    "FLOAT",
                    {"default": 30.0, "min": 0.0, "max": 50.0, "step": 0.1},
                ),
                "image_strength": (
                    "FLOAT",
                    {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.01},
                ),
                "scheduler": ("STRING", {"default": ""}),
                "lora_paths": ("STRING", {"multiline": True, "default": ""}),
                "lora_scales": ("STRING", {"default": ""}),
                "seeds": ("STRING", {"default": ""}),
                "auto_seeds": ("INT", {"default": -1, "min": -1, "max": 1024}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "MFlux API"

    def run(
        self,
        model,
        prompt,
        images,
        masks,
        steps,
        seed,
        width,
        height,
        **kwargs,
    ):
        temp_dir = None
        quantize = kwargs.get("quantize", "none")
        quantize_value = None if quantize == "none" else int(quantize)
        model_path = kwargs.get("model_path", "") or None
        lora_paths = _parse_list_arg(kwargs.get("lora_paths", ""))
        lora_scales = _parse_list_arg(kwargs.get("lora_scales", ""), float)
        image_strength = kwargs.get("image_strength", -1.0)
        image_strength_value = (
            None if image_strength is None or image_strength < 0 else image_strength
        )
        scheduler = kwargs.get("scheduler", "") or None
        seed_values = _resolve_seeds(
            seed, kwargs.get("seeds", ""), kwargs.get("auto_seeds", -1)
        )

        temp_dir, image_paths, mask_paths, original_images, mask_images = (
            _collect_batch_image_and_mask_paths(images, masks)
        )

        try:
            output_images = []

            if model == "flux1-fill-dev":
                fill_model = _invoke_with_supported_kwargs(
                    Flux1Fill,
                    {
                        "quantize": quantize_value,
                        "model_path": model_path,
                        "lora_paths": lora_paths,
                        "lora_scales": lora_scales,
                    },
                )

                for current_seed in seed_values:
                    for image_path, mask_path in zip(image_paths, mask_paths):
                        generated = _invoke_with_supported_kwargs(
                            fill_model.generate_image,
                            {
                                "seed": int(current_seed),
                                "prompt": prompt,
                                "image_path": image_path,
                                "masked_image_path": mask_path,
                                "num_inference_steps": steps,
                                "height": height,
                                "width": width,
                                "guidance": kwargs.get("guidance", 30.0),
                                "image_strength": image_strength_value,
                                "scheduler": scheduler,
                            },
                        )
                        output_images.extend(_extract_pil_images(generated))
            else:
                config_factory_name = MODEL_CONFIG_FACTORY.get(model)
                if not config_factory_name:
                    raise ValueError(f"Unsupported model: {model}")

                model_config = getattr(ModelConfig, config_factory_name)()

                edit_model = _invoke_with_supported_kwargs(
                    Flux2KleinEdit,
                    {
                        "quantize": quantize_value,
                        "model_path": model_path,
                        "lora_paths": lora_paths,
                        "lora_scales": lora_scales,
                        "model_config": model_config,
                        "low_ram": kwargs.get("low_ram", False),
                        "mlx_cache_limit_gb": kwargs.get("mlx_cache_limit_gb", -1.0),
                        "battery_percentage_stop_limit": kwargs.get(
                            "battery_percentage_stop_limit", -1
                        ),
                    },
                )

                resample_lanczos = (
                    Image.Resampling.LANCZOS
                    if hasattr(Image, "Resampling")
                    else Image.LANCZOS
                )

                for current_seed in seed_values:
                    generated = _invoke_with_supported_kwargs(
                        edit_model.generate_image,
                        {
                            "seed": int(current_seed),
                            "prompt": prompt,
                            "num_inference_steps": steps,
                            "height": height,
                            "width": width,
                            "guidance": kwargs.get("guidance", 30.0),
                            "image_paths": image_paths,
                            "image_strength": image_strength_value,
                            "scheduler": scheduler,
                        },
                    )

                    generated_images = _extract_pil_images(generated)
                    if len(generated_images) == 1 and len(original_images) > 1:
                        generated_images = generated_images * len(original_images)

                    if len(generated_images) != len(original_images):
                        raise ValueError(
                            "Flux.2 edit returned an unexpected number of images: "
                            f"{len(generated_images)} (expected {len(original_images)})"
                        )

                    for generated_image, original_image, mask_image in zip(
                        generated_images, original_images, mask_images
                    ):
                        generated_rgb = generated_image.convert("RGB")
                        if generated_rgb.size != original_image.size:
                            generated_rgb = generated_rgb.resize(
                                original_image.size, resample_lanczos
                            )
                        composited = Image.composite(
                            generated_rgb,
                            original_image,
                            mask_image,
                        )
                        output_images.append(composited)

            tensors = [_to_tensor(image) for image in output_images]
            if not tensors:
                raise ValueError("No images were generated")
            return (torch.cat(tensors, dim=0),)
        finally:
            if temp_dir and tempfile.gettempdir() in temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)


class MFluxConceptFromImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (CONCEPT_MODEL_OPTIONS, {"default": "dev"}),
                "images": ("IMAGE",),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Two puffins are perched on a grassy cliffside near the ocean.",
                    },
                ),
                "concept": ("STRING", {"default": "bird"}),
                "steps": ("INT", {"default": 4, "min": 2, "max": 50}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFFFFFFFFFF}),
                "width": (
                    "INT",
                    {"default": 1024, "min": 64, "max": 65535, "step": 16},
                ),
                "height": (
                    "INT",
                    {"default": 1024, "min": 64, "max": 65535, "step": 16},
                ),
            },
            "optional": {
                "quantize": (QUANTIZE_OPTIONS, {"default": "none"}),
                "model_path": ("STRING", {"default": ""}),
                "guidance": (
                    "FLOAT",
                    {"default": 4.0, "min": 0.0, "max": 50.0, "step": 0.1},
                ),
                "image_strength": (
                    "FLOAT",
                    {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.01},
                ),
                "scheduler": ("STRING", {"default": ""}),
                "lora_paths": ("STRING", {"multiline": True, "default": ""}),
                "lora_scales": ("STRING", {"default": ""}),
                "heatmap_layer_indices": ("STRING", {"default": "15 16 17 18"}),
                "heatmap_timesteps": ("STRING", {"default": ""}),
                "seeds": ("STRING", {"default": ""}),
                "auto_seeds": ("INT", {"default": -1, "min": -1, "max": 1024}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "heatmap")
    FUNCTION = "run"
    CATEGORY = "MFlux API"

    def run(
        self,
        model,
        images,
        prompt,
        concept,
        steps,
        seed,
        width,
        height,
        **kwargs,
    ):
        if images is None or not hasattr(images, "shape") or int(images.shape[0]) <= 0:
            raise ValueError("MFluxConceptFromImage requires at least one IMAGE")

        quantize = kwargs.get("quantize", "none")
        quantize_value = None if quantize == "none" else int(quantize)
        model_path = kwargs.get("model_path", "") or None
        lora_paths = _parse_list_arg(kwargs.get("lora_paths", ""))
        lora_scales = _parse_list_arg(kwargs.get("lora_scales", ""), float)
        image_strength = kwargs.get("image_strength", -1.0)
        image_strength_value = (
            None if image_strength is None or image_strength < 0 else image_strength
        )
        scheduler = kwargs.get("scheduler", "") or None
        seed_values = _resolve_seeds(
            seed, kwargs.get("seeds", ""), kwargs.get("auto_seeds", -1)
        )
        heatmap_layer_indices = _parse_list_arg(
            kwargs.get("heatmap_layer_indices", ""), int
        )
        heatmap_timesteps = _parse_list_arg(kwargs.get("heatmap_timesteps", ""), int)

        config_name = FLUX1_MODEL_CONFIG_FACTORY.get(model)
        if not config_name:
            raise ValueError(f"Unsupported concept-from-image model: {model}")
        model_config = getattr(ModelConfig, config_name)()

        concept_model = _invoke_with_supported_kwargs(
            Flux1ConceptFromImage,
            {
                "quantize": quantize_value,
                "model_path": model_path,
                "lora_paths": lora_paths,
                "lora_scales": lora_scales,
                "model_config": model_config,
            },
        )

        temp_dir, image_paths = _collect_batch_image_paths(images)
        try:
            output_images = []
            output_heatmaps = []

            for current_seed in seed_values:
                for image_path in image_paths:
                    generated = _invoke_with_supported_kwargs(
                        concept_model.generate_image,
                        {
                            "seed": int(current_seed),
                            "prompt": prompt,
                            "concept": concept,
                            "image_path": image_path,
                            "num_inference_steps": steps,
                            "height": height,
                            "width": width,
                            "guidance": kwargs.get("guidance", 4.0),
                            "image_strength": image_strength_value,
                            "scheduler": scheduler,
                            "heatmap_layer_indices": heatmap_layer_indices or None,
                            "heatmap_timesteps": heatmap_timesteps or None,
                        },
                    )

                    output_images.extend(_extract_pil_images(generated))

                    concept_heatmap = getattr(generated, "concept_heatmap", None)
                    heatmap_image = (
                        concept_heatmap.image
                        if concept_heatmap is not None
                        and hasattr(concept_heatmap, "image")
                        and isinstance(concept_heatmap.image, Image.Image)
                        else None
                    )
                    if heatmap_image is None:
                        generated_images = _extract_pil_images(generated)
                        heatmap_image = generated_images[0].copy()
                    output_heatmaps.append(heatmap_image)

            image_tensors = [_to_tensor(image) for image in output_images]
            heatmap_tensors = [_to_tensor(image) for image in output_heatmaps]
            if not image_tensors:
                raise ValueError("No images were generated")
            if not heatmap_tensors:
                raise ValueError("No concept heatmaps were generated")
            return (torch.cat(image_tensors, dim=0), torch.cat(heatmap_tensors, dim=0))
        finally:
            if temp_dir and tempfile.gettempdir() in temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)


class MFluxControlNet:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (CONTROLNET_MODEL_OPTIONS, {"default": "flux1-dev"}),
                "controlnet_images": ("IMAGE",),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "A comic strip with a joker in a purple suit.",
                    },
                ),
                "steps": ("INT", {"default": 20, "min": 2, "max": 200}),
                "seed": (
                    "INT",
                    {"default": -1, "min": -1, "max": 0xFFFFFFFFFFFFFFFF},
                ),
                "width": (
                    "INT",
                    {"default": 1024, "min": 64, "max": 65535, "step": 16},
                ),
                "height": (
                    "INT",
                    {"default": 1024, "min": 64, "max": 65535, "step": 16},
                ),
            },
            "optional": {
                "quantize": (QUANTIZE_OPTIONS, {"default": "8"}),
                "guidance": (
                    "FLOAT",
                    {"default": 4.0, "min": 0.0, "max": 50.0, "step": 0.1},
                ),
                "scheduler": ("STRING", {"default": ""}),
                "controlnet_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01},
                ),
                "controlnet_model": (
                    "STRING",
                    {"default": "xlabs-ai/flux-controlnet-canny-v3"},
                ),
                "model_path": ("STRING", {"default": ""}),
                "lora_paths": ("STRING", {"multiline": True, "default": ""}),
                "lora_scales": ("STRING", {"default": ""}),
                "prompt_file": ("STRING", {"default": ""}),
                "seeds": ("STRING", {"default": ""}),
                "auto_seeds": ("INT", {"default": -1, "min": -1, "max": 1024}),
                "output": ("STRING", {"default": ""}),
                "metadata": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "MFlux API"

    def run(
        self,
        model,
        controlnet_images,
        prompt,
        steps,
        seed,
        width,
        height,
        **kwargs,
    ):
        if (
            controlnet_images is None
            or not hasattr(controlnet_images, "shape")
            or int(controlnet_images.shape[0]) <= 0
        ):
            raise ValueError("MFluxControlNet requires at least one IMAGE")

        prompt_value = _read_prompt(prompt, kwargs.get("prompt_file", ""))
        quantize = kwargs.get("quantize", "8")
        quantize_value = None if quantize == "none" else int(quantize)
        model_path = kwargs.get("model_path", "") or None
        lora_paths = _parse_list_arg(kwargs.get("lora_paths", ""))
        lora_scales = _parse_list_arg(kwargs.get("lora_scales", ""), float)
        scheduler = kwargs.get("scheduler", "") or "linear"
        controlnet_strength = kwargs.get("controlnet_strength", 1.0)
        seed_values = _resolve_seeds(
            seed, kwargs.get("seeds", ""), kwargs.get("auto_seeds", -1)
        )
        model_config = _get_controlnet_model_config(model)
        controlnet_model = kwargs.get("controlnet_model", "") or None

        model_instance = _invoke_with_supported_kwargs(
            Flux1Controlnet,
            {
                "quantize": quantize_value,
                "model_path": model_path,
                "lora_paths": lora_paths,
                "lora_scales": lora_scales,
                "controlnet_path": controlnet_model,
                "model_config": model_config,
            },
        )

        temp_dir, controlnet_image_paths = _collect_batch_image_paths(controlnet_images)
        try:
            output_images = []
            export_output = kwargs.get("output", "") or None
            export_metadata = kwargs.get("metadata", False)

            for current_seed in seed_values:
                for controlnet_image_path in controlnet_image_paths:
                    generated = _invoke_with_supported_kwargs(
                        model_instance.generate_image,
                        {
                            "seed": int(current_seed),
                            "prompt": prompt_value,
                            "controlnet_image_path": controlnet_image_path,
                            "num_inference_steps": steps,
                            "height": height,
                            "width": width,
                            "guidance": kwargs.get("guidance", 4.0),
                            "controlnet_strength": controlnet_strength,
                            "scheduler": scheduler,
                        },
                    )

                    generated_images = _extract_pil_images(generated)
                    output_images.extend(generated_images)

                    if export_output:
                        for index, image in enumerate(generated_images):
                            output_path = export_output
                            if "{seed}" in output_path:
                                output_path = output_path.format(seed=int(current_seed))
                            if len(generated_images) > 1 and "{index}" in output_path:
                                output_path = output_path.format(index=index)

                            if hasattr(generated, "save"):
                                try:
                                    generated.save(
                                        path=output_path,
                                        export_json_metadata=export_metadata,
                                    )
                                    continue
                                except TypeError:
                                    pass

                            image.save(output_path)

            tensors = [_to_tensor(image) for image in output_images]
            if not tensors:
                raise ValueError("No images were generated")
            return (torch.cat(tensors, dim=0),)
        finally:
            if temp_dir and tempfile.gettempdir() in temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)
