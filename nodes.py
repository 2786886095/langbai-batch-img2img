from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError


SUPPORTED_IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}


def _natural_sort_key(path: Path) -> tuple[object, ...]:
    """Sort digit runs numerically: 1.png, 2.png, 10.png."""
    parts = re.split(r"(\d+)", path.name.casefold())
    return tuple(int(part) if part.isdigit() else part for part in parts)


def _parse_prompts(value: str) -> list[str]:
    """Return non-empty, trimmed prompt lines without shifting later pairs."""
    return [line.strip() for line in value.splitlines() if line.strip()]


def _normalize_directory(value: str) -> Path:
    cleaned = value.strip().strip('"').strip("'")
    expanded = os.path.expandvars(os.path.expanduser(cleaned))
    return Path(expanded)


def _find_images(directory: Path) -> list[Path]:
    images = [
        item
        for item in directory.iterdir()
        if item.is_file() and item.suffix.casefold() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    return sorted(images, key=lambda path: (_natural_sort_key(path), path.name.casefold()))


def _validate_and_collect(
    image_directory: str, positive_prompts: str
) -> tuple[Path, list[Path], list[str]]:
    directory = _normalize_directory(image_directory)

    if not image_directory.strip():
        raise ValueError("图片文件夹不能为空。")
    if not directory.is_absolute():
        raise ValueError(f"图片文件夹必须是绝对路径：{image_directory}")
    if not directory.exists():
        raise ValueError(f"图片文件夹不存在：{directory}")
    if not directory.is_dir():
        raise ValueError(f"给定路径不是文件夹：{directory}")

    try:
        images = _find_images(directory)
    except OSError as exc:
        raise ValueError(f"无法读取图片文件夹：{directory}（{exc}）") from exc

    prompts = _parse_prompts(positive_prompts)

    if not images:
        extensions = ", ".join(sorted(SUPPORTED_IMAGE_EXTENSIONS))
        raise ValueError(
            f"文件夹当前层没有支持的图片：{directory}。支持格式：{extensions}"
        )
    if not prompts:
        raise ValueError("正面提示词不能为空；请输入至少一行非空提示词。")
    if len(images) != len(prompts):
        raise ValueError(
            "图片与提示词数量不一致，任务未开始："
            f"找到 {len(images)} 张图片、{len(prompts)} 行非空提示词。"
        )

    return directory, images, prompts


def _load_image(path: Path) -> torch.Tensor:
    try:
        with Image.open(path) as source:
            source.seek(0)
            transposed = ImageOps.exif_transpose(source)
            rgb = transposed.convert("RGB")
            pixels = np.asarray(rgb, dtype=np.float32) / 255.0
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise RuntimeError(f"图片读取失败：{path}（{exc}）") from exc

    # ComfyUI IMAGE: [batch, height, width, channels], float values in [0, 1].
    return torch.from_numpy(pixels).unsqueeze(0)


def _file_state(paths: Iterable[Path]) -> Iterable[str]:
    for path in paths:
        stat = path.stat()
        yield f"{path.name}\0{stat.st_size}\0{stat.st_mtime_ns}"


class LangbaiBatchImg2ImgInput:
    """Load ordered image/prompt pairs as ComfyUI execution lists."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_directory": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": r"例如：F:\图片\待处理",
                        "tooltip": "任意本地绝对路径；仅扫描当前层，不扫描子文件夹。",
                    },
                ),
                "positive_prompts": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": False,
                        "placeholder": "每行一条正面提示词，与自然排序后的图片一一对应",
                        "tooltip": "空白行会被忽略；非空行数必须与图片数完全一致。",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "positive_prompts")
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "load_pairs"
    CATEGORY = "Langbai/批量图生图"
    DESCRIPTION = (
        "按自然文件名顺序加载文件夹当前层的图片，并与逐行正面提示词一一配对。"
        "数量不一致时会在工作流执行前报错。"
    )
    SEARCH_ALIASES = [
        "batch img2img",
        "batch image prompt loader",
        "批量图生图",
        "按行提示词",
    ]

    @classmethod
    def VALIDATE_INPUTS(cls, image_directory: str, positive_prompts: str):
        try:
            _validate_and_collect(image_directory, positive_prompts)
        except ValueError as exc:
            return str(exc)
        return True

    @classmethod
    def IS_CHANGED(cls, image_directory: str, positive_prompts: str):
        try:
            directory, images, prompts = _validate_and_collect(
                image_directory, positive_prompts
            )
            state = [str(directory.resolve()), *_file_state(images), *prompts]
        except (OSError, ValueError):
            state = [image_directory, positive_prompts]

        digest = hashlib.sha256()
        for value in state:
            digest.update(value.encode("utf-8", errors="surrogatepass"))
            digest.update(b"\0")
        return digest.hexdigest()

    def load_pairs(self, image_directory: str, positive_prompts: str):
        _directory, image_paths, prompts = _validate_and_collect(
            image_directory, positive_prompts
        )

        # Load every source before returning any output. If a source is corrupt,
        # downstream sampling nodes do not receive a partial list.
        images = [_load_image(path) for path in image_paths]
        return images, prompts
