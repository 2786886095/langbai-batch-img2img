from .nodes import LangbaiBatchImg2ImgInput


NODE_CLASS_MAPPINGS = {
    "LangbaiBatchImg2ImgInput": LangbaiBatchImg2ImgInput,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LangbaiBatchImg2ImgInput": "Langbai 批量图生图",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
