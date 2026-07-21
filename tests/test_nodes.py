import os
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from nodes import (
    LangbaiBatchImg2ImgInput,
    _find_images,
    _parse_prompts,
)


class LangbaiBatchImg2ImgInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def make_image(self, name, size, color):
        path = self.root / name
        Image.new("RGB", size, color).save(path)
        return path

    def test_natural_sort_only_reads_current_directory(self):
        self.make_image("10.png", (10, 10), (10, 0, 0))
        self.make_image("2.PNG", (10, 10), (2, 0, 0))
        self.make_image("1.jpg", (10, 10), (1, 0, 0))
        (self.root / "notes.txt").write_text("not an image", encoding="utf-8")
        child = self.root / "child"
        child.mkdir()
        Image.new("RGB", (10, 10)).save(child / "0.png")

        self.assertEqual(
            [path.name for path in _find_images(self.root)],
            ["1.jpg", "2.PNG", "10.png"],
        )

    def test_blank_prompt_lines_are_ignored(self):
        self.assertEqual(
            _parse_prompts(" first prompt \n\n   \n second prompt\n"),
            ["first prompt", "second prompt"],
        )

    def test_count_mismatch_is_rejected_before_execution(self):
        self.make_image("1.png", (8, 8), (0, 0, 0))
        self.make_image("2.png", (8, 8), (0, 0, 0))

        result = LangbaiBatchImg2ImgInput.VALIDATE_INPUTS(
            str(self.root), "only one prompt"
        )

        self.assertIsInstance(result, str)
        self.assertIn("2 张图片", result)
        self.assertIn("1 行非空提示词", result)

    def test_relative_directory_is_rejected(self):
        result = LangbaiBatchImg2ImgInput.VALIDATE_INPUTS(
            "relative/path", "prompt"
        )
        self.assertIsInstance(result, str)
        self.assertIn("绝对路径", result)

    def test_outputs_keep_order_pairing_and_original_dimensions(self):
        self.make_image("10.png", (9, 4), (250, 0, 0))
        self.make_image("2.png", (7, 5), (20, 0, 0))
        node = LangbaiBatchImg2ImgInput()

        images, prompts = node.load_pairs(
            str(self.root), "prompt for 2\nprompt for 10"
        )

        self.assertEqual(prompts, ["prompt for 2", "prompt for 10"])
        self.assertEqual(tuple(images[0].shape), (1, 5, 7, 3))
        self.assertEqual(tuple(images[1].shape), (1, 4, 9, 3))
        self.assertAlmostEqual(float(images[0][0, 0, 0, 0]), 20 / 255, places=6)
        self.assertAlmostEqual(float(images[1][0, 0, 0, 0]), 250 / 255, places=6)

    def test_corrupt_image_fails_without_partial_result(self):
        (self.root / "1.png").write_bytes(b"not a png")
        node = LangbaiBatchImg2ImgInput()

        with self.assertRaisesRegex(RuntimeError, "图片读取失败"):
            node.load_pairs(str(self.root), "prompt")

    def test_change_fingerprint_tracks_file_updates_and_prompts(self):
        image_path = self.make_image("1.png", (8, 8), (0, 0, 0))
        first = LangbaiBatchImg2ImgInput.IS_CHANGED(str(self.root), "prompt one")

        time.sleep(0.01)
        Image.new("RGB", (9, 8), (1, 0, 0)).save(image_path)
        os.utime(image_path, None)
        second = LangbaiBatchImg2ImgInput.IS_CHANGED(str(self.root), "prompt one")
        third = LangbaiBatchImg2ImgInput.IS_CHANGED(str(self.root), "prompt two")

        self.assertNotEqual(first, second)
        self.assertNotEqual(second, third)

    def test_node_declares_two_execution_lists(self):
        self.assertEqual(
            LangbaiBatchImg2ImgInput.RETURN_TYPES, ("IMAGE", "STRING")
        )
        self.assertEqual(LangbaiBatchImg2ImgInput.OUTPUT_IS_LIST, (True, True))


if __name__ == "__main__":
    unittest.main()
