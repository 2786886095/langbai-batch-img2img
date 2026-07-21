# Langbai 批量图生图

一个轻量的 ComfyUI 自定义节点：从任意本地绝对路径读取图片，按自然文件名顺序排列，并与多行正面提示词逐项配对后送入下游工作流。

## 功能

- 点击一次 Queue，依次处理全部“图片 + 正面提示词”配对。
- 自然排序：`1.png`、`2.png`、`10.png`。
- 每一行非空提示词对应一张图片，空白行自动忽略。
- 图片数量与非空提示词行数不一致时，在工作流开始执行前报错并停止。
- 每张图片保持自身原始尺寸，不缩放、不裁剪；不同尺寸也可以放在同一任务中依次处理。
- 只读取指定文件夹当前层，不扫描子文件夹。
- 支持 PNG、JPG/JPEG、WebP、BMP、TIFF。
- 不引入 ComfyUI 之外的第三方依赖。

## 安装

在 ComfyUI 的 `custom_nodes` 目录执行：

```powershell
git clone https://github.com/2786886095/langbai-batch-img2img.git
```

然后重启 ComfyUI。在节点菜单的 `Langbai > 批量图生图` 中添加 **Langbai 批量图生图**。

## 使用

1. 在 `image_directory` 填入图片文件夹的绝对路径，例如 `F:\图片\待处理`。
2. 在 `positive_prompts` 中每行输入一条正面提示词。
3. 确保非空提示词行数与文件夹当前层的图片数量完全一致。
4. 将 `images` 连接到 VAE Encode 等图像输入节点。
5. 将 `positive_prompts` 连接到 CLIP Text Encode 的 `text` 输入。
6. 按正常图生图工作流连接 VAE、KSampler、负面提示词和 Save Image，点击一次 Queue。

推荐连接关系：

```text
Langbai 批量图生图.images
  -> VAE Encode.pixels
  -> KSampler.latent_image

Langbai 批量图生图.positive_prompts
  -> CLIP Text Encode.text
  -> KSampler.positive
```

ComfyUI 会按照列表索引依次运行下游节点，因此第 1 张图片只会使用第 1 行提示词，第 2 张图片只会使用第 2 行提示词，依此类推。

## 重要说明

- 本节点只负责提供图片和正面提示词，不加载模型、不编码 VAE、不采样、不保存图片。
- 路径可以位于 `ComfyUI/input` 之外。请只使用你信任的本地目录。
- 图片会先全部读取到系统内存，再交给下游逐项执行；非常大的批次需要留意内存占用。
- 带 EXIF 方向信息的照片会先按该方向转正，输出尺寸以转正后的画面为准。
- 动图只读取第一帧。

## 开发测试

在仓库根目录执行：

```powershell
python -m unittest discover -s tests -v
```

## 许可证

[MIT](LICENSE)
