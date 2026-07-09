# Neural Style Transfer with NIN and Color Preservation

A Lua/Torch7 implementation of neural artistic style transfer based on the algorithm introduced by Gatys et al. (2015), using the **Network in Network (NIN)** architecture as the feature extractor. The project includes two Python preprocessing utilities for luminance-based and linear color transfer, enabling color-independent style synthesis.

Submitted as a Bachelor of Engineering college project.

---

## Table of Contents

1. [Background](#background)
2. [Architecture](#architecture)
3. [Loss Functions](#loss-functions)
4. [Color Preservation](#color-preservation)
5. [Requirements](#requirements)
6. [Installation](#installation)
7. [Usage](#usage)
8. [Parameters](#parameters)
9. [File Structure](#file-structure)
10. [References](#references)

---

## Background

Neural style transfer optimizes a generated image $\hat{x}$ such that it simultaneously matches the **content** of a content image $p$ and the **style** of a style image $a$. The core idea is to treat the feature activations of a pretrained CNN as perceptual representations and minimize a weighted combination of content and style losses via gradient descent directly on the pixel values of $\hat{x}$.

The total loss is:

$$\mathcal{L}_{\text{total}}(\hat{x}, p, a) = \alpha \, \mathcal{L}_{\text{content}}(\hat{x}, p) + \beta \, \mathcal{L}_{\text{style}}(\hat{x}, a) + \gamma \, \mathcal{L}_{\text{TV}}(\hat{x})$$

where $\alpha$, $\beta$, and $\gamma$ are scalar weighting hyperparameters.

---

## Architecture

This implementation uses the **Network in Network (NIN)** model (`nin_imagenet_conv.caffemodel`) as the feature backbone, in contrast to the VGG-19 model used in the original paper. NIN uses $1 \times 1$ convolutions (mlpconv layers) to build micro-networks within each receptive field, resulting in:

| Property | NIN | VGG-19 (reference) |
|---|---|---|
| Parameters | ~7.6M | ~143M |
| Architecture depth | 12 layers | 19 layers |
| Pooling | Global Average | Max |
| GPU memory footprint | Low | High |

The reduced parameter count allows processing images at resolutions up to 1080 pixels on hardware with as little as 4 GB of VRAM.

The `convis.lua` script provides a complementary CNN feature map visualization tool that forwards a content image through a selected layer and saves the sum of feature maps as a grayscale image. This is useful for understanding which layers encode spatial structure versus texture.

### Default Layer Configuration

| Role | Layers |
|---|---|
| Content | `relu1`, `relu7`, `relu12` |
| Style | `relu1`, `relu3`, `relu5`, `relu7`, `relu9` |

### Image Preprocessing

Images are preprocessed to match the Caffe model's expected input format:

1. Rescale pixel values from $[0, 1]$ to $[0, 256]$
2. Convert channel order from RGB to BGR
3. Subtract the ImageNet mean pixel: $\mu = (103.939,\ 116.779,\ 123.68)$ (BGR)

$$x_{\text{caffe}} = \text{BGR}(256 \cdot x_{\text{rgb}}) - \mu$$

---

## Loss Functions

### Content Loss

Let $F^l(\hat{x}) \in \mathbb{R}^{C_l \times H_l W_l}$ denote the feature map of the generated image at layer $l$, and $P^l$ the corresponding map of the content image. The content loss is the mean squared error between these activations:

$$\mathcal{L}_{\text{content}}(\hat{x}, p, l) = \frac{1}{2} \sum_{i,j} \left( F^l_{ij}(\hat{x}) - P^l_{ij} \right)^2$$

### Style Loss

Style is captured via the **Gram matrix** $G^l \in \mathbb{R}^{C_l \times C_l}$, which encodes pairwise feature correlations across all spatial positions:

$$G^l_{ij}(\hat{x}) = \frac{1}{C_l H_l W_l} \sum_{k=1}^{H_l W_l} F^l_{ik}(\hat{x}) \; F^l_{jk}(\hat{x})$$

The style loss over all style layers $\mathcal{S}$ with per-layer weights $w_l$ is:

$$\mathcal{L}_{\text{style}}(\hat{x}, a) = \sum_{l \in \mathcal{S}} w_l \sum_{i,j} \left( G^l_{ij}(\hat{x}) - A^l_{ij} \right)^2$$

where $A^l$ is the Gram matrix of the style image $a$ at layer $l$.

### Total Variation Regularization

The anisotropic total variation loss penalizes pixel-level discontinuities to encourage spatial smoothness:

$$\mathcal{L}_{\text{TV}}(\hat{x}) = \sum_{c,i,j} \left( \hat{x}_{c,i,j+1} - \hat{x}_{c,i,j} \right)^2 + \left( \hat{x}_{c,i+1,j} - \hat{x}_{c,i,j} \right)^2$$

### Multi-Style Blending

For $N$ style images with user-specified weights $\{v_i\}_{i=1}^N$, the normalized blend weights are:

$$w_i = \frac{v_i}{\displaystyle\sum_{k=1}^{N} v_k}$$

The blended target Gram matrix at each style layer is accumulated as:

$$A^l_{\text{blend}} = \sum_{i=1}^{N} w_i \, A^l_i$$

---

## Color Preservation

Two Python scripts provide color-independent style transfer by decoupling luminance from chrominance before and after optimization.

### Luminance Transfer (`lum-transfer.py`)

Projects images onto the luminance channel using the ITU-R BT.601 luma coefficients:

$$Y = 0.299 R + 0.587 G + 0.114 B$$

**Mode `lum`:** Both content and style images are projected to grayscale. The style mean luminance is shifted to match content mean luminance, then style transfer is run on these luminance-only inputs.

**Mode `lum2`:** After style transfer, the generated image's mean intensity replaces the luminance ($L$) channel of the original content image in YUV space, then converts back to RGB. This composites the stylized structure back into the original colors.

**Mode `match`:** Applies PCA-based linear color transfer to match the style image color distribution to the content image.

**Mode `match_style`:** Applies PCA-based linear color transfer to match the content image color distribution to the style image.

### Linear Color Transfer (`linear-color-transfer.py`)

Matches the full RGB color distribution of a target image to a source image using one of three linear transforms derived from the image covariance matrices $C_t$ (target) and $C_s$ (source):

| Mode | Transform matrix $T$ |
|---|---|
| `pca` | $T = Q_s Q_t^{-1}$, where $Q = E \sqrt{\Lambda} E^\top$ (eigen-decomposition) |
| `chol` | $T = L_s L_t^{-1}$, where $L$ is the Cholesky factor of the covariance |
| `sym` | $T = Q_t^{-1} (Q_t C_s Q_t)^{1/2} Q_t^{-1}$ (symmetric square root) |

The transformed pixel vector is:

$$\hat{t} = T (t - \mu_t) + \mu_s$$

where $\mu_t$ and $\mu_s$ are the per-channel mean pixel values of the target and source images, respectively.

---

## Requirements

### Lua / Torch7

| Package | Purpose |
|---|---|
| `torch` | Core tensor library and neural network framework |
| `nn` | Neural network module library |
| `image` | Image loading, saving, and scaling |
| `optim` | L-BFGS and Adam optimization |
| `loadcaffe` | Loading pretrained Caffe `.caffemodel` / `.prototxt` files |
| `cutorch` + `cunn` | CUDA GPU acceleration (optional) |
| `cudnn` | NVIDIA cuDNN backend (optional) |
| `clnn` + `cltorch` | OpenCL backend (optional) |

### Python

| Package | Version | Purpose |
|---|---|---|
| `numpy` | >= 1.19 | Array math and linear algebra |
| `scikit-image` | >= 0.18 | Image I/O (`imread`, `imsave`) and resizing |

Install Python dependencies:

```bash
pip install numpy scikit-image
```

---

## Installation

See [INSTALL.md](INSTALL.md) for the full step-by-step guide covering Torch7, `loadcaffe`, CUDA, and cuDNN setup.

### Quick Start (CPU)

```bash
# 1. Install Torch7
curl -s https://raw.githubusercontent.com/torch/ezinstall/master/install-deps | bash
git clone https://github.com/torch/distro.git ~/torch --recursive
cd ~/torch && ./install.sh
source ~/.bashrc

# 2. Install loadcaffe
sudo apt-get install libprotobuf-dev protobuf-compiler
luarocks install loadcaffe

# 3. Download pretrained models
bash models/download_models.sh
```

---

## Usage

### Style Transfer (`creating.lua`)

```bash
th creating.lua \
  -content_image examples/inputs/tubingen.jpg \
  -style_image examples/inputs/starry_night.jpg \
  -output_image out.png \
  -gpu -1
```

**GPU mode (CUDA + cuDNN):**

```bash
th creating.lua \
  -content_image examples/inputs/tubingen.jpg \
  -style_image examples/inputs/starry_night.jpg \
  -gpu 0 \
  -backend cudnn \
  -output_image out.png
```

**Multi-style blending:**

```bash
th creating.lua \
  -style_image style1.jpg,style2.jpg \
  -style_blend_weights 0.7,0.3 \
  -content_image content.jpg \
  -output_image out.png
```

**Multi-GPU:**

```bash
th creating.lua \
  -content_image content.jpg \
  -style_image style.jpg \
  -gpu 0,1 \
  -multigpu_strategy 3 \
  -output_image out.png
```

### Feature Map Visualization (`convis.lua`)

```bash
th convis.lua \
  -content_image examples/inputs/tubingen.jpg \
  -layer relu4_2 \
  -output_image relu4_2_map.png
```

### Luminance Preprocessing (`lum-transfer.py`)

```bash
# Step 1: Convert to luminance only
python lum-transfer.py \
  --cp_mode lum \
  --content_image content.jpg \
  --style_image style.jpg \
  --output_content_image content_lum.png \
  --output_style_image style_lum.png

# Step 2: Run style transfer on luminance images (see above)

# Step 3: Composite luminance result back into original color
python lum-transfer.py \
  --cp_mode lum2 \
  --org_content content.jpg \
  --output_lum2 out_lum.png \
  --output_image final_color_preserved.png
```

### Linear Color Transfer (`linear-color-transfer.py`)

```bash
python linear-color-transfer.py \
  --target_image out.png \
  --source_image content.jpg \
  --output_image color_corrected.png \
  --mode pca
```

### Recommended Workflow for Color-Independent Style Transfer

```bash
# 1. Preprocess: project both images to luminance
python lum-transfer.py --cp_mode lum \
  --content_image content.jpg --style_image style.jpg \
  --output_content_image content_lum.png \
  --output_style_image style_lum.png

# 2. Run style transfer on luminance images
th creating.lua \
  -content_image content_lum.png \
  -style_image style_lum.png \
  -output_image out_lum.png \
  -gpu 0

# 3. Composite luminance back into original color
python lum-transfer.py --cp_mode lum2 \
  --org_content content.jpg \
  --output_lum2 out_lum.png \
  --output_image final_color_preserved.png
```

---

## Parameters

### `creating.lua`

| Flag | Default | Description |
|---|---|---|
| `-content_image` | `examples/inputs/tubingen.jpg` | Path to the content image |
| `-style_image` | `examples/inputs/seated-nude.jpg` | Comma-separated style image path(s) |
| `-style_blend_weights` | `nil` (equal) | Comma-separated blend weights for multiple styles |
| `-image_size` | `1080` | Maximum output dimension in pixels |
| `-gpu` | `0` | GPU index (0-indexed); `-1` for CPU |
| `-multigpu_strategy` | `''` | Layer indices at which to split across GPUs |
| `-content_weight` | `10` | Content loss weight $\alpha$ |
| `-style_weight` | `1000` | Style loss weight $\beta$ |
| `-tv_weight` | `0.0001` | Total variation regularization weight $\gamma$ |
| `-num_iterations` | `1000` | Total optimization iterations |
| `-normalize_gradients` | `false` | Normalize loss gradients by L1 norm |
| `-init` | `random` | Initialization: `random` or `image` |
| `-init_image` | `''` | Path to initialization image (if `-init image`) |
| `-optimizer` | `adam` | Optimizer: `adam` or `lbfgs` |
| `-learning_rate` | `10` | Learning rate (Adam only) |
| `-lbfgs_num_correction` | `0` | Number of L-BFGS history corrections |
| `-print_iter` | `50` | Print loss every N iterations (`0` = disabled) |
| `-save_iter` | `100` | Save intermediate image every N iterations (`0` = disabled) |
| `-output_image` | `out.png` | Final output image path |
| `-style_scale` | `1.0` | Scale factor applied to style image before processing |
| `-original_colors` | `0` | Set `1` to preserve content image colors via YUV |
| `-pooling` | `max` | Pooling type: `max` or `avg` |
| `-proto_file` | `models/train_val.prototxt` | Network architecture file |
| `-model_file` | `models/nin_imagenet_conv.caffemodel` | Pretrained model weights |
| `-backend` | `nn` | Compute backend: `nn`, `cudnn`, or `clnn` |
| `-cudnn_autotune` | `false` | Enable cuDNN autotuning |
| `-seed` | `-1` | Random seed; `-1` = unseeded |
| `-content_layers` | `relu1,relu7,relu12` | Layers used for content loss |
| `-style_layers` | `relu1,relu3,relu5,relu7,relu9` | Layers used for style loss |

### `convis.lua`

| Flag | Default | Description |
|---|---|---|
| `-content_image` | `examples/inputs/tubingen.jpg` | Input image path |
| `-image_size` | `800` | Resize longest dimension to this value (pixels) |
| `-proto_file` | VGG-19 prototxt | Network architecture file |
| `-model_file` | VGG-19 caffemodel | Pretrained weights file |
| `-layer` | `relu4_2` | Layer whose feature maps to visualize |
| `-output_image` | `out.png` | Output visualization path |
| `-seed` | `876` | Random seed |

### `lum-transfer.py`

| Flag | Default | Description |
|---|---|---|
| `--cp_mode` | `lum` | Mode: `lum`, `lum2`, `match`, `match_style` |
| `--content_image` | -- | Path to the content image |
| `--style_image` | -- | Path to the style image |
| `--output_content_image` | `output_content.png` | Output path for processed content image |
| `--output_style_image` | `output_style.png` | Output path for processed style image |
| `--org_content` | `original_content_image.png` | Original content image (for `lum2` mode) |
| `--output_lum2` | `out.png` | Style-transferred image to composite (for `lum2` mode) |
| `--output_image` | `output_style.png` | Final output image path |

### `linear-color-transfer.py`

| Flag | Default | Description |
|---|---|---|
| `--target_image` | (required) | Image to transfer color onto |
| `--source_image` | (required) | Image to sample color distribution from |
| `--output_image` | `output.png` | Path to save the result |
| `--mode` | `pca` | Transfer mode: `pca`, `chol`, or `sym` |
| `--eps` | `1e-5` | Regularization epsilon for covariance matrix stability |

---

## File Structure

```
neural-style-transfer/
├── creating.lua                  # Main style transfer script
├── convis.lua                    # CNN feature map visualization
├── linear-color-transfer.py      # Linear RGB color distribution transfer
├── lum-transfer.py               # Luminance-based color preservation
├── INSTALL.md                    # Full installation guide
├── LICENSE                       # MIT License
├── .gitignore
├── models/
│   ├── train_val.prototxt                    # NIN network architecture
│   ├── deploy_10.prototxt                    # Alternative NIN architecture
│   ├── VGG_ILSVRC_19_layers_deploy.prototxt  # VGG-19 architecture (for convis)
│   ├── solver.prototxt                       # Caffe solver config
│   └── download_models.sh                    # Script to download pretrained weights
├── examples/
│   ├── inputs/                               # Content and style images
│   ├── outputs/                              # Generated output images
│   └── multigpu_scripts/
│       └── starry_stanford.sh                # Example multi-GPU run script
└── report/
    ├── CREATING_ARTISTIC_IMAGES(...).pdf     # Project report
    ├── finalppt.key                          # Presentation slides
    ├── mjcontent.jpg                         # Report figure: content image
    └── fig3_style1.jpg                       # Report figure: style image
```

---

## References

1. Gatys, L. A., Ecker, A. S., and Bethge, M. (2016). "A Neural Algorithm of Artistic Style." *Journal of Vision*, 16(12), 326. [arXiv:1508.06576](https://arxiv.org/abs/1508.06576)
2. Lin, M., Chen, Q., and Yan, S. (2014). "Network In Network." *ICLR 2014*. [arXiv:1312.4400](https://arxiv.org/abs/1312.4400)
3. Johnson, J. (2015). `neural-style`. GitHub. [github.com/jcjohnson/neural-style](https://github.com/jcjohnson/neural-style)
4. Gatys, L. A., Ecker, A. S., Bethge, M., Hertzmann, A., and Shechtman, E. (2016). "Controlling Perceptual Factors in Neural Style Transfer." *CVPR 2017*. [arXiv:1611.07865](https://arxiv.org/abs/1611.07865)
5. Simonyan, K. and Zisserman, A. (2015). "Very Deep Convolutional Networks for Large-Scale Image Recognition." *ICLR 2015*. [arXiv:1409.1556](https://arxiv.org/abs/1409.1556)

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
