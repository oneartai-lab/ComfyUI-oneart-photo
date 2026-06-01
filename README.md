# OneArt Photo for ComfyUI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom_Nodes-orange.svg)](https://github.com/comfyanonymous/ComfyUI)
[![Release v0.1.0](https://img.shields.io/badge/release-v0.1.0-green.svg)](https://github.com/oneartai-lab/ComfyUI-oneart-photo/releases/tag/v0.1.0)

**OneArt Photo** is a professional, lightweight, open-source custom node pack for ComfyUI. It is designed to bridge the gap between clinical AI generations and realistic photographic imagery by introducing high-fidelity post-processing, camera sensor simulation, lens distortion modeling, color science (LUTs), and EXIF metadata injection.

Whether you need to replicate analog film stock, emulate a modern CMOS sensor, apply professional color grading, or inject accurate camera information (EXIF) for workflow reproducibility and production pipelines, OneArt Photo provides a modular, clean, and highly optimized toolkit.

---

## 📸 Showcase & Results

Here is a visual demonstration of the OneArt Photo node suite in action:

![OneArt Photo Workflow Demonstration](assets/workflow_demo.gif)
*Example workflow applying camera sensor noise, color grading via LUT, and film grain.*

### Before & After Comparison

| Original AI Output | OneArt Photo Processed (Analog/Cinematic Look) |
| :---: | :---: |
| ![Original Image](assets/comparison_before.png) | ![Processed Image](assets/comparison_after.png) |

---

## ✨ Features

### 🎞️ Film & Sensor Simulation
* **Procedural Grain:** Simulates silver halide film structures with adjustable scale and intensity.
* **Sensor Noise:** Emulates CMOS/CCD sensor noise with Poisson-Gaussian shot and read noise algorithms, including color correlation.
* **Analog Effects:** Fast, high-quality implementations of Halation, Bloom, Soft Portrait diffusion, and Cinematic Grading.

### 🔍 Optical & Lens Modeling
* **Lens Distortion:** Modifies image perspective using realistic radial/tangential warp coefficients.
* **Vignetting:** Separately control inner and outer image brightness with smooth, quadratic falloffs to emulate physical lens shading.

### 🎨 Color Science & LUT Support
* **3D Lookup Tables:** Apply professional `.cube` files directly inside your ComfyUI workflow.
* **Image LUTs:** Supports 3D strip LUT files (`.png`, `.jpg`, `.jpeg`, `.tiff`).
* **Intensity Control:** Blend color transformations smoothly with real-time opacity controls.

### 📝 Production EXIF Metadata Injection
* **Camera Emulation:** Embed detailed EXIF metadata mimicking actual camera models (Canon, Nikon, Sony, Fujifilm, Leica, etc.).
* **Custom Fields:** Inject ISO, aperture (F-Number), focal length, shutter speed, exposure bias, white balance, and random/fixed hardware serial numbers.
* **File Integration:** Expose generation details directly in JPEG, TIFF, or DNG metadata to make outputs indistinguishable from camera photos.

---

## 🧩 Included Nodes

| Category | Node Name | Description |
| :--- | :--- | :--- |
| **Finishing & Effects** | `OneArt Photo Noise` | Adds high-frequency color noise with custom blue-channel bias. |
| | `OneArt Photo Grain` | Applies customizable procedural film grain. |
| | `OneArt Photo Tone Adjust` | Fine-tune brightness, contrast, shadows, highlights, midtones, and warmth. |
| | `OneArt Photo Vignette` | Simulates optical vignetting with independent center/corner control. |
| | `OneArt Photo Style FX` | Advanced artistic presets: *GlitchArt*, *SoftPortrait*, *CinematicGrade*, *Halation*, *Bloom*. |
| **Color Grading** | `OneArt Photo LUT` | Applies `.cube` or image-based LUTs with custom intensity blending. |
| **Optical / Lens** | `OneArt Photo Lens Warp` | Simulates optical aberrations and lens distorion. |
| **Metadata & I/O** | `OneArt Photo Metadata` | Builds EXIF data payload using presets or custom fields. |
| | `OneArt Photo Load RAW / HEIC` | Loads raw files (`.dng`, `.cr2`, `.nef`) and iPhone HEIC files directly. |
| | `OneArt Photo Save JPEG` | Saves images as JPEGs with embedded EXIF payloads. |
| | `OneArt Photo Save JPEG Direct` | Combines metadata injection and JPEG exporting into a single node. |
| | `OneArt Photo Save RAW` | Saves 16-bit TIFF or DNG files with full calibration matrix configurations. |
| **All-In-One** | `OneArt Photo All In One` | Unified node combining noise, presets, metadata, and JPEG output. |

---

## 🚀 Installation

### Option 1: Via ComfyUI Manager (Recommended)
1. Open ComfyUI and click on **Manager**.
2. Click **Install Custom Nodes**.
3. Search for `OneArt Photo`.
4. Click **Install** and restart ComfyUI.

### Option 2: Manual Installation
1. Clone the repository into your ComfyUI `custom_nodes` folder:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/oneartai-lab/ComfyUI-oneart-photo.git
   ```
2. Install the python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: If your system uses a virtual environment, ensure you run the pip command inside that environment.*
3. Restart ComfyUI.

#### Optional Dependencies
OneArt Photo will dynamically activate extra features if the following packages are present:
* `pillow-heif` — Enables iPhone HEIC image loading.
* `rawpy` — Enables professional digital RAW camera file loading.
* `tifffile` — Enables high-depth TIFF and calibration-accurate DNG exports.

---

## 📖 Examples & Workflows

We have provided sample workflows in the `workflows/` directory. Simply drag and drop any of these JSON files (or the output images themselves) into ComfyUI to load them:

### 1. Analog Portrait Finishing
* **Workflow File:** `workflows/analog_portrait_finishing.json`
* **Details:** This workflow inputs a clean portrait image, applies a subtle Canon preset, injects custom EXIF details, adds low-intensity sensor noise, runs a Cinematic color LUT, and outputs an optimized JPEG.

### 2. High-Dynamic DNG/TIFF Archiving
* **Workflow File:** `workflows/dng_archiving.json`
* **Details:** Simulates digital negative pipelines by embedding color matrices and illuminant calibrations into a DNG file format, useful for Lightroom/Photoshop grading.

---

## 🗺️ Project Roadmap

### v0.1.0 (Current Release)
- [x] Initial EXIF metadata builder and encoder.
- [x] Radial lens distortion simulation.
- [x] 3D LUT parser (`.cube`) and 3D strip LUT reader.
- [x] Procedural film grain and Poisson-Gaussian sensor noise.
- [x] HEIC/RAW loader and high-depth TIFF/DNG exporter.
- [x] OneArt Photo Vignette and Style FX nodes (Bloom, Halation, Cinematic Grade, Soft Portrait).

### v0.2.0 (Next Up)
- [ ] Direct workflow embedding inside EXIF metadata.
- [ ] Improved batch performance using CUDA-accelerated processing where available.
- [ ] Advanced camera lens aberration templates (chromatic aberration, barrel/pincushion presets).
- [ ] Interactive UI color previewer for LUTs.

### v0.3.0
- [ ] Custom curve editor node.
- [ ] Automated LUT matching based on target reference images.
- [ ] Support for ACEScg and custom film profiles.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions, issues, and feature requests are very welcome! Feel free to check the [issues page](https://github.com/oneartai-lab/ComfyUI-oneart-photo/issues) if you want to contribute or suggest improvements. 

---

## 💖 Acknowledgements

Special thanks to the ComfyUI community and AI creators pushing the boundaries of realistic photography. 
