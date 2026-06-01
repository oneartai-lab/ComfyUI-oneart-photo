# OneArt Photo

Professional photo finishing nodes for ComfyUI.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom_Nodes-orange.svg)](https://github.com/comfyanonymous/ComfyUI)
[![Release v0.1.0](https://img.shields.io/badge/release-v0.1.0-green.svg)](https://github.com/oneartai-lab/ComfyUI-oneart-photo/releases/tag/v0.1.0)

OneArt Photo is an open-source custom node pack that brings photography-inspired post-processing workflows to ComfyUI. It focuses on image finishing, metadata embedding, lens effects, LUT-based color grading, and camera-style transformations for AI-generated content.

Designed for creators, photographers, and workflow builders who want more control over the final visual look of generated images.

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

## ⚡ Features

### 🎞️ Photo Finishing Workflow
Add post-processing directly inside ComfyUI without leaving your generation pipeline.
* Lens distortion simulation
* Camera-style image transforms
* Creative photo presets
* LUT color grading workflows
* Metadata embedding

### 🎨 LUT Processing
Apply professional color grading workflows using LUTs.
* Cinematic looks
* Film-inspired color styles
* Consistent visual branding
* Reusable grading pipelines

### 📝 Metadata Injection
Store workflow information directly inside exported images.
* **Asset management:** Easily track settings and styles.
* **Workflow reproducibility:** Retrieve generation settings from exported files.
* **Production pipelines:** Automate indexing and categorization.
* **Dataset organization:** Standardize tags and camera profiles.

### 🔌 ComfyUI Native Integration
Built as native custom nodes for ComfyUI.
* Lightweight & modular
* Workflow-friendly
* Fully compatible with existing pipelines

---

## 🧩 Included Nodes

To keep the interface clean while supporting advanced features, OneArt Photo groups its physical nodes into four main modules:

| Module / Action | ComfyUI Node Class | Description |
| :--- | :--- | :--- |
| **Metadata Injection** | `OneArt Photo Metadata`<br>`OneArt Photo Save JPEG Direct` | Embeds EXIF metadata payloads (presets or overrides) directly into JPEGs. |
| **Lens Effects** | `OneArt Photo Lens Warp`<br>`OneArt Photo Vignette` | Simulates lens distortion coefficients and quadratic vignetting falloffs. |
| **LUT Processor** | `OneArt Photo LUT` | Applies `.cube` LUT color lookup tables with custom intensity blending. |
| **Photo Presets / Styles** | `OneArt Photo All In One`<br>`OneArt Photo Style FX`<br>`OneArt Photo Noise`<br>`OneArt Photo Grain`<br>`OneArt Photo Tone Adjust` | High-level presets (Bloom, Halation, Cinematic Grade, Soft Portrait), film grain, sensor noise, and fine-tuned tone curves. |
| **File I/O** | `OneArt Photo Load RAW / HEIC`<br>`OneArt Photo Save RAW` | Loads RAW / HEIC formats and saves uncompressed 16-bit DNG/TIFF files. |

---

## ⚙️ Installation

### Option 1: Via ComfyUI Manager
Search for **`OneArt Photo`** in the custom nodes catalog and click **Install**.

### Option 2: Manual Installation
1. Clone the repository into your ComfyUI `custom_nodes` folder:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/oneartai-lab/ComfyUI-oneart-photo.git
   ```
2. Install python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Restart ComfyUI.

---

## 🔄 Example Workflow

```mermaid
graph TD
    Gen[Image Generation] --> Finish[Photo Finishing & Tone Adjust]
    Finish --> Lens[Lens Effects & Vignette]
    Lens --> LUT[LUT Processing / Color Grading]
    LUT --> Meta[Metadata Export / Save JPEG]
    Meta --> Prod[Final Production Image]
    style Gen fill:#f9f,stroke:#333,stroke-width:2px
    style Prod fill:#bbf,stroke:#333,stroke-width:2px
```

*You can find ready-to-use ComfyUI workflow JSON files in the [workflows/](workflows/) directory.*

---

## 💡 Use Cases

### Portrait Enhancement
Add subtle photographic finishing, lens characteristics, and color grading to improve portrait realism.

### Social Media Content
Create consistent visual styles and color grading profiles across multiple generated images.

### Photography Simulation
Apply physics-based camera sensor noise and lens configurations to AI-generated outputs.

### Production Pipelines
Integrate finishing steps directly into automated, batch-processed ComfyUI workflows.

---

## ❓ Why OneArt Photo?

Many ComfyUI workflows focus on the generation phase. OneArt Photo focuses on the final stage:

* Image finishing & polishing
* Presentation quality
* Color consistency
* EXIF metadata preservation
* Photography-inspired rendering

This makes it easier to build production-ready workflows entirely inside ComfyUI.

---

## 🗺️ Project Roadmap

### v0.1
* Metadata injection
* Lens distortion
* LUT support
* Preset system

### v0.2
* Additional camera simulations
* Batch processing support
* Advanced metadata templates

### v0.3
* Film emulation workflows
* Extended finishing toolkit
* Creator workflow presets

---

## 🤝 Contributing

Issues, feature requests, pull requests, and workflow examples are welcome! If you build interesting workflows with OneArt Photo, feel free to share them in the issues or discussions.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 💖 Acknowledgements

Built for the ComfyUI ecosystem and the open-source AI creator community.
