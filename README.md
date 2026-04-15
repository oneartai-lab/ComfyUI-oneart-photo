# OneArt Photo

`oneart-photo` is a ComfyUI custom node pack for image finishing, metadata injection, and camera-style transforms.

## Included nodes

- OneArt Photo Noise
- OneArt Photo Tone Adjust
- OneArt Photo Style FX
- OneArt Photo LUT
- OneArt Photo Grain
- OneArt Photo Metadata
- OneArt Photo Load RAW / HEIC
- OneArt Photo Save JPEG
- OneArt Photo Save JPEG Direct
- OneArt Photo Save RAW
- OneArt Photo Sensor Noise
- OneArt Photo Lens Warp
- OneArt Photo All In One

## Design goals

- Clean, original implementation
- Neutral naming and branding
- Practical image finishing instead of detector-focused framing
- Optional metadata and file export support

## Install

1. Put the folder in `ComfyUI/custom_nodes`
2. Install dependencies from `requirements.txt`
3. Restart ComfyUI

## LUT folder

- Put LUT files into `oneart-photo/luts`
- Supported files: `.cube`, `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`
- `OneArt Photo LUT` shows the folder contents in a dropdown list

## Notes

- `pillow-heif` enables HEIC loading when available.
- `rawpy` enables RAW loading when available.
- `tifffile` improves TIFF/DNG-style output when available.

## License

This rewrite is authored for OneArt Photo.
