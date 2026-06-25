# Multiple Barcode Decoding

A classical computer vision pipeline for detecting, correcting orientation, and decoding multiple barcodes from images. The system works entirely on CPU using OpenCV and PyZbar, requiring no trained models or GPU acceleration.

## Technical Report

A detailed explanation of the detection and decoding pipeline is available in the project report:

**https://drive.google.com/file/d/1cfUwYGxJ4ZCAhaIPrmn1VGlL1bj-ZyJ3/view?usp=sharing**


## Features

* Detect multiple barcodes in a single image
* Correct barcode orientation automatically
* Perspective correction for skewed tags
* Decode barcode contents
* Process individual images or entire folders
* Save annotated output images

## Project Structure

```text
.
├── main.py
├── full_folder_output.py
├── output/
│   ├── IMG_20260520_162128_175.jpg.jpeg
│   ├── IMG_20260520_162133_348.jpg.jpeg
│   └── ...
├── tagss/
│   ├── IMG_20260520_162128_175.jpg.jpeg
│   ├── IMG_20260520_162133_348.jpg.jpeg
│   └── ...
└── README.md
```

### Files

* **main.py** – Process and decode barcodes from a single image.
* **full_folder_output.py** – Batch process all images inside a folder and save results.
* **tagss/** – Input images containing barcode tags.
* **output/** – Generated output images with detections and decoded results.

## Usage

### Single Image

```bash
python main.py
```

### Batch Processing

```bash
python full_folder_output.py
```

