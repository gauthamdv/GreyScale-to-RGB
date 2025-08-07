# GrayScale to RGB Colorization

Turn your grayscale images into vibrant color outputs using deep learning.

## Project Overview
This repository provides a pipeline to convert grayscale images to color images using deep learning techniques. It includes everything needed to train, evaluate, and deploy your image colorization model.

## Repository Structure
```
├── Codes/             ← Model training and evaluation scripts
├── Models/            ← Stored trained models
├── Datasets/          ← Input grayscale and reference color images
├── Packages.txt       ← Required Python packages and versions
└── README.md          ← Project overview (you’re here!)
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/gauthamdv/GreyScale_to_RGB.git
   cd GreyScale_to_RGB
   ```

2. Create a new Python virtual environment (optional):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r Packages.txt
   ```

## Getting Started
- Place your image datasets inside the `Datasets/` directory.
- Run the training and evaluation scripts inside `Codes/` to start the image colorization process.
- Trained models will be saved to the `Models/` folder.

## Features
- End-to-end setup for grayscale-to-color image conversion.
- Modular structure with separate folders for codes, datasets, and outputs.
- Easy environment setup with `Packages.txt`.

## Future Enhancements
- Include details on preprocessing steps (scaling, normalization).
- Provide example usage or sample results (e.g., before-and-after images).
- Add usage instructions for inference using trained models.
- Integrate visual results or interactive Demo via GitHub Pages or Jupyter notebooks.
