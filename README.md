# BLIP Image Captioning

This repository contains code for running and evaluating a BLIP (Bootstrapping Language-Image Pre-training) model for image captioning. BLIP is a powerful vision-language model that can generate natural language descriptions of images.

## Overview

BLIP is a state-of-the-art vision-language model that excels in image captioning, visual question answering, and other multimodal tasks. This implementation focuses on the image captioning capabilities of BLIP.

## Requirements

```
torch
transformers
pillow
matplotlib
tqdm
nltk
numpy
```

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install torch transformers pillow matplotlib tqdm nltk numpy
   ```
3. Download the COCO dataset (optional for evaluation):
   ```
   # Download COCO 2017 Train/Val images and annotations
   wget http://images.cocodataset.org/zips/train2017.zip
   wget http://images.cocodataset.org/zips/val2017.zip
   wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
   
   # Extract files
   unzip train2017.zip -d coco_data/
   unzip val2017.zip -d coco_data/
   unzip annotations_trainval2017.zip -d coco_data/
   ```

## Usage

### Single Image Captioning

```python
import torch
import matplotlib.pyplot as plt
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

def generate_caption(image_path, processor, model, device):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    encoding = processor(images=image, return_tensors="pt").to(device)
    pixel_values = encoding["pixel_values"]

    with torch.no_grad():
        outputs = model.generate(pixel_values=pixel_values, max_length=50)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.axis("off")
    plt.title(caption)
    plt.show()

    return caption

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load your custom weights if needed
model.load_state_dict(torch.load("model.pth", map_location=device))
model = model.to(device)

# Test on an image
sample_image_path = "path/to/your/image.jpg"
caption = generate_caption(sample_image_path, processor, model, device)
print("Generated Caption:", caption)
```

### Evaluation on COCO Dataset

```python
import torch
import numpy as np
from PIL import Image
import json
import os
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration
from nltk.translate.bleu_score import corpus_bleu

# Load model and processor as above...

def evaluate_on_coco(model, processor, annotation_file, image_dir, num_samples=100):
    # Load and process images from COCO dataset
    # Calculate BLEU scores
    # See full implementation in evaluation.py
    
    return results

annotation_file = "coco_data/annotations/captions_train2017.json"
image_dir = "coco_data/train2017"

results = evaluate_on_coco(model, processor, annotation_file, image_dir, num_samples=50)
print("Evaluation Results:", results)
```

## Model Details

The BLIP model used in this repository is pretrained on large-scale image-text pairs. It uses a Vision Transformer (ViT) as the image encoder and a text decoder based on BERT architecture. The model can be fine-tuned on specific datasets for improved performance.

The `model.pth` file contains the weights of a fine-tuned model. To use the default pretrained weights, omit the `load_state_dict` line.

## Evaluation Metrics

- **BLEU-1**: Measures unigram precision (how many individual words match between generated captions and reference captions)
- Other metrics that can be implemented: METEOR, CIDEr, SPICE, ROUGE-L

## Acknowledgments

- Salesforce Research for the BLIP model
- COCO dataset for evaluation
- Hugging Face Transformers library for model implementation
