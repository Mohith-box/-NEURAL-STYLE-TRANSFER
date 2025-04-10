# -NEURAL-STYLE-TRANSFER

COMPANY: CODTECH IT SOLUTIONS

NAME: Mohith B

INTERN ID: CT06WRL

DOMAIN: Artificial intelligence

DURATION: 6 WEEKS

MENTOR: NEELA SANTOSH

TASK 3

#Objective
The goal of Task 3 was to build a Neural Style Transfer tool — a deep learning-based system that takes two images as input:

A content image (e.g., a photo of a landscape or person)

A style image (e.g., a painting by Van Gogh)

…and blends them to produce a new image that preserves the content of the first and applies the artistic style of the second.

#What Is the Tool We Created?

We developed an AI-based style transfer system using a pre-trained convolutional neural network (CNN) — VGG19, available via PyTorch.

🔹 Functionality:
Takes a content image and a style image as input.

Uses a CNN to extract content and style features from the images.

Applies the style features to the content structure using optimization.

Saves the output as a stylized image.

🔹 Technology Used:
Neural Style Transfer algorithm based on the original paper by Gatys et al.

VGG19 model for feature extraction (pre-trained on ImageNet).

Gradient descent optimization to blend style and content.

🔹 Key Features:
High-quality artistic outputs.

Fully customizable (style weight, content weight, iterations).

Runs on both CPU and GPU.

#Platform & Technology Stack


Component	Details

Language:	Python

Libraries:	torch, torchvision, PIL

Model Used:	VGG19 (pretrained on ImageNet, via PyTorch)

Execution:	Local Python script (no internet needed once model is cached)

Runtime:	CPU or GPU (automatically detected)

Development: Python Script (style_transfer.py)

#How the Tool Works (Behind the Scenes)
Image Loading & Preprocessing:

Both the content and style images are resized, normalized, and converted to tensors.

Feature Extraction:

The tool extracts content features and style features using the VGG19 model.

Content is taken from deeper layers; style is computed using Gram matrices across multiple shallower layers.

Optimization:

A copy of the content image is treated as the "target".

Gradient descent is applied to update the target image, minimizing the content and style loss over several iterations.

Result Saving:

The final styled image is de-normalized and saved as output_styled_image.jpg.

#Why This Is Valuable

Neural Style Transfer combines the power of computer vision and creativity, allowing machines to generate visually stunning artwork by understanding the structure and texture of images.

This task not only teaches deep learning concepts like CNNs and feature maps, but also provides a hands-on project in AI-assisted design — an area that's increasingly relevant in media, design, and entertainment industries.

OUTPUT:

![Image](https://github.com/user-attachments/assets/29242134-adfd-4140-97b9-10ab252d1028)
