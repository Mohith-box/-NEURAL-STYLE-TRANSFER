import warnings
warnings.filterwarnings("ignore")  # Optional: suppress deprecation warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torchvision.models import vgg19, VGG19_Weights
from PIL import Image

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device set to: {device}")

# Load image
def load_image(img_path, max_size=400):
    image = Image.open(img_path).convert('RGB')
    size = min(max(image.size), max_size)
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])
    image = transform(image).unsqueeze(0)
    return image.to(device)

# Save image
def save_image(tensor, path):
    image = tensor.clone().squeeze(0)
    image = transforms.ToPILImage()(image.cpu())
    image.save(path)

# Load content and style images
content = load_image("input_image.jpg")
style = load_image("style_image.jpg")

# Load VGG19 with updated weight syntax
weights = VGG19_Weights.DEFAULT
vgg = vgg19(weights=weights).features.to(device).eval()

# Define style and content layers
style_layers = ['0', '5', '10', '19', '28']
content_layers = ['21']

# Extract features
def get_features(image, model, layers):
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[name] = x
    return features

# Compute gram matrix for style comparison
def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    features = tensor.view(c, h * w)
    return torch.mm(features, features.t())

# Get features for both content and style
content_features = get_features(content, vgg, content_layers)
style_features = get_features(style, vgg, style_layers)

# Create target image (initially a clone of content)
target = content.clone().requires_grad_(True).to(device)

# Optimizer setup
optimizer = optim.Adam([target], lr=0.003)

# Style transfer weights
style_weight = 1e6
content_weight = 1

# Training loop
for step in range(1, 501):
    target_features = get_features(target, vgg, style_layers + content_layers)

    # Content loss
    content_loss = torch.mean((target_features['21'] - content_features['21']) ** 2)

    # Style loss
    style_loss = 0
    for layer in style_layers:
        target_gram = gram_matrix(target_features[layer])
        style_gram = gram_matrix(style_features[layer])
        style_loss += torch.mean((target_gram - style_gram) ** 2)

    # Total loss
    total_loss = style_weight * style_loss + content_weight * content_loss

    # Backpropagation
    optimizer.zero_grad()
    total_loss.backward(retain_graph=True)
    optimizer.step()

    # Logging
    if step % 100 == 0:
        print(f"Step {step} | Total Loss: {total_loss.item():.2f}")

# Save final output
save_image(target, "output_styled_image.jpg")
print("✅ Style transfer complete! Output saved as 'output_styled_image.jpg'")


