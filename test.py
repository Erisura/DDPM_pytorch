import torch
from torchvision.utils import save_image
from diffusion import *
from dataset import *
import matplotlib.pyplot as plt

model = torch.load('./trained_models/ddpm.pth',map_location='cpu')
# sample 64 images
samples = sample(model, img_size=img_size, batch_size=64, channels=channels)

# show a random one
random_index = 5
plt.imshow(samples[-1][random_index].reshape(img_size, img_size, channels), cmap="gray")