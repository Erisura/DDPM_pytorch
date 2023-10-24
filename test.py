import torch
from torchvision.utils import save_image
from diffusion import *

model = torch.load('./trained_models/ddpm.pth')
imgs = sample(model, 28,28, batch_size=4, channels=1)
imgs = (imgs + 1)/2
save_image(imgs)
save_image(imgs, str('result_imgs' + f'samples.png'),nrow=2)

