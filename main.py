from PIL import Image
import requests
from torchvision.transforms import ToPILImage, ToTensor, Compose, CenterCrop, Resize, Lambda
import numpy as np

url = 'http://images.cocodataset.org/test-stuff2017/000000013007.jpg'
img = Image.open(requests.get(url, stream=True).raw)
img_size = 128

# img -> tensor -> normalization
transform = Compose([
    Resize(img_size),
    CenterCrop(img_size),
    ToTensor(), # turn into torch Tensor of shape CHW, divide by 255
    Lambda(lambda t:(t*2)-1), # turn value scope from [0,1] ->[-1,1]
])
x_start = transform(img).unsqueeze(0)

reverse_transform = Compose([
    Lambda(lambda t:(t+1)/2),
    Lambda(lambda t: t.permute(1,2,0)),
    Lambda(lambda t: t*255.),
    Lambda(lambda t: t.numpy().astype(np.uint8)),
    ToPILImage(),
])

