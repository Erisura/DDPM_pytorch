from diffusion import pixel_loss
from pathlib import Path
import torch
from torch.optim import Adam
from torch.utils import data
from torchvision.utils import save_image
import models
import dataset
from tqdm import tqdm
from parameters import *
from diffusion import *
from dataset import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 10

def num_to_groups(num, divisor):
    groups = num // divisor
    reminder = num % divisor
    arr = [divisor] * groups
    if reminder:
        arr.apend(reminder)
    
    return arr

result_folder = Path('./result_imgs')
result_folder.mkdir(exist_ok=True)

save_and_sample_every = 1000

model = models.UNet(
    dim=img_size,
    channels=channels,
    dim_mults=(1,2,4,),
)

model.to(device)

optimizer = Adam(model.parameters(), lr=1e-3)

for epoch in tqdm(range(epochs)):
    for step, batch in tqdm(enumerate(dataset.train_loader)):
        optimizer.zero_grad()

        # batch里面有标签和label，batch[0]是数据
        batch = batch[0].to(device)
        batch_size = len(batch)

        # 随机选择噪声
        t = torch.randint(0, timesteps, (batch_size, ), device=device).long()

        loss = pixel_loss(model, batch, t, loss_type='huber')

        if step%100 == 0:
            print(f'at epoch: {epoch}, loss: {loss}')

        loss.backward()
        optimizer.step()

        if (step+1)% save_and_sample_every == 0:
            # 每隔一段时间sample一次，存下来
            milestone = (step+1) // save_and_sample_every
            batches = num_to_groups(4, batch_size)
            all_images_list = list(map(lambda n: sample(model, batch_size=n, channels=channels, img_size=img_size), batches)) # 原文中这里没指定img_size，感觉会有问题
            all_images = torch.cat(all_images_list, dim=0)
            all_images = (all_images + 1)/2 # value scope from [-1, 1] to [0, 1]
            save_image(all_images, str(result_folder / f'sample-{milestone}.png'),nrow=6)


torch.save(model,'./trained_models/ddpm.pth')

        

