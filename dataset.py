import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Lambda(lambda t: (t*2)-1),
])

batch_size = 128
channels = 1
img_size = 28

dataset = torchvision.datasets.FashionMNIST(
    root='./data',
    train=True,
    transform=transforms,
    download=True
)

train_loader = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=True
)



