import PIL.Image as Image
import numpy as np
from matplotlib import pyplot as plt
from model import CSRNet
import torch
from torchvision import transforms
transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ])
model = CSRNet()
model = model.cpu()
checkpoint = torch.load('0model_best.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
from matplotlib import cm as c
img = transform(Image.open('dataset/part_A_final/train_data/images/IMG_302.jpg').convert('RGB')).cpu()

output = model(img.unsqueeze(0))
print("Predicted Count : ",int(output.detach().cpu().sum().numpy()))
temp = np.asarray(output.detach().cpu().reshape(output.detach().cpu().shape[2],output.detach().cpu().shape[3]))
plt.imshow(temp,cmap = c.jet)
plt.show()