import torch 
from UNETR_model import UNETR

image = torch.rand(1,1, 512,512)
image.shape


model = UNETR(in_channels= 1, out_channels=1, img_size= (512,512), spatial_dims=2, patch_size = 8)
output = model(image)
