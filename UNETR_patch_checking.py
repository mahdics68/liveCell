import torch
from UNETR_model import UNETR

shape = (256, 256)
image = torch.rand(*((1, 1) + shape))
patch_size = 8

model = UNETR(in_channels=1, out_channels=1, img_size=shape, spatial_dims=len(shape), patch_size=patch_size)
output = model(image)

print(output.shape)
