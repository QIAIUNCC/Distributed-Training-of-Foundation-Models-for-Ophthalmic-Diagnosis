import torchvision
from torchvision.models import Swin_V2_B_Weights, Swin_V2_T_Weights
import torch

model = torchvision.models.swin_v2_t(weights=Swin_V2_T_Weights.DEFAULT, progress=True)
print(model)
x = torch.rand((1, 3, 256, 256))
out = model.features[0](x)
print(out.shape)