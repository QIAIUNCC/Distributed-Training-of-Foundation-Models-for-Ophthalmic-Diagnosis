import glob
import os
import torchvision
from torchvision.models import ResNet18_Weights, ResNet50_Weights, Swin_V2_B_Weights, Swin_V2_T_Weights, Swin_V2_S_Weights, ResNeXt101_64X4D_Weights, ViT_L_32_Weights, ConvNeXt_Large_Weights
from models.cls import ClsMIM
from models.mim import MIMWrapper


def get_mim_model(comment, root="../centralized", epochs=100, devices=[1]):
    mim_path = glob.glob(os.path.join(root, "checkpoints",comment, "*.ckpt"))
    print(os.path.join(root, "checkpoints", "archive",comment, "*.ckpt"))
    mim = MIMWrapper.load_from_checkpoint(mim_path[0],
                                            lr=1.5e-3,
                                            wd=0.05,
                                            min_lr=1e-5,
                                            patience=10,
                                            epochs=epochs,
                                            warmup_lr=1e-5,
                                            warmup_epochs=10,
                                            gamma=0.1,
                                            device="cuda:" + str(devices[0]),
                                            weights=False
                                            )
    return mim


def get_cls_model(comment, encoder, classes, client_name, param, root="../centralized"):
    print(os.path.join(root, "checkpoints", comment, client_name, "version_0", "checkpoints", "*.ckpt"))
    cls_path = glob.glob(
        os.path.join(root, "checkpoints", comment, client_name, "version_0", "checkpoints", "*.ckpt"))
    cls = ClsMIM.load_from_checkpoint(cls_path[0],
                                      encoder=encoder,
                                      wd=param["wd"],
                                      lr=param["lr"],
                                      beta1=param["beta1"],
                                      beta2=param["beta2"],
                                      step_size=param["step_size"],
                                      gamma=0.5,
                                      classes=classes,
                                      feature_dim=1000
                                      )
    return cls


def get_baseline_model(model_architecture, pretrained=False):
    if model_architecture == "resnet18":
        return torchvision.models.resnet18(ResNet18_Weights.IMAGENET1K_V1) if pretrained else (
            torchvision.models.resnet18())
    elif model_architecture == "resnet50":
        return torchvision.models.resnet50(ResNet50_Weights.IMAGENET1K_V1) if pretrained else (
            torchvision.models.resnet50())
    elif model_architecture == "swinv2_b":
        return torchvision.models.swin_v2_b(
            Swin_V2_B_Weights.IMAGENET1K_V1) if pretrained else torchvision.models.swin_v2_b()
    elif model_architecture == "swinv2_t":
        return torchvision.models.swin_v2_t(
            Swin_V2_T_Weights.IMAGENET1K_V1) if pretrained else torchvision.models.swin_v2_t()
    elif model_architecture == "swinv2_s":
        return torchvision.models.swin_v2_s(
            Swin_V2_S_Weights.IMAGENET1K_V1) if pretrained else torchvision.models.swin_v2_s()
    elif model_architecture == "resnext101_64x4d":
        return torchvision.models.resnext101_64x4d(ResNeXt101_64X4D_Weights.IMAGENET1K_V1) if pretrained else torchvision.models.resnext101_64x4d()
    elif model_architecture == "vit_l_32":
        return torchvision.models.vit_l_32(ViT_L_32_Weights.IMAGENET1K_V1) if pretrained else torchvision.models.vit_l_32()
    elif model_architecture == "convnext_large":
        return torchvision.models.convnext_large(ConvNeXt_Large_Weights.IMAGENET1K_V1) if pretrained else torchvision.models.convnext_large()
    

