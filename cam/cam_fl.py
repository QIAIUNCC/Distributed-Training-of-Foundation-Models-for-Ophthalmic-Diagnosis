import copy
import glob
import os.path
import torch
import torchvision.transforms.v2 as T
from torchvision.transforms import InterpolationMode
from cam.functions import apply_cam
from cam.utils import reshape_transform, get_args
from dataset.datamodule_handler import get_data_modules
from models.cls import MLP, ClsMIM
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad

from models.mim import MIMWrapper
from transforms.apply_transforms import get_finetune_transformation, get_test_transformation
from util.data_labels import get_merged_classes
from util.utils import set_seed, find_key_by_value

if __name__ == '__main__':
    """ python swinT_example.py -image-path <path_to_image>
    Example usage of using cam-methods on a SwinTransformers network.

    """
    method = "eigencam"
    args = get_args(method)
    methods = \
        {
            "gradcam": GradCAM,
            "scorecam": ScoreCAM,
            "gradcam++": GradCAMPlusPlus,
            "ablationcam": AblationCAM,
            "xgradcam": XGradCAM,
            "eigencam": EigenCAM,
            "eigengradcam": EigenGradCAM,
            "layercam": LayerCAM,
            "fullgrad": FullGrad
        }

    if args.method not in list(methods.keys()):
        raise Exception(f"method should be one of {list(methods.keys())}")
    set_seed(42)
    img_size = 128
    batch_size = 1
    param = {
        "wd": 1e-6,
        "lr": 3e-5,
        "beta1": 0.9,
        "beta2": 0.999,
    }
    data_modules = get_data_modules(batch_size=batch_size,
                                    classes=get_merged_classes(),
                                    train_transform=get_finetune_transformation(img_size, apply_adaptation=True),
                                    test_transform=get_test_transformation(img_size,
                                                                           apply_adaptation=True),
                                    threemm=True,
                                    env_path="../data/.env")
    resize = T.Resize(size=(224, 224), interpolation=InterpolationMode.BICUBIC)

    to_pil = T.Compose([T.Grayscale(1),
                        T.ToImage()])
    architecture = "fl_ex15_5_100_0.5"

    phase = "fl"
    for data_module, client_name, classes in data_modules:
        if client_name != "DS5":
            continue
        if not os.path.exists(f"./{method}_res/{phase}/" + client_name):
            os.makedirs(f"./{method}_res/{phase}/" + client_name)
        param["step_size"] = len(data_module.train_dataloader())
        print(os.path.join("../centralized", "checkpoints", architecture, client_name,
                                            "version_0", "checkpoints", "*.ckpt"))
        model_path = glob.glob(os.path.join("../centralized", "checkpoints", architecture, client_name,
                                            "version_0", "checkpoints", "*.ckpt"))

        mim = MIMWrapper(
            lr=1.5e-3,
            wd=0.05,
            min_lr=1e-5,
            patience=10,
            epochs=100,
            warmup_lr=1e-5,
            warmup_epochs=10,
            gamma=0.1,
            device="cuda:1",
            weights=False
        )

        state_dict = torch.load('../fl/ex15_f_5.pth')
        # Adjust the keys by removing the 'model.' prefix
        adjusted_state_dict = {'model.' + key: value for key, value in state_dict.items()}
        mim.load_state_dict(adjusted_state_dict)

        model = ClsMIM.load_from_checkpoint(model_path[0],
                                            encoder=copy.deepcopy(mim.model.encoder),
                                            wd=param["wd"],
                                            lr=param["lr"],
                                            beta1=param["beta1"],
                                            beta2=param["beta2"],
                                            step_size=param["step_size"],
                                            gamma=0.5,
                                            classes=classes,
                                            feature_dim=1000
                                            )

        model.eval()
        model.to("cuda:1")

        # model = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
        # model.eval()
        # print(model)

        # target_layers = [model.layers[-1].blocks[-1].norm1]
        target_layers = [model.encoder[0].features[-1][-1].norm2]
        apply_cam(args, methods, model, target_layers, data_modules, client_name, phase, batch_size, resize, classes,
                  method)
