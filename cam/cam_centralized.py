import copy
import os.path
import torchvision.transforms.v2 as T
from torchvision.transforms import InterpolationMode
from cam.functions import apply_cam
from cam.utils import get_args
from dataset.datamodule_handler import get_data_modules
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad

from transforms.apply_transforms import get_finetune_transformation, get_test_transformation
from util.data_labels import get_merged_classes
from util.get_models import get_mim_model, get_cls_model
from util.utils import set_seed

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
    mim_architecture = "centralized/mim_ex56"
    img_size = 128
    batch_size = 1
    mim = get_mim_model(mim_architecture)
    phase = "centralized"
    mim.eval()
    param = {
        "wd": 1e-6,
        "lr": 3e-5,
        "beta1": 0.9,
        "beta2": 0.999,
    }
    data_modules = get_data_modules(batch_size=batch_size,
                                    classes=get_merged_classes(),
                                    train_transform=get_finetune_transformation(img_size),
                                    test_transform=get_test_transformation(img_size,
                                                                           apply_adaptation=True),
                                    threemm=True,
                                    env_path="../data/.env")
    resize = T.Resize(size=(224, 224), interpolation=InterpolationMode.BICUBIC)

    to_pil = T.Compose([T.Grayscale(1),
                        T.ToImage()])
    for data_module, client_name, classes in data_modules:
        if client_name != "DS5":
            continue
        if not os.path.exists(f"./{method}_res/{phase}/" + client_name):
            os.makedirs(f"./{method}_res/{phase}/" + client_name)
        param["step_size"] = len(data_module.train_dataloader())
        model = get_cls_model(mim_architecture, copy.deepcopy(mim.model.encoder),
                              client_name=client_name,
                              classes=classes,
                              param=param).cuda()
        # print(model.encoder[0].features[-1][-1])
        model.eval()
        model.to("cuda:1")

        # model = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
        # model.eval()
        # print(model)

        # target_layers = [model.layers[-1].blocks[-1].norm1]
        target_layers = [model.encoder[0].features[-1][-1].norm2]
        apply_cam(args, methods, model, target_layers, data_modules, client_name, phase, batch_size, resize, classes,
                  method)
