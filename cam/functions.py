import os
import matplotlib.pyplot as plt
import numpy as np
import torch

from cam.lib.ablation_layer import AblationLayerVit
from cam.lib.utils.image import show_cam_on_image
from cam.utils import reshape_transform
from util.utils import find_key_by_value


def apply_cam(args, methods, model, target_layers, data_modules, client_name, phase, batch_size, resize, classes, method):
    if args.method not in methods:
        raise Exception(f"Method {args.method} not implemented")

    if args.method == "ablationcam":
        cam = methods[args.method](model=model,
                                   target_layers=target_layers,
                                   reshape_transform=reshape_transform,
                                   ablation_layer=AblationLayerVit())
    else:
        cam = methods[args.method](model=model,
                                   target_layers=target_layers,
                                   reshape_transform=reshape_transform)

    # Initialize Grad-CAM
    # grad_cam = GradCAMM(model, target_layers)
    for module, c_name, cc in data_modules:
        if c_name != client_name:
            continue

        for cat in cc:
            if not os.path.exists(f"./{method}_res/{phase}/" + client_name + "/" + cat):
                os.makedirs(f"./{method}_res/{phase}/" + client_name + "/" + cat)

        test_loader = module.filtered_test_dataloader(classes)

        # AblationCAM and ScoreCAM have batched implementations.
        # You can override the internal batch size for faster computation.
        cam.batch_size = batch_size

        j = 1
        counter = {}
        for data in test_loader:
            images = data[0]
            label = data[1]
            if label in counter:
                if counter[label] > 200:
                    continue
                counter[label] += 1
            else:
                counter[label] = 1
            with torch.no_grad():
                preds = model(images.to("cuda:1")).argmax(dim=-1)
                prediction = preds.cpu().item() == label
                # print(preds, label, prediction)

            grayscale_cam = cam(input_tensor=images,
                                targets=None,
                                eigen_smooth=args.eigen_smooth,
                                aug_smooth=args.aug_smooth)

            # Here grayscale_cam has only one image in the batch
            grayscale_cam = grayscale_cam[0, :]
            cam_image = show_cam_on_image(np.float32((images.squeeze(0).permute(1, 2, 0).numpy() + 1) / 2),
                                          grayscale_cam,
                                          image_weight=0.7,
                                          use_rgb=True)
            plt.imshow(resize(cam_image), "gray")  # Make sure the image is in RGB format if it's a color image
            plt.axis('off')  # Hide the axis
            actual_category = find_key_by_value(cc, label.item())
            predicted = find_key_by_value(classes, preds.item()) + "_"

            plt.savefig(os.path.join(f"./{method}_res", phase,
                                     client_name, actual_category, str(prediction.item())
                                     + "_" + predicted + str(j) + ".png"),
                        bbox_inches='tight', pad_inches=0)
            plt.close()
            print(f"image {j} has been proceed!")
            j += 1