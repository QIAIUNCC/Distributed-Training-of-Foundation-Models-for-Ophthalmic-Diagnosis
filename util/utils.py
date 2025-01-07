import collections
import os
import random
from typing import List, Tuple
import numpy as np
import torch
import pytorch_lightning as pl
from flwr.common import Metrics
import random
import bisect

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    print("============Server Aggregation=============")
    print(metrics)
    counter = collections.Counter()
    total_examples = sum([example[0] for example in metrics])
    # aggregate the metrics
    for m in metrics:
        counter.update({key: m[0] * value / total_examples for key, value in m[1].items()})
    result = dict(counter)
    # Aggregate and return custom metric (weighted average)
    return result


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pl.seed_everything(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.set_float32_matmul_precision('medium')


def get_AMD_classes():
    srinivasan_classes = [("NORMAL", 0),
                          ("AMD", 1),
                          ]

    kermany_classes = [("NORMAL", 0),
                       ("AMD", 1),
                       ]

    oct500_classes = [("NORMAL", 0),
                      ("AMD", 1),
                      ]
    return kermany_classes, srinivasan_classes, oct500_classes



def count_classes_and_compute_weights(data_loader):
    class_counts = {0: 0, 1: 0}

    for batch in data_loader:
        labels = batch["label"]  # Adjust this if your label key is different
        class_counts[0] += torch.sum(labels == 0).item()
        class_counts[1] += torch.sum(labels == 1).item()

    total_counts = sum(class_counts.values())
    if total_counts == 0:
        raise ValueError("No labels found in the data loader.")

    # Calculate weights inversely proportional to class frequencies
    class_weights = {cls: total_counts / count for cls, count in class_counts.items()}

    # Convert to a tensor
    weights_tensor = torch.tensor([class_weights[0], class_weights[1]], dtype=torch.float32)
    return class_counts, weights_tensor

def find_key_by_value(my_dict, target_value):
    """
    Return the first key from the dictionary that has the specified value.

    Parameters:
    - my_dict (dict): The dictionary to search.
    - target_value: The value for which the key needs to be found.

    Returns:
    - key: The first key that matches the given value.
    - None: If no key with that value exists.
    """
    # Iterate over each key-value pair in the dictionary
    for key, value in my_dict.items():
        if value == target_value:
            return key  # Return the first key that matches the value

    return None  # Return None if no key matches


def random_choice_with_weighted_distribution(bins, cumulative_weights):
    # Randomly generate a number between 0 and the sum of weights (1 in this case)
    rnd = random.random()

    # Use bisect to find the corresponding bin based on cumulative weights
    bin_index = bisect.bisect(cumulative_weights, rnd)

    # Generate a random value within the selected bin
    selected_bin = bins[bin_index]
    selected_value = random.uniform(selected_bin[0], selected_bin[1])

    return selected_value


# Example usage