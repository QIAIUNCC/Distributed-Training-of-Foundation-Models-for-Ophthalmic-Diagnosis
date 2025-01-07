"""
datamodule.py: Handles the initialization of data modules
__author: Sina Gholami
__update: add comments
__update_date: 7/5/2024
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

from dataset.datamodule import KermanyDataModule, SrinivasanDataModule, OCT500DataModule, NurDataModule, \
    WaterlooDataModule, OCTDLDataModule, UICDRDataModule, MarioDataModule, WFDataModule, OIMHSDataModule, THOCTDataModule, OLIVEDataModule
from transforms.apply_transforms import get_test_transformation
from util.data_labels import get_full_classes
from util.utils import set_seed


def get_datamodule(dataset_name,
                   dataset_path,
                   batch_size,
                   train_transform,
                   test_transform,
                   kermany_classes=None,
                   srinivasan_classes=None,
                   oct500_classes=None,
                   nur_classes=None,
                   waterloo_classes=None,
                   octdl_classes=None,
                   uic_dr_classes=None,
                   filter_img=True,
                   merge=None,
                   DS3_3mm=True,
                   DS8_fov=None
                   ):
    """
    auxiliary function to create and return a datamodule based on the dataset name
    """
    datamodule = None
    if dataset_name == "DS1":
        datamodule = KermanyDataModule(dataset_name=dataset_name,
                                       data_dir=dataset_path,
                                       batch_size=batch_size,
                                       classes=kermany_classes,
                                       split=[0.9, 0.025, 0.075],
                                       train_transform=train_transform,
                                       test_transform=test_transform,
                                       )
        # preparing config
        datamodule.prepare_data()
        datamodule.setup("train")
        datamodule.setup("val")
        datamodule.setup("test")

    elif dataset_name == "DS2":
        datamodule = SrinivasanDataModule(dataset_name=dataset_name,
                                          data_dir=dataset_path,
                                          batch_size=batch_size,
                                          classes=srinivasan_classes,
                                        #   0.67, 0.14, 0.19
                                          split=[0.67, 0.13, 0.2],
                                          train_transform=train_transform,
                                          test_transform=test_transform,
                                          )
        # preparing config
        datamodule.prepare_data()
        datamodule.setup("train")
        datamodule.setup("val")
        datamodule.setup("test")

    elif dataset_name == "DS3":
        datamodule = OCT500DataModule(dataset_name=dataset_name,
                                      data_dir=dataset_path,
                                      batch_size=batch_size,
                                      classes=oct500_classes,
                                      split=[0.85, 0.05, 0.1],
                                      train_transform=train_transform,
                                      test_transform=test_transform,
                                      filter_img=filter_img,
                                      merge=merge,
                                      threemm=DS3_3mm
                                      )
        # preparing config
        datamodule.prepare_data()
        datamodule.setup("train")
        datamodule.setup("val")
        datamodule.setup("test")

    elif dataset_name == "DS4":
        datamodule = NurDataModule(dataset_name=dataset_name,
                                   data_dir=dataset_path,
                                   batch_size=batch_size,
                                   classes=nur_classes,
                                   split=[0.8, 0.05, 0.15],
                                   train_transform=train_transform,
                                   test_transform=test_transform,
                                   )
        # preparing config
        datamodule.prepare_data()
        datamodule.setup("train")
        datamodule.setup("val")
        datamodule.setup("test")

    elif dataset_name == "DS5":
        datamodule = WaterlooDataModule(dataset_name=dataset_name,
                                        data_dir=dataset_path,
                                        batch_size=batch_size,
                                        classes=waterloo_classes,
                                        split=[0.7, 0.1, 0.2],
                                        train_transform=train_transform,
                                        test_transform=test_transform,
                                        )
        # preparing config
        datamodule.prepare_data()
        datamodule.setup("train")
        datamodule.setup("val")
        datamodule.setup("test")

    elif dataset_name == "DS6":
        datamodule = OCTDLDataModule(dataset_name=dataset_name,
                                     data_dir=dataset_path,
                                     batch_size=batch_size,
                                     classes=octdl_classes,
                                     split=[0.7, 0.10, 0.2],
                                     train_transform=train_transform,
                                     test_transform=test_transform,
                                     )
        # preparing config
        datamodule.prepare_data()
        datamodule.setup("train")
        datamodule.setup("val")
        datamodule.setup("test")

    elif dataset_name == "DS7":
        datamodule = UICDRDataModule(dataset_name=dataset_name,
                                     data_dir=dataset_path,
                                     batch_size=batch_size,
                                     classes=uic_dr_classes,
                                     split=[0.25, 0.25, 0.5],
                                     train_transform=train_transform,
                                     test_transform=test_transform,
                                     fov=DS8_fov
                                     )
        # preparing config
        datamodule.prepare_data()
        datamodule.setup("train")
        datamodule.setup("val")
        datamodule.setup("test")
    
    return datamodule



def get_data_modules(batch_size, classes, train_transform=None, test_transform=None, filter_img=True, DS3_3mm=True, merge={}, DS8_fov=["3mm", "6mm", "8mm"],
                      env_path="./data/.env", pretraining=False):
    """
    create and return all available data modules
    returns a list of data modules
    """
    load_dotenv(dotenv_path=env_path)  #read the dataset paths from the .env file
    client_name = "DS1"
    DATASET_PATH = os.getenv(client_name + "_PATH")
    kermany_datamodule = get_datamodule(dataset_name=client_name,
                                        dataset_path=DATASET_PATH,
                                        batch_size=batch_size,
                                        kermany_classes=classes[0],
                                        train_transform=train_transform,
                                        test_transform=test_transform
                                        )

    client_name = "DS2"
    DATASET_PATH = os.getenv(client_name + "_PATH")
    srinivasan_datamodule = get_datamodule(dataset_name=client_name,
                                           dataset_path=DATASET_PATH,
                                           batch_size=batch_size,
                                           srinivasan_classes=classes[1],
                                           train_transform=train_transform,
                                           test_transform=test_transform
                                           )

    client_name = "DS3"
    DATASET_PATH = os.getenv(client_name + "_PATH")
    oct500_datamodule = get_datamodule(dataset_name=client_name,
                                       dataset_path=DATASET_PATH,
                                       batch_size=batch_size,
                                       oct500_classes=classes[2],
                                       train_transform=train_transform,
                                       test_transform=test_transform,
                                       filter_img=filter_img,
                                       merge=merge,
                                       DS3_3mm=DS3_3mm
                                       )

    client_name = "DS4"
    DATASET_PATH = os.getenv(client_name + "_PATH")
    nur_datamodule = get_datamodule(dataset_name=client_name,
                                    dataset_path=DATASET_PATH,
                                    batch_size=batch_size,
                                    nur_classes=classes[3],
                                    train_transform=train_transform,
                                    test_transform=test_transform,
                                    )
    client_name = "DS5"
    DATASET_PATH = os.getenv(client_name + "_PATH")
    waterloo_datamodule = get_datamodule(dataset_name=client_name,
                                         dataset_path=DATASET_PATH,
                                         batch_size=batch_size,
                                         waterloo_classes=classes[4],
                                         train_transform=train_transform,
                                         test_transform=test_transform,
                                         )

    client_name = "DS6"
    DATASET_PATH = os.getenv(client_name + "_PATH")
    octdl_datamodule = get_datamodule(dataset_name=client_name,
                                      dataset_path=DATASET_PATH,
                                      batch_size=batch_size,
                                      octdl_classes=classes[5],
                                      train_transform=train_transform,
                                      test_transform=test_transform,
                                      )
    client_name = "DS7"
    DATASET_PATH = os.getenv(client_name + "_PATH")
    uic_dr_datamodule = get_datamodule(dataset_name=client_name,
                                       dataset_path=DATASET_PATH,
                                       batch_size=batch_size,
                                       uic_dr_classes=classes[6],
                                       train_transform=train_transform,
                                       test_transform=test_transform,
                                       DS8_fov=DS8_fov
                                       )
    data_modules = [
        kermany_datamodule,
        srinivasan_datamodule,
        oct500_datamodule,
        nur_datamodule,
        waterloo_datamodule,
        octdl_datamodule,
        uic_dr_datamodule,
    ]
    
    return data_modules


def plot_pie_chart(data_list, set_name, dataset_name, classes):
    label_colors = {
        'AMD': (27 / 255, 161 / 255, 226 / 255),
        'CNV': (255 / 255, 148 / 255, 71 / 255),
        'CSC': (223 / 255, 90 / 255, 83 / 255),
        'CSR': (223 / 255, 90 / 255, 83 / 255),
        'DME': (106 / 255, 221 / 255, 100 / 255),
        'DR': (227 / 255, 200 / 255, 0),
        'DRUSEN': (27 / 255, 161 / 255, 226 / 255),
        'HR': (245 / 255, 71 / 255, 164 / 255),
        'NORMAL': (249 / 255, 247 / 255, 237 / 255),
        'OTHERS': (36 / 255, 214 / 255, 214 / 255),
        'RVO': (225 / 255, 213 / 255, 231 / 255),
        # Add more label-color mappings as needed
    }

    # Generate a list of colors in the order of labels
    colors = [label_colors[label] for label in classes]
    label_counts = {}
    reverse_dict = {value: key for key, value in classes.items()}
    for _, label in data_list:
        if reverse_dict[label] in label_counts:
            label_counts[reverse_dict[label]] += 1
        else:
            label_counts[reverse_dict[label]] = 1
    print(label_counts)
    # Prepare data for plotting
    labels = list(label_counts.keys())
    colors = [label_colors[label] for label in labels]
    counts = list(label_counts.values())

    # Function to format label with percentage and count
    def func(pct, allvals):
        absolute = int(pct / 100. * sum(allvals))
        return "{:.1f}%\n({:d})".format(pct, absolute)

    # Plot the data
    plt.pie(counts, labels=labels, autopct=lambda pct: func(pct, counts), colors=colors)
    plt.title(f"Percentage and Total Count of {dataset_name}'s {set_name}")
    plt.show()


def accumulate_data_counts(data_list, classes):
    label_counts = {label: 0 for label in classes.keys()}  # Initialize counts for each class
    for _, label in data_list:
        for key, value in classes.items():
            if value == label:
                label_counts[key] += 1
    return label_counts


def plot_bar_chart(train_counts, val_counts, test_counts, classes, dataset_name):
    labels = list(classes.keys())
    x = np.arange(len(labels))  # the label locations
    width = 0.3  # the width of the bars

    # Set the figure size here
    fig, ax = plt.subplots(figsize=(14, 8))  # Example: 10 inches by 6 inches
    train_bars = ax.bar(x - width, [train_counts[label] for label in labels], width, label='Train')
    val_bars = ax.bar(x, [val_counts[label] for label in labels], width, label='Validation')
    test_bars = ax.bar(x + width, [test_counts[label] for label in labels], width, label='Test')

    # Add some text for labels, title, and custom x-axis tick labels, etc.
    ax.set_ylabel('Counts')
    # ax.set_title(f'{dataset_name} Counts by dataset and class')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()

    # Function to calculate and format percentages for legend
    def calculate_percentages(counts_dict):
        total = sum(counts_dict.values())
        percentages = {label: f"{count / total * 100:.1f}%" for label, count in counts_dict.items()}
        return ", ".join([f"{label}: {percent}" for label, percent in percentages.items()])

    train_percentages = calculate_percentages(train_counts)
    val_percentages = calculate_percentages(val_counts)
    test_percentages = calculate_percentages(test_counts)

    # Customizing the legend to include percentages
    legend_labels = [
        f"Train         ({train_percentages})",
        f"Validation ({val_percentages})",
        f"Test           ({test_percentages})"
    ]

    # Manually adding legend entries
    ax.legend([train_bars, val_bars, test_bars], legend_labels)

    # Function to attach a text label above each bar, displaying its height.
    def autolabel(bars):
        """Attach a text label above each bar in *bars*, displaying its height."""
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    # Call the autolabel function for each bar chart
    autolabel(train_bars)
    autolabel(val_bars)
    autolabel(test_bars)

    plt.show()


"""
Plotting the data
"""
if __name__ == "__main__":
    set_seed(42)
    # Set the font to Times New Roman and increase default font sizes
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 13  # For general text
    plt.rcParams['axes.labelsize'] = 14  # For x and y labels
    plt.rcParams['axes.titlesize'] = 16  # For the title
    plt.rcParams['xtick.labelsize'] = 12  # For x tick labels
    plt.rcParams['ytick.labelsize'] = 12  # For y tick labels
    plt.rcParams['legend.fontsize'] = 14  # For legend

    data_modules = get_data_modules(batch_size=1,
                                    classes=get_full_classes(),
                                    train_transform=get_test_transformation(10),
                                    test_transform=get_test_transformation(10),
                                    filter_img=True,
                                    threemm=False,
                                    env_path="../data/.env")[2:]

    # Accumulate counts for each dataset
    train_counts, val_counts, test_counts = {}, {}, {}

    for data_module in data_modules:
        train_counts = accumulate_data_counts(data_module.data_train, data_module.classes)
        val_counts = accumulate_data_counts(data_module.data_val, data_module.classes)
        test_counts = accumulate_data_counts(data_module.data_test, data_module.classes)

        # Now plot the bar chart with these counts
        plot_bar_chart(train_counts, val_counts, test_counts, data_module.classes, data_module.dataset_name)

    for data_module in data_modules:
        plot_pie_chart(data_module.data_train, "Train Data", dataset_name=data_module.dataset_name,
                       classes=data_module.classes)
        plot_pie_chart(data_module.data_val, "Validation Data", dataset_name=data_module.dataset_name,
                       classes=data_module.classes)
        plot_pie_chart(data_module.data_test, "Test Data", dataset_name=data_module.dataset_name,
                       classes=data_module.classes)

"""
merged since in DS3:
Train:
    - full classes             : {'NORMAL': 12150, 'AMD': 2268, 'DR': 3078, 'CNV': 729, 'OTHERS': 4617, 'RVO': 486, 'CSC': 648}
    - full classes no merging  : {'NORMAL': 12150, 'AMD': 2268, 'DR': 3078, 'OTHERS': 4617}
    - full classes and merging : {'NORMAL': 12150, 'AMD': 3078, 'DR': 3078, 'OTHERS': 5832}
    - merged classes and merging
Val: 
    - full classes             : {'NORMAL': 3969, 'AMD': 729, 'DR': 891, 'CNV': 162, 'OTHERS': 1539, 'RVO': 81, 'CSC': 162}
    - full classes no merging  : {'NORMAL': 3969, 'AMD': 729, 'DR': 891, 'OTHERS': 1539}
    - full classes and merging : {'NORMAL': 3969, 'AMD': 972, 'DR': 891, 'OTHERS': 1863}
    - merged classes and merging {'NORMAL': 3969, 'AMD': 972, 'DR': 891, 'OTHERS': 1863}

Test: 
    - full classes             : {'NORMAL': 3888, 'AMD': 648, 'DR': 891, 'CNV': 81, 'OTHERS': 1458, 'RVO': 81, 'CSC': 162}
    - full classes no merging  : {'NORMAL': 3888, 'AMD': 648, 'DR': 891, 'OTHERS': 1458}
    - full classes and merging : {'NORMAL': 3888, 'AMD': 891, 'DR': 891, 'OTHERS': 1863}
    - merged classes and merging  'NORMAL': 3888, 'AMD': 891, 'DR': 891, 'OTHERS': 1863}
"""
