"""
datamodule.py: data module object to load data
__author: Sina Gholami
__update: add comments
__update_date: 7/5/2024
"""
import copy
import os.path

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader
from dataset.OCT_dataset import OCTDataset, get_kermany_imgs, get_srinivasan_imgs, get_oct500_imgs, get_nur_dataset, \
    get_waterloo_dataset, get_class, get_UIC_DR_imgs, get_Mario_imgs, get_WF_imgs, get_OIMHS_imgs, get_OLIVE_imgs, get_THOCT_imgs, OCTSeqDataset


class KermanyDataModule(pl.LightningDataModule):
    """
    Kermany dataset
    """

    def __init__(self, dataset_name: str, data_dir: str, batch_size: int, classes: dict, split=None,
                 train_transform=None,
                 test_transform=None):
        """
        :param dataset_name: str, the name of the dataset
        :param data_dir: str
        :param batch_size: int
        :param classes: dictionary
        :param split: (flot, flot, flot), percentage of training, validation, testing, the sum should be 1
        :param train_transform: transforms
        :param test_transform: transforms
        """
        super().__init__()
        if split is None:
            split = [1]
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.split = split
        self.classes = classes

    def prepare_data(self):
        # img_paths is a list of lists
        # get_kermany returns 3 list of tuples: 1- training 2- validation 3- testing
        self.img_paths = get_kermany_imgs(data_dir=self.data_dir, split=self.split, classes=self.classes)

    def setup(self, stage: str) -> None:
        # Assign Train for use in Dataloaders
        if stage == "train":
            self.data_train = OCTDataset(transform=self.train_transform, img_paths=self.img_paths[0])
            print("Kermany train data len:", len(self.data_train))
        # Assign val split(s) for use in Dataloaders
        elif stage == "val":
            self.data_val = OCTDataset(transform=self.test_transform, img_paths=self.img_paths[1])
            print("Kermany val data len:", len(self.data_val))

        # Assign Test split(s) for use in Dataloaders
        if stage == "test":
            self.data_test = OCTDataset(transform=self.test_transform, img_paths=self.img_paths[2])
            print("Kermany test data len:", len(self.data_test))

    def train_dataloader(self, shuffle: bool = True, drop_last: bool = True, pin_memory: bool = True,
                         workers: int = torch.cuda.device_count() * 2):
        """
        :param num_workers: int, number of workers for training loder training
        """
        return DataLoader(self.data_train,
                          batch_size=self.batch_size,
                          shuffle=shuffle,
                          pin_memory=pin_memory,
                          drop_last=drop_last,
                          num_workers=workers)

    def val_dataloader(self, shuffle: bool = False, drop_last: bool = False, pin_memory: bool = True,
                       workers: int = torch.cuda.device_count() * 2):
        return DataLoader(self.data_val,
                          batch_size=self.batch_size,
                          shuffle=shuffle,
                          drop_last=drop_last,
                          num_workers=workers,
                          pin_memory=pin_memory)

    def test_dataloader(self, shuffle: bool = False, drop_last: bool = False, pin_memory: bool = True,
                        workers: int = torch.cuda.device_count() * 2):
        return DataLoader(self.data_test,
                          batch_size=self.batch_size,
                          shuffle=shuffle,
                          drop_last=drop_last,
                          num_workers=workers,
                          pin_memory=pin_memory)

    def filtered_test_dataloader(self, source_classes):
        """
        :param source_classes: dictionary,
        """
        t_classes = copy.deepcopy(source_classes)
        if "DRUSEN" in t_classes and "AMD" in self.classes:
            t_classes["AMD"] = copy.deepcopy(self.classes["AMD"])
            del t_classes["DRUSEN"]
        elif "AMD" in t_classes and "DRUSEN" in self.classes:
            t_classes["DRUSEN"] = copy.deepcopy(self.classes["DRUSEN"])
            del t_classes["AMD"]
        # Using set intersection
        self.filtered_classes = {key: t_classes[key] for key in t_classes.keys() & self.classes.keys()}

        filtered_elements = [(item[0], get_class(item[1][1], t_classes)) for
                             item in self.data_test.img_paths if item[1][1] in t_classes.keys()]
        new_data_test = OCTDataset(transform=self.test_transform, data_dir=self.data_dir,
                                   img_paths=filtered_elements)

        return DataLoader(new_data_test, batch_size=self.batch_size, shuffle=False, drop_last=False,
                          num_workers=torch.cuda.device_count() * 2, pin_memory=True)


class SrinivasanDataModule(KermanyDataModule):
    def __init__(self, dataset_name: str, data_dir: str, batch_size: int, classes: dict, split=None,
                 train_transform=None,
                 test_transform=None, num_workers=10):
        """
        :param dataset_name: str
        :param data_dir: str
        :param batch_size: int
        :param classes: dictionary
        :param split: (flot, flot, flot), percentage of training, validation, testing, the sum should be 1
        :param train_transform: transforms
        :param test_transform: transforms

        """
        super().__init__(dataset_name, data_dir, batch_size, classes, split, train_transform,
                         test_transform)
        self.files = [i for i in range(1, 16)]  # there are 15 subjects for each category

    def prepare_data(self):
        train_idx = int(len(self.files) * self.split[0])
        val_dix = int(len(self.files) * self.split[1])
        # set the folders for training, validation, and testing
        train_subj = self.files[:train_idx]
        val_subj = self.files[train_idx:train_idx + val_dix]
        test_subj = self.files[train_idx + val_dix:]
        # img_paths is a list of lists
        self.train_imgs = get_srinivasan_imgs(data_dir=self.data_dir, ignore_folders=val_subj + test_subj,
                                              classes=self.classes, mode="train")
        self.val_imgs = get_srinivasan_imgs(data_dir=self.data_dir, ignore_folders=test_subj + train_subj,
                                            classes=self.classes, mode="val")
        self.test_imgs = get_srinivasan_imgs(data_dir=self.data_dir, ignore_folders=val_subj + train_subj,
                                             classes=self.classes, mode="test")

    def setup(self, stage: str) -> None:
        # Assign Train for use in Dataloaders
        if stage == "train":
            self.data_train = OCTDataset(transform=self.train_transform, img_paths=self.train_imgs)
            print("Srinivasan train data len:", len(self.data_train))
        # Assign val split(s) for use in Dataloaders
        elif stage == "val":
            self.data_val = OCTDataset(transform=self.test_transform, img_paths=self.val_imgs)
            print("Srinivasan val data len:", len(self.data_val))

        # Assign Test split(s) for use in Dataloaders
        if stage == "test":
            self.data_test = OCTDataset(transform=self.test_transform, img_paths=self.test_imgs)
            print("Srinivasan test data len:", len(self.data_test))


class OCT500DataModule(SrinivasanDataModule):
    def __init__(self, dataset_name: str, data_dir: str, batch_size: int, classes: dict, split=None,
                 train_transform=None,
                 test_transform=None, filter_img: bool = True, merge: dict = None,
                 threemm: bool = True):
        """
                :param data_dir: str
                :param batch_size: int
                :param classes: dictionary
                :param split: (flot, flot, flot), percentage of training, validation, testing, the sum should be 1
                :param train_transform: transforms
                :param test_transform: transforms
                :param filter_img: bool, if True only foveal images will be loaded (OCT slices taken from fovea)
                :param merge: dict: to merge n classes into another.
                    exp: {"AMD": ["CNV"], "OTHERS": ["RVO", "CSC"]} --> merging CNV into AMD, or RVO and CSC into OTHERS
                :param threemm: whether to add 3-mm images to the datamodule
                """
        super().__init__(dataset_name, data_dir, batch_size, classes, split, train_transform,
                         test_transform)
        self.filter_img = filter_img
        self.merge = merge
        self.threemm = threemm

    def prepare_data(self):
        #load 6-mm training data
        self.train_imgs = get_oct500_imgs(self.data_dir + "/OCTA_6mm", classes=self.classes, split=self.split,
                                          mode="train", filter_img=self.filter_img, merge=self.merge)
        #load 6-mm validation data

        self.val_imgs = get_oct500_imgs(self.data_dir + "/OCTA_6mm", classes=self.classes, split=self.split,
                                        mode="val", filter_img=self.filter_img, merge=self.merge)
        #load 6-mm testing data
        self.test_imgs = get_oct500_imgs(self.data_dir + "/OCTA_6mm", classes=self.classes, split=self.split,
                                         mode="test", filter_img=self.filter_img, merge=self.merge)

        if self.threemm:
            # append 3-mm training data
            self.train_imgs += get_oct500_imgs(self.data_dir + "/OCTA_3mm", classes=self.classes, split=self.split,
                                               mode="train", filter_img=self.filter_img, merge=self.merge)
            # append 3-mm validation data
            self.val_imgs += get_oct500_imgs(self.data_dir + "/OCTA_3mm", classes=self.classes, split=self.split,
                                             mode="val", filter_img=self.filter_img, merge=self.merge)
            # append 3-mm testing data
            self.test_imgs += get_oct500_imgs(self.data_dir + "/OCTA_3mm", classes=self.classes, split=self.split,
                                              mode="test", filter_img=self.filter_img, merge=self.merge)

    def setup(self, stage: str) -> None:
        # Assign Train for use in Dataloaders
        if stage == "train":
            self.data_train = OCTDataset(transform=self.train_transform, img_paths=self.train_imgs)
            print("OCT500 train data len:", len(self.data_train))
        # Assign val split(s) for use in Dataloaders
        elif stage == "val":
            self.data_val = OCTDataset(transform=self.test_transform, img_paths=self.val_imgs)
            print("OCT500 val data len:", len(self.data_val))

        # Assign Test split(s) for use in Dataloaders
        if stage == "test":
            self.data_test = OCTDataset(transform=self.test_transform, img_paths=self.test_imgs)
            print("OCT500 test data len:", len(self.data_test))


class NurDataModule(KermanyDataModule):
    def __init__(self, dataset_name: str, data_dir: str, batch_size: int, classes: dict, split=None,
                 train_transform=None,
                 test_transform=None):
        super().__init__(dataset_name, data_dir, batch_size, classes, split, train_transform, test_transform)

    def prepare_data(self):
        # img_paths is a list of three lists corresponds to training, validation and testing
        self.img_paths = get_nur_dataset(data_dir=self.data_dir,
                                         csv_filename="data_information.csv",
                                         split=self.split,
                                         classes=self.classes)

    def setup(self, stage: str) -> None:
        # Assign Train for use in Dataloaders
        if stage == "train":
            self.data_train = OCTDataset(transform=self.train_transform, img_paths=self.img_paths[0])
            print("Nur train data len:", len(self.data_train))
        # Assign val split(s) for use in Dataloaders
        elif stage == "val":
            self.data_val = OCTDataset(transform=self.test_transform, img_paths=self.img_paths[1])
            print("Nur val data len:", len(self.data_val))

        # Assign Test split(s) for use in Dataloaders
        if stage == "test":
            self.data_test = OCTDataset(transform=self.test_transform, img_paths=self.img_paths[2])
            print("Nur test data len:", len(self.data_test))


class WaterlooDataModule(KermanyDataModule):
    def __init__(self, dataset_name: str, data_dir: str, batch_size: int, classes: dict, split=None,
                 train_transform=None,
                 test_transform=None):
        super().__init__(dataset_name, data_dir, batch_size, classes, split, train_transform,
                         test_transform)

    def prepare_data(self):
        # img_paths is a list of three lists corresponds to training, validation and testing
        self.img_paths = get_waterloo_dataset(data_dir=self.data_dir,
                                              classes=self.classes,
                                              split=self.split)

    def setup(self, stage: str) -> None:
        # Assign Train for use in Dataloaders
        if stage == "train":
            self.data_train = OCTDataset(transform=self.train_transform, img_paths=self.img_paths[0])
            print("Waterloo train data len:", len(self.data_train))
        # Assign val split(s) for use in Dataloaders
        elif stage == "val":
            self.data_val = OCTDataset(transform=self.test_transform, img_paths=self.img_paths[1])
            print("Waterloo val data len:", len(self.data_val))

        # Assign Test split(s) for use in Dataloaders
        if stage == "test":
            self.data_test = OCTDataset(transform=self.test_transform, img_paths=self.img_paths[2])
            print("Waterloo test data len:", len(self.data_test))


class OCTDLDataModule(KermanyDataModule):
    def __init__(self, dataset_name: str, data_dir: str, batch_size: int, classes: dict, split=None,
                 train_transform=None,
                 test_transform=None):
        super().__init__(dataset_name, data_dir, batch_size, classes, split, train_transform,
                         test_transform)

    def prepare_data(self):
        # img_paths is a list of three lists corresponds to training, validation and testing
        self.img_paths = get_kermany_imgs(data_dir=self.data_dir,
                                          classes=self.classes,
                                          split=self.split,
                                          tokenizer="_")

    def setup(self, stage: str) -> None:
        # Assign Train for use in Dataloaders
        if stage == "train":
            self.data_train = OCTDataset(transform=self.train_transform, img_paths=self.img_paths[0])
            print("OCTDL train data len:", len(self.data_train))
        # Assign val split(s) for use in Dataloaders
        elif stage == "val":
            self.data_val = OCTDataset(transform=self.test_transform, img_paths=self.img_paths[1])
            print("OCTDL val data len:", len(self.data_val))

        # Assign Test split(s) for use in Dataloaders
        if stage == "test":
            self.data_test = OCTDataset(transform=self.test_transform, img_paths=self.img_paths[2])
            print("OCTDL test data len:", len(self.data_test))


class UICDRDataModule(OCTDLDataModule):
    def __init__(self, dataset_name: str, data_dir: str, batch_size: int, classes: dict, split=None,
                 train_transform=None, test_transform=None, fov=None):
        super().__init__(dataset_name, data_dir, batch_size, classes, split, train_transform,
                         test_transform)
        self.fov = fov

    def prepare_data(self):
        # img_paths is a list of lists
        self.img_paths = get_UIC_DR_imgs(data_dir=self.data_dir, classes=self.classes, fov=self.fov)

    def setup(self, stage: str) -> None:
        # Assign Train for use in Dataloaders
        if stage == "train":
            self.data_train = OCTDataset(transform=self.train_transform, img_paths=self.img_paths[0])
            print("UIC-DR train data len:", len(self.data_train))
        # Assign val split(s) for use in Dataloaders
        elif stage == "val":
            self.data_val = OCTDataset(transform=self.test_transform, img_paths=self.img_paths[1])
            print("UIC-DR val data len:", len(self.data_val))

        # Assign Test split(s) for use in Dataloaders
        if stage == "test":
            self.data_test = OCTDataset(transform=self.test_transform, img_paths=self.img_paths[2])
            print("UIC-DR test data len:", len(self.data_test))
            
