from typing import Callable, List, Union

import pytorch_lightning as pl

import torch
from torch.utils import data as data
import json
import os
import cv2
import numpy as np
import random
from PIL import Image


class RelightDataset(data.Dataset):
    def __init__(self, opt, shuffle_lines=False, shuffle_seed=0):
        super(RelightDataset, self).__init__()
        self.data_dir = opt["data_dir"]
        self.size = opt["size"]
        self.rng = random.Random()
        self.data_paths = self.get_data_paths(
            opt["json_dir_list"],
            shuffle_lines,
            shuffle_seed,
        )

    def get_data_paths(
            self,
            json_dir_list,
            shuffle_lines,
            shuffle_seed,
    ):
        data_paths = []
        for json_path in json_dir_list:
            with open(json_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            if shuffle_lines:
                self.rng.seed(shuffle_seed)
                self.rng.shuffle(raw_data)
            data_paths.extend([json_data for json_data in raw_data])
        return data_paths

    def parse_json(self, json_item):

        # origin degraded image
        degraded_names = json_item['degraded_names']
        degraded_indx = np.random.randint(0, len(degraded_names))
        degraded_name = degraded_names[degraded_indx]
        degraded_image_path = os.path.join(self.data_dir, json_item['degraded_dir'], degraded_name)
        degraded_image_pil = Image.open(degraded_image_path).convert('RGB')
        tgz_w, tgz_h = degraded_image_pil.size  #
        degraded_image = np.array(degraded_image_pil)

        # target_image
        image_path = os.path.join(self.data_dir, json_item['image_dir'], json_item['image_name'])
        target_image = np.array(Image.open(image_path).convert('RGB'))
        h, w, _ = target_image.shape
        if tgz_w != w or tgz_h != h:
            target_image = cv2.resize(target_image, (tgz_w, tgz_h), interpolation=cv2.INTER_LINEAR)
            h, w, _ = target_image.shape   
        
        # new degraded image
        alpha_path = os.path.join(self.data_dir, json_item['alpha_dir'], json_item['alpha_name'])
        alpha = np.array(Image.open(alpha_path).convert('L'))
        alpha_h, alpha_w = alpha.shape[:2]
        if tgz_w != alpha_w or tgz_h != alpha_h:
            alpha = cv2.resize(alpha, (tgz_w, tgz_h), interpolation=cv2.INTER_NEAREST)
        alpha = alpha[:, :, None]
        alpha[alpha <= 2] = 0.0
        alpha[alpha > 2] = 1.0
        degraded_image = degraded_image * alpha + (1 - alpha) * target_image

        if 'instruction' in json_item:
            prompt = json_item['instruction']
        else:
            prompt = 'Change the lighting to studio lighting and retain the details of the face.'

        return dict(
            degraded_image=degraded_image,
            target_image=target_image,
            mask=alpha,
            prompt=prompt,
        )

    def data_preprocess(self, data):

        degraded_image = np.array(data["degraded_image"])[:, :, :3] / 255.
        target_image = np.array(data["target_image"])[:, :, :3] / 255.
        degraded_image = torch.from_numpy(degraded_image.transpose(2, 0, 1)).to(dtype=torch.float32)
        target_image = torch.from_numpy(target_image.transpose(2, 0, 1)).to(dtype=torch.float32)
        data["mask"] = torch.from_numpy(data["mask"].transpose(2, 0, 1)).to(dtype=torch.float32)
        data["degraded_image"] = degraded_image * 2. - 1.
        data["target_image"] = target_image * 2. - 1.
        data["mask"] = data["mask"] * 2. - 1.

        return data

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        while True:
            try:
                json_item = self.data_paths[index]
                data = self.parse_json(json_item)
                data = self.data_preprocess(data)
                return data
            except Exception as e:
                print(f"-> [RelightDataset], load images failed: {e}, data_meta: {self.data_paths[index]} ...")
                index = random.randint(0, len(self.data_paths) - 1)


def collate_fn(examples):
    degraded_image = torch.stack([example["degraded_image"] for example in examples]).to(
        memory_format=torch.contiguous_format).float()
    target_image = torch.stack([example["target_image"] for example in examples]).to(
        memory_format=torch.contiguous_format).float()
    mask = torch.stack([example["mask"] for example in examples]).to(
        memory_format=torch.contiguous_format).float()
    add_prompt = ""
    prompt = [example["prompt"] + add_prompt for example in examples]

    return {
        "source_image": degraded_image,
        "target_image": target_image,
        "mask": mask,
        "prompt": prompt,
    }

class DataRelightModule(pl.LightningDataModule):
    def __init__(
            self,
            train_config=None,
            eval_config=None,
            batch_size=1,
            num_workers=1,

    ):
        super().__init__()

        self.train_config = train_config
        self.eval_config = eval_config
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        """
        Setup the data module and create the webdataset processing pipelines
        """

        # train dataset
        self.train_dataset = RelightDataset(
            opt=self.train_config,
            shuffle_lines=True,
            shuffle_seed=0,
        )

        # eval dataset
        if self.eval_config is not None:
            self.eval_dataset = RelightDataset(
                opt=self.eval_config,
                shuffle_lines=True,
                shuffle_seed=0,
            )

    def train_dataloader(self):
        return data.DataLoader(
            self.train_dataset,
            shuffle=True,
            collate_fn=collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return data.DataLoader(
            self.eval_dataset,
            shuffle=True,
            collate_fn=collate_fn,
            batch_size=1,
        )
