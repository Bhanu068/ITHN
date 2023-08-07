import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.ann = json.loads(open(self.ann_path, 'r').read())
        self.examples = self.ann[self.split]
        for i in range(len(self.examples)):
            self.examples[i]['ids'] = tokenizer(self.examples[i]['report'])[:self.max_seq_length]
            self.examples[i]['neg_report_ids'] = tokenizer(self.examples[i]['neg_report'][0])[:self.max_seq_length]
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])
            self.examples[i]['neg_mask'] = [1] * len(self.examples[i]['neg_report_ids'])

    def __len__(self):
        return len(self.examples)


class IuxrayMultiImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example["id"]
        neg_image_id = example["neg_image_path"][0]
        image = Image.open(f"{self.image_dir}/{image_id}").convert('RGB')
        neg_image = Image.open(f"{self.image_dir}/{neg_image_id}").convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
            neg_image = self.transform(neg_image)
        report_ids = example['ids']
        report_masks = example['mask']
        neg_report_ids = example['neg_report_ids']
        neg_report_masks = example['neg_mask']
        seq_length = len(report_ids)
        neg_seq_length = len(neg_report_ids)
        sample = (image, neg_image, report_ids, neg_report_ids, report_masks, neg_report_masks, seq_length, neg_seq_length)
        return sample


class MimiccxrSingleImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        neg_image_path = example['neg_image_path'][0]
        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        neg_image = Image.open(os.path.join(self.image_dir, neg_image_path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
            neg_image = self.transform(neg_image)
        report_ids = example['ids']
        report_masks = example['mask']
        neg_report_ids = example['neg_report_ids']
        neg_report_masks = example['neg_mask']
        seq_length = len(report_ids)
        neg_seq_length = len(neg_report_ids)
        sample = (image, neg_image, report_ids, neg_report_ids, report_masks, neg_report_masks, seq_length, neg_seq_length)
        return sample
