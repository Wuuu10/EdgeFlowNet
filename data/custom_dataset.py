import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2


class LightweightEdgeDataset(Dataset):
    def __init__(self, config, split='train'):
        self.config = config
        self.split = split
        self.data_root = config.DATA_ROOT

        self.image_dir = os.path.join(self.data_root, config.IMAGE_DIR)
        self.label_dir = os.path.join(self.data_root, config.LABEL_DIR)

        list_file = config.TRAIN_LIST if split == 'train' else config.VAL_LIST
        list_path = os.path.join(self.data_root, list_file)

        with open(list_path, 'r') as f:
            self.file_list = [line.strip() for line in f.readlines()]

        print(f"Loaded {len(self.file_list)} {split} samples")

        self.input_size = config.INPUT_SIZE
        self.water_label = config.WATER_LABEL

        if split == 'train':
            self.transform = LightweightDataAugmentation(config)
        else:
            self.transform = None

        self.normalize = transforms.Normalize(mean=config.MEAN, std=config.STD)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]

        if filename.endswith(('.jpg', '.png')):
            filename = os.path.splitext(filename)[0]

        image_path = os.path.join(self.image_dir, filename + '.jpg')
        label_path = os.path.join(self.label_dir, filename + '.png')

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file not found: {label_path}")

        image = Image.open(image_path).convert('RGB')

        label_pil = Image.open(label_path)
        if label_pil.mode == 'P':
            label_array = np.array(label_pil)
        else:
            label_array = np.array(label_pil.convert('L'))

        image = image.resize((self.input_size, self.input_size), Image.BILINEAR)

        label_pil_for_resize = Image.fromarray(label_array.astype(np.uint8))
        label_resized = label_pil_for_resize.resize(
            (self.input_size, self.input_size), Image.NEAREST
        )
        label_array_resized = np.array(label_resized)

        image = transforms.ToTensor()(image)
        label = torch.from_numpy(label_array_resized.astype(np.int64))

        binary_label = torch.zeros_like(label)
        binary_label[label == self.water_label] = 1
        label = binary_label

        if self.transform is not None:
            image, label = self.transform(image, label)

        image = self.normalize(image)

        edge_weight = self._generate_edge_weights(label)

        return {
            'image': image,
            'label': label,
            'edge_weight': edge_weight,
            'filename': filename
        }

    def _generate_edge_weights(self, mask):
        if isinstance(mask, torch.Tensor):
            mask_np = mask.numpy().astype(np.uint8)
        else:
            mask_np = mask.astype(np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated = cv2.dilate(mask_np, kernel, iterations=1)
        eroded = cv2.erode(mask_np, kernel, iterations=1)
        edges = dilated - eroded

        weights = np.ones_like(mask_np, dtype=np.float32)
        weights[edges > 0] = self.config.EDGE_ENHANCEMENT_STRENGTH

        return torch.from_numpy(weights).float()


class LightweightDataAugmentation:
    def __init__(self, config):
        self.config = config

        self.flip_prob = 0.5
        self.color_prob = 0.4
        self.rotate_prob = 0.3

        self.brightness = 0.15
        self.contrast = 0.15
        self.saturation = 0.1
        self.rotation_degrees = 10

    def __call__(self, image, label):
        if torch.rand(1) < self.flip_prob:
            image = transforms.functional.hflip(image)
            label = transforms.functional.hflip(label.unsqueeze(0)).squeeze(0)

        if torch.rand(1) < 0.2:
            image = transforms.functional.vflip(image)
            label = transforms.functional.vflip(label.unsqueeze(0)).squeeze(0)

        if torch.rand(1) < self.color_prob:
            if torch.rand(1) < 0.7:
                brightness_factor = 1.0 + (torch.rand(1) - 0.5) * 2 * self.brightness
                image = transforms.functional.adjust_brightness(image, brightness_factor.item())

            if torch.rand(1) < 0.6:
                contrast_factor = 1.0 + (torch.rand(1) - 0.5) * 2 * self.contrast
                image = transforms.functional.adjust_contrast(image, contrast_factor.item())

            if torch.rand(1) < 0.4:
                saturation_factor = 1.0 + (torch.rand(1) - 0.5) * 2 * self.saturation
                image = transforms.functional.adjust_saturation(image, saturation_factor.item())

        if torch.rand(1) < self.rotate_prob:
            angle = (torch.rand(1) - 0.5) * 2 * self.rotation_degrees
            image = transforms.functional.rotate(image, angle.item(), fill=0)
            label = transforms.functional.rotate(label.unsqueeze(0), angle.item(), fill=0).squeeze(0)

        label = torch.clamp(label, 0, 1)

        return image, label


def lightweight_collate_fn(batch_list):
    if not batch_list:
        return {}

    batch_size = len(batch_list)
    first_sample = batch_list[0]

    image_shape = first_sample['image'].shape
    label_shape = first_sample['label'].shape

    images = torch.empty((batch_size,) + image_shape, dtype=torch.float32)
    labels = torch.empty((batch_size,) + label_shape, dtype=torch.long)
    edge_weights = torch.empty((batch_size,) + label_shape, dtype=torch.float32)
    filenames = []

    for i, sample in enumerate(batch_list):
        images[i] = sample['image']
        labels[i] = sample['label']
        edge_weights[i] = sample['edge_weight']
        filenames.append(sample['filename'])

    return {
        'image': images,
        'label': labels,
        'edge_weight': edge_weights,
        'filename': filenames
    }


def get_dataloader(config, split='train'):
    dataset = LightweightEdgeDataset(config, split)

    if split == 'train':
        batch_size = config.BATCH_SIZE
        shuffle = True
        drop_last = True
    else:
        batch_size = config.VAL_BATCH_SIZE
        shuffle = False
        drop_last = False

    dataloader_kwargs = {
        'dataset': dataset,
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': config.NUM_WORKERS,
        'pin_memory': config.PIN_MEMORY,
        'drop_last': drop_last,
        'collate_fn': lightweight_collate_fn,
        'persistent_workers': config.NUM_WORKERS > 0
    }

    if config.NUM_WORKERS > 0:
        dataloader_kwargs['prefetch_factor'] = config.PREFETCH_FACTOR

    dataloader = DataLoader(**dataloader_kwargs)

    print(f"Created {split} dataloader: batch_size={batch_size}, num_workers={config.NUM_WORKERS}")

    return dataloader


def preprocess_batch(batch, device):
    processed_batch = {}

    if 'image' in batch:
        processed_batch['image'] = batch['image'].to(device, non_blocking=True)

    if 'label' in batch:
        processed_batch['label'] = batch['label'].to(device, non_blocking=True)

    if 'edge_weight' in batch:
        processed_batch['edge_weight'] = batch['edge_weight'].to(device, non_blocking=True)

    if 'filename' in batch:
        processed_batch['filename'] = batch['filename']

    return processed_batch
