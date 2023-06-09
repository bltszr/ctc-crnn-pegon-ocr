# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import glob
import os
from torchvision.transforms import transforms
from torch.utils.data import Dataset
from PIL import Image

class FilenameOCRDataset(Dataset):
    def __init__(self, directory, image_transform=transforms.ToTensor(),
                label_transform=lambda x:x, filetypes=['png']):
        self.directory = directory
        self.files = sum((glob.glob(os.path.join(directory, f'*.{filetype}')) for filetype in filetypes), [])
        self.image_transform = image_transform
        self.label_transform = label_transform

    def __getitem__(self, idx):
        path = self.files[idx]
        img = self.image_transform(Image.open(path))
        label = self.label_transform(os.path.basename(path[:path.rfind('.')]))
        return img, label
    def __len__(self):
        return len(self.files)


# +
from collections import OrderedDict

# ROMAN_NUMERALS = "\u2160 \u2161 \u2162 \u2163 \u2164 \u2165 \u2166 \u2167 \u2168 \u2169 \u216A \u216B \u216C \u216D \u216E \u216F \u2180 \u2181 \u2182 \u2183".split()
# PEGON_CHARS = ['-'] + [' '] + 'َ ِ ُ ً ٍ ٌ ْ ّ ٰ ࣤ \u06e4 \u0653 ١ ٢ ٣ ٤ ٥ ٦ ٧ ٨ ٩ ٠ ة ح چ ج ث ت ب ا أ إ آ ؤ ى س ز ر ࢮ ڎ ذ د خ ع ظ ڟ ط ض ص ش ڮ ك ق ڤ ف ڠ غ ي ه و ۑ ئ ن م ل ۔ : ؛ ، ﴾ ﴿ ( ) ! ؟َ « » ۞ ء'.split()
PEGON_CHARS = ['_'] + [' '] + list(OrderedDict.fromkeys('َ ِ ُ ً ٍ ٌ ْ ّ َ ٰ ࣤ \u06e4 \u0653 ١ ٢ ٣ ٤ ٥ ٦ ٧ ٨ ٩ ٠ ة ح چ ج ث ت ب ا أ إ آ ؤ ى س ز ر ࢮ ڎ ذ د خ ع ظ ڟ ط ض ص ش ڮ ك ق ڤ ف ڠ غ ي ه و ۑ ئ ن م ل ۔ : ؛ ، ( ) ! ؟ « » ۞ ء'.split())) + ['\ufffd']
CHAR_MAP = {letter: idx for idx, letter in enumerate(PEGON_CHARS)}

# +
import unicodedata, glob, os, re
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image

class AnnotatedDataset(Dataset):
    tokens_to_ignore = []
    tokens_to_unknown = []

    translation_table = str.maketrans('', '')
    
    def __init__(self, directory, image_transform=transforms.ToTensor(),
                 blank_idx=0, unknown_idx=-1,
                 filetypes=['png'], char_map=CHAR_MAP):
        self.directory = directory
        self.files = sum((glob.glob(os.path.join(directory, f'*.{filetype}')) for filetype in filetypes), [])
        self.image_transform = image_transform
        self.char_map = char_map
        self.blank_char = list(self.char_map.keys())[blank_idx]
        self.unknown_char = list(self.char_map.keys())[unknown_idx]
        self.blank_idx = self.char_map[self.blank_char]
        self.unknown_idx = self.char_map[self.unknown_char]
        if len(self.__class__.tokens_to_ignore) == 0:
            self.__class__.ignore_pattern = r'$^'
        else:
            self.__class__.ignore_pattern = "|".join(map(re.escape, self.__class__.tokens_to_ignore))
        if len(self.__class__.tokens_to_unknown) == 0:
            self.__class__.unknown_pattern = r'$^'
        else:
            self.__class__.unknown_pattern = "|".join(map(re.escape, self.__class__.tokens_to_unknown))
    
    def filename_to_label(self, filename):
        return filename
    def to_class(self, char):
        try:
            return self.char_map[char]
        except KeyError:
            return self.char_map[self.unknown_char]

    def char_segment(self, label):
        return list(label)
    
    def label_transform(self, label):
        label = self.filename_to_label(label)
        label = re.sub(self.__class__.unknown_pattern, self.unknown_char, label)
        label = re.sub(self.__class__.ignore_pattern, '', label)
        label = label.translate(self.__class__.translation_table)
        
#         label = [self.to_class(c) for c in filter(lambda c:unicodedata.category(c)[0] != 'C',
#                                                   f'-{"-".join(self.char_segment(label))}-')]
        label = list(map(self.to_class, filter(lambda c:unicodedata.category(c)[0] != 'C', label)))
        return label

    def __getitem__(self, idx):
        path = self.files[idx]
        img = self.image_transform(Image.open(path))
        label = self.label_transform(os.path.basename(path[:path.rfind('.')]))
        return img, label
    
    
    def __len__(self):
        return len(self.files)


# -


class PegonAnnotatedDataset(AnnotatedDataset):
    tokens_to_unknown = ['[CALLIGRAPHY]',
                        '[NASTALIQ]',
                        '[UNKNOWN]',
                        '[VERT]']
    
    tokens_to_ignore = ['‌']

    translation_table = str.maketrans('1234567890', '١٢٣٤٥٦٧٨٩٠')
    
    def filename_to_label(self, filename):
        return filename.split(';')[0]


class QuranAnnotatedDataset(AnnotatedDataset):
    tokens_to_ignore = []
    tokens_to_unknown = []
    
    # there are far more characters that can be translated, however
    # at this point it's way too late to correct that
    translation_table = translation_table = str.maketrans(
        ''.join(list('1234567890') + ['\u06e1'] + ['\u0657'] + ['\u065e']), # + ['\u0671']),
        ''.join(list('١٢٣٤٥٦٧٨٩٠') + ['\u0652'] + ['\u064f'] + ['\u064e'])) # + ['\u0627']))


# +
from tqdm.notebook import tqdm
import json
import shutil
import random
import datetime
import pickle

class OCRDataset(Dataset):
    
    def __init__(self, char_map=CHAR_MAP, img_format='png',
                 target_dir='tmp', transform=transforms.ToTensor(),
                 max_seq_len=0, files_and_labels=[],
                 avg_img_w=0, avg_img_h=0, avg_seq_len=0,
                 blank_idx=0, unknown_idx=-1,):
        self.img_format = img_format
        self.target_dir = target_dir
        self.files_and_labels = files_and_labels
        
        self.char_map = char_map
        self.blank_char = list(self.char_map.keys())[blank_idx]
        self.unknown_char = list(self.char_map.keys())[unknown_idx]
        self.blank_idx = self.char_map[self.blank_char]
        self.unknown_idx = self.char_map[self.unknown_char]
        
        self.max_seq_len = max_seq_len
        self.files_and_labels = files_and_labels
        self.transform = transform
        self.avg_seq_len = avg_seq_len
        self.avg_img_w = avg_img_w
        self.avg_img_h = avg_img_h
    
    def read_directory(self, file_dir,
                       force_rewrite=False,
                       skip_duplicates=False,
                       in_img_format='png',
                       sample=1):
        if force_rewrite:
            self.files_and_labels = []
        total_seq_len = 0
        total_img_h = 0
        total_img_w = 0
        
        if os.path.exists(self.target_dir):
            if not force_rewrite:
                raise FileExistsError(f'{self.target_dir} already exists and force rewrite not allowed')
            else:
                shutil.rmtree(self.target_dir)

        os.makedirs(self.target_dir)
        
        files = glob.glob(os.path.join(file_dir, f"**/*.{in_img_format}"), recursive=True)
        
        for fullpath in tqdm(random.sample(files, int(len(files) * sample))):

            with open(re.sub(f'.{in_img_format}$', '.json', fullpath), 'r') as f:
                bbox_list = json.load(f)
            with Image.open(fullpath) as image:
                for idx, obj in enumerate(bbox_list):
                    label = obj['text']
                    seq_len = len(label)
                    left, top, right, bottom = obj['bbox']
                    if (right - left) * (bottom - top) == 0:
                        label = obj['text']
                        print(f"Skipping empty bbox ({left}, {top}, {right}, {bottom}) and label {label}")
                        continue

                    cropped_image = image.crop(obj['bbox'])
                    target_timestamp = str(datetime.datetime.now().timestamp())
                    target_filepath = os.path.join(self.target_dir, f'{target_timestamp}.{self.img_format}')
                    cropped_image.save(target_filepath)
            
                    self.files_and_labels.append((target_timestamp, label))
                
                    self.max_seq_len = max(seq_len, self.max_seq_len)
                    total_seq_len += seq_len
                    total_img_h += bottom - top
                    total_img_w += right - left

        self.avg_seq_len = int(total_seq_len / len(self.files_and_labels))
        self.avg_img_w = int(total_img_w / len(self.files_and_labels))
        self.avg_img_h = int(total_img_h / len(self.files_and_labels))
    
    
    def save(self, filename='metadata.json'):
        # Create a dictionary with all JSON-serializable attributes
        data = {}
        for attr in dir(self):
            if not attr.startswith("__"):
                try:
                    json.dumps(getattr(self, attr))
                    data[attr] = getattr(self, attr)
                except TypeError:
                    pass
        # Write the dictionary to a JSON file
        with open(os.path.join(self.target_dir, filename), 'w') as outfile:
            json.dump(data, outfile)

    @classmethod
    def load_static(cls, filename):
        self = object.__new__(cls)
        self.__class__ = cls
        with open(filename) as infile:
            data = json.load(infile)
        # Set the selected attributes in the object
        for attr, value in data.items():
            setattr(self, attr, value)
        return self
    
    def load(self, filename):
        with open(filename) as infile:
            data = json.load(infile)
        # Set the selected attributes in the object
        for attr, value in data.items():
            setattr(self, attr, value)
        return self
    
    def set_transform(self, transform):
        self.transform = transform
    
    def __len__(self):
        return len(self.files_and_labels)

    def to_class(self, char):
        try:
            return self.char_map[char]
        except KeyError:
            return self.char_map[self.unknown_char]

    def char_segment(self, label):
        # later this can be fine-tuned to only split characters and not harakat
        return list(label)

    def __getitem__(self, idx):
        image_timestamp, label = self.files_and_labels[idx]
        image_path = os.path.join(self.target_dir, f'{image_timestamp}.{self.img_format}')
        cropped_image = self.transform(Image.open(image_path).convert('RGB'))
#         blanked_label = self.blank_char + self.blank_char.join(self.char_segment(label)) + self.blank_char
#         label = list(map(self.to_class, filter(lambda c:unicodedata.category(c)[0] != 'C',
#                                                blanked_label)))
        label = list(map(self.to_class,
                         filter(lambda c:unicodedata.category(c)[0] != 'C',
                                label)))
        return cropped_image, label


# -
def ctc_collate_fn(batch):
    
    batch = sorted(batch, key=lambda x: x[0].shape[2], reverse=True)
    
    images = [item[0] for item in batch]
    labels = [torch.Tensor(item[1]) for item in batch]
    label_lengths = torch.LongTensor([len(label) for label in labels])
    
    labels = torch.cat((labels))

    images = torch.stack(images, dim=0)

    return images, labels, label_lengths


if (__name__ == '__main__') and reread:
    import unicodedata
    dataset = OCRDataset()
    dataset.read_directory('/workspace/Dataset/Synthesized',
                           force_rewrite=True, skip_duplicates=True,
                           sample=1)
    dataset.save()
    dataset = OCRDataset().load('tmp/metadata.json')
    dataset[0]

# +
import matplotlib.pyplot as plt

class LossHistory:
    def __init__(self):
        self.step_history = []
        self.epoch_boundaries = []
        self.val_history = []
    
    def append(self, loss):
        self.step_history.append(loss)
        
    def mark_epoch(self):
        self.epoch_boundaries.append(len(self.step_history))
        
    def add_val(self, value):
        self.val_history.append(value)
    
    def plot(self, epoch_alpha=0.5):
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(self.step_history, color='C0')
        if len(self.val_history) == len(self.epoch_boundaries):
            ax2.plot(self.epoch_boundaries, self.val_history, color='C1')
        for epoch in self.epoch_boundaries:
            ax1.axvline(x=epoch, color='C2', linestyle='--', alpha=epoch_alpha)


# +
from torch.utils.data import DataLoader, Dataset
import torch.nn.utils as utils
from torch.utils.data import RandomSampler

class CTCTrainer:
    def __init__(self, model,
                 batch_size,
                 dataset,
                 collate_fn=ctc_collate_fn,
                 image_dir=None,
                 blank_idx=0, lr=0.001,
                 added_padding=1,
                 dropout_rate=0.25,
                 max_norm=1.0,
                 criterion=None,
                 optimizer=None,
                 num_workers=0,
                 alphabet=PEGON_CHARS,
                 sample_rate=None):
        self.blank_idx = blank_idx
        self.collate_fn = collate_fn
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = dataset
        self.max_norm = max_norm
        self.sample_rate = sample_rate
        if sample_rate != None:
            sampler = RandomSampler(self.dataset, num_samples=int(len(dataset)*sample_rate))
            shuffle = False
        else:
            sampler = None
            shuffle = True
        self.dataloader = DataLoader(dataset,
                                     batch_size=self.batch_size,
                                     shuffle=shuffle,
                                     num_workers=num_workers,
                                     collate_fn=self.collate_fn, 
                                     sampler=sampler)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.alphabet = alphabet
        
        # Define the model, criterion, and optimizer
        self.model = model.to(self.device)
        if criterion == None:
            self.criterion = nn.CTCLoss(blank=blank_idx)
        else:
            self.criterion = criterion
        
        if optimizer == None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        else:
            self.optimizer = optimizer
    
    def debug_blank_labels(self, labels):
        try:
            assert (labels != self.blank_idx).all()
        except AssertionError:
            labels_ = labels.split(tuple(label_lengths.cpu()))
            problem_labels = []
            for label in labels_:
                if (label == self.blank_idx).any():
                    problem_labels.append(''.join(self.alphabet[idx] for idx in label.type(torch.int).numpy()))
            problem_labels = "\n".join(problem_labels)
            raise ValueError(f'Blank label ({self.alphabet[self.blank_idx]}) at the following labels:\n{problem_labels}')
    def debug_output_nan(self, output, images, labels):        
        # Find the index of the batch that caused the NaN output
        batch_indices = output.mean(dim=(1, 2)).isnan().nonzero(as_tuple=True)[0]
        # Extract the corresponding image(s)
        problematic_images = images.index_select(dim=0, index=batch_indices).cpu()
        labels_ = labels.split(tuple(label_lengths.cpu()))
        problematic_labels = labels_.index_select(dim=0, index=batch_indices).cpu()

        fig, axs = plt.subplots(nrows=len(problematic_images), ncols=1, squeeze=False,
                               figsize=(20, 2*(len(problematic_images))))
        # Plot each image on its own subplot
        axs = axs.ravel()
        if problematic_images.ndim == 3:
            # Add a channel dimension
            problematic_images = problematic_images.unsqueeze(1)
        for i, (img, label) in enumerate(zip(problematic_images, problematic_labels)):
            axs[i].imshow(img.permute(1, 2, 0))
            axs[i].set_title(f"label = {''.join(self.alphabet[idx] for idx in label.numpy())}")
        # Show the plot
        plt.tight_layout()
        plt.show()
        raise ValueError(f'Output is NaN at {batch_indices}, {output}')
    
    def _train_loop(self, loss_history, num_epochs, dataloader, debug, save_path=None,
                   val_dataloader=None, eval_routine=None, plot_path=None):
        for epoch in range(num_epochs):
            running_loss = 0.0
            running_tokens = 0
            for i, (images, labels, label_lengths) in (pbar := tqdm(enumerate(dataloader),
                                                                   total=len(dataloader))):
                
                if debug:
                    self.debug_blank_labels(labels)
                # Move the data to the device (GPU if available)
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                output = self.model(images)
                
                input_lengths = torch.full((len(labels),), output.shape[1],
                                            dtype=torch.long)
                
                if debug:
                    if output.mean().isnan():
                        self.debug_output_nan(output, images, labels)
                    
                # Compute the loss
                image_lengths = output.new_full((output.shape[0],),
                                                output.shape[1],
                                                dtype=torch.int)

                loss = self.criterion(output.transpose(0, 1), labels,
                                      input_lengths=image_lengths,
                                      target_lengths=label_lengths)

                # Backward pass and optimize
                loss.backward()
                
                # Clip gradients
                if self.max_norm:
                    utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
                
                self.optimizer.step()

                # Add the batch loss to the running loss
                running_loss += loss.item() * labels.shape[0]
                running_tokens = labels.numel()

                curr_loss = running_loss/running_tokens
                loss_history.append(loss=curr_loss)
                # Print the average loss every batch
                pbar.set_description(f"Epoch [{epoch+1}/{num_epochs}] | Batch [{i+1}/{len(dataloader)}] | Running Loss: {curr_loss:.4f}")
            if val_dataloader and eval_routine:
                loss_history.add_val(eval_routine(self.model, val_dataloader))
                self.model.train()
            loss_history.mark_epoch()
            if save_path != None:
                self.save(save_path)
            if plot_path != None:
                self.plot_history(plot_path, save_only=True)
    
    def train(self, num_epochs, debug=False, save_path=None,
              val_dataloader=None, eval_routine=None, plot_path=None):
        before = datetime.datetime.now()
        torch.autograd.set_detect_anomaly(debug)
        self.model.train()
        
        self.loss_history = LossHistory()
        self._train_loop(self.loss_history, num_epochs, self.dataloader, debug, plot_path=plot_path,
                         save_path=save_path, val_dataloader=val_dataloader, eval_routine=eval_routine)
        after = datetime.datetime.now()
        print(f"Finished training! Took {after - before}.")
        return self.model

    def save(self, path):
        torch.save(self.model, path)

    def plot_history(self, path=None, epoch_alpha=0.5, save_only=False):
        self.loss_history.plot(epoch_alpha)
        if path != None:
            plt.savefig(path)
        if not save_only:
            plt.show()


# -

class SPDFCTCTrainer(CTCTrainer):
    def __init__(self, finetune_dataset, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.finetune_dataset = finetune_dataset
        self.finetune_dataloader = DataLoader(self.dataset,
                                              batch_size=self.batch_size,
                                              shuffle=True,
                                              collate_fn=self.collate_fn)
        
        assert callable(getattr(self.model, "sparse", None)), "Model must implement `toggle_sparsity` method!"
    def train(self, num_epochs_sparse, num_epochs_dense, debug=False):
        torch.autograd.set_detect_anomaly(debug)
        self.model.sparse(True)
        self.model.train()
        self.loss_history = LossHistory()
        self._train_loop(self.loss_history, num_epochs_sparse, self.dataloader, debug)
        print("Finished sparse pretraining. Dense finetuning...")
        self.model.sparse(False)
        self._train_loop(self.loss_history, num_epochs_dense, self.finetune_dataloader, debug)
        print("Finished training!")
        return self.model    


def model_length(b=2, c=1):
    return lambda length:(b * length) + c


# +
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalCTCLoss(nn.Module):
    reductions = {
        'mean': torch.mean,
        'sum': torch.sum,
        'none': lambda x:x
    }
    def __init__(self, blank=0, reduction='mean',
                 zero_infinity=False,
                 alpha=0.5, gamma=2):
        """
        Args: 
            blank (int, optional) – blank label. Default 0.
            reduction (str, optional) – Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. 'none': no reduction will be applied, 'mean': the output losses will be divided by the target lengths and then the mean over the batch is taken. Default: 'mean'
            zero_infinity (bool, optional) – Whether to zero infinite losses and the associated gradients. Default: False Infinite losses mainly occur when the inputs are too short to be aligned to the targets.

            alpha (float): The weighting factor for hard examples.
            gamma (float): The focusing parameter for hard examples.
        """
        super().__init__()
        self.ctc_loss = nn.CTCLoss(blank=blank, zero_infinity=zero_infinity,
                                   reduction='none')
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = FocalCTCLoss.reductions[reduction]
        
        
    def forward(self, log_probs, targets, input_lengths, target_lengths):
        """
        Computes the Focal CTC loss for a batch of predictions and targets.

        Args:
            Log_probs (torch.Tensor): Tensor of size (T,N,C) or (T,C),
            where T=input length, N=batch size, and C=number of classes (including blank).
            The logarithmized probabilities of the outputs (e.g. obtained with torch.nn.functional.log_softmax()).

            Targets: Tensor of size (N,S) or (sum⁡(target_lengths)),
            where N=batch sizeN=batch size and S=max target length,
            if shape is (N,S), if shape is (N,S).
            It represent the target sequences.
            Each element in the target sequence is a class index.
            And the target index cannot be blank (default=0).
            In the (N,S) form, targets are padded to the length of the longest sequence, and stacked.
            In the (sum(target_lengths)) form, the targets are assumed to be un-padded
            and concatenated within 1 dimension.

            Input_lengths: Tuple or tensor of size (N) or (), where N=batch size.
            It represent the lengths of the inputs (must each be ≤T).
            And the lengths are specified for each sequence to achieve
            masking under the assumption that sequences are padded to equal lengths.

            Target_lengths: Tuple or tensor of size (N) or ), where N=batch size.
            It represent lengths of the targets.
            Lengths are specified for each sequence to achieve masking
            under the assumption that sequences are padded to equal lengths.
            If target shape is (N,S), target_lengths are effectively the stop index s_n
            for each target sequence, such that target_n = targets[n,0:s_n] for each target in a batch.
            Lengths must each be ≤ If the targets are given as a 1d tensor
            that is the concatenation of individual targets,
            the target_lengths must add up to the total length of the tensor.

        Returns:
            torch.Tensor: A scalar tensor containing the Focal CTC loss.
        """
        try:
            T, N, C = log_probs.size()
        except ValueError:
            N = 1
            T, C = log_probs.size()

        # Compute the CTC loss.
        ctc_loss = self.ctc_loss(log_probs,
                                 targets,
                                 input_lengths,
                                 target_lengths)

        # Compute the weights for the focal term.
        p = torch.exp(-ctc_loss)
        focal_weight = self.alpha * (1 - p)**self.gamma

        # Compute the final Focal CTC loss.
        loss = self.reduction(ctc_loss * focal_weight)
        return loss

# +
from ctc_decoder import best_path, beam_search

class CTCDecoder:
    def __init__(self, model, char_map, blank_char='-'):
        self.model = model.eval()
        self.alphabet = list(char_map.keys())
        self.blank_char = blank_char
        blank_idxs = [idx for idx, x in enumerate(self.alphabet) if x == self.blank_char]
        assert len(blank_idxs) == 1
        self.blank_index = blank_idxs[0]
    
    @classmethod
    def from_path(cls, model_path, alphabet, *args, **kwargs):
        saved_model = torch.load(model_path)
        return cls(saved_model, alphabet, *args, **kwargs)
    
    def convert_to_text(self, output):
        output = torch.argmax(output, dim=2).detach().cpu().numpy()
        texts = []
        for i in range(output.shape[0]):
            text = ''
            for j in range(output.shape[1]):
                if output[i, j] != self.blank_index and (j == 0 or output[i, j] != output[i, j-1]):
                    text += self.alphabet[output[i, j]]
            texts.append(text)
        return texts
    
    def infer(self, data):
        model_out = self.model(data)
#         print(model_out.shape)
        return self.convert_to_text(model_out)

class BestPathDecoder(CTCDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.all_chars = [x for x in self.alphabet if x != self.blank_char]
        
    def convert_to_text(self, output):
        output = torch.roll(F.softmax(output, dim=2), -1, 2).detach().cpu().numpy()
        texts = []
        for i in range(output.shape[0]):
#             pdb.set_trace()
            texts.append(best_path(output[i, :, :], self.all_chars))
        return texts


# +
# from ctcdecode import CTCBeamDecoder

# class BeamSearchDecoder(CTCDecoder):
#     def __init__(self, *args, beam_width=100, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.all_chars = [x for x in self.alphabet if x != self.blank_char]
#         self.beam_width=beam_width
# #         self.decoder = CTCBeamDecoder(
# #             self.alphabet,
# #             model_path=None,
# #             alpha=0,
# #             beta=0,
# #             cutoff_top_n=40,
# #             cutoff_prob=1.0,
# #             beam_width=beam_width,
# #             num_processes=4,
# #             blank_id=self.blank_index,
# #             log_probs_input=False)
    
#     @classmethod
#     def from_path(cls, *args, beam_width=100, **kwargs):
#         self = super().from_path(*args, **kwargs)
#         self.all_chars = [x for x in self.alphabet if x != self.blank_char]
#         self.beam_width = beam_width
#         return self
        
#     def convert_to_text(self, output):
#         output = torch.roll(F.softmax(output, dim=2), -1, 2).detach()
#         texts = []
#         for i in range(output.shape[0]):
#             texts.append(beam_search(output[i, :, :], self.all_chars,
#                                      beam_width=self.beam_width))
# #         res, _, _, out_lens = self.decoder.decode(output)
# #         texts = [''.join([tensor[0][:out_len]]) for tensor, out_len in zip(res, out_lens)]
#         return texts

# +
from jiwer import wer, cer
from itertools import starmap

def evaluate(decoder, dataloader):
    char_map = decoder.alphabet
    
    def decode(arr):
        try:
            return ''.join(list(map(lambda x:char_map[x], arr)))
        except IndexError as e:
            e.args += (arr, f"Char map length: {len(char_map)}")
            raise
    
    cers = []
    wers = []
    tot_cer = 0
    tot_wer = 0
    num_examples = 0
    for i, (images, labels, target_lengths) in (pbar := tqdm(enumerate(dataloader),
                                                             total=len(dataloader))):
            images = images.to('cuda')
            labels = map(lambda tensor:tensor.type(torch.int), labels.split(tuple(target_lengths.cpu())))
            outputs = decoder.infer(images)
            for j, (output, true_label) in enumerate(zip(outputs, map(decode, labels))):
                label = true_label.replace(decoder.blank_char, '')
                try:
                    cer_ = cer(label, output)
                    wer_ = wer(label, output)
                except ValueError:
                    pbar.write(f'[WARN]: ValueError with predicted {output} and label {label}')
                    if label == '':
                        continue
                    cer_ = 1
                    wer_ = 1
                cers.append(cer_)
                wers.append(wer_)
                tot_cer += cer_
                tot_wer += wer_
                num_examples += 1
                pbar.set_description(f" Example: {num_examples} | CER: {tot_cer/(num_examples):.4f} | WER: {tot_wer/(num_examples):.4f}")
    return cers, wers


# +
import numpy as np

def plot_cer_wer(cers, wers, path=None):
    cer_mean = np.mean(cers)
    wer_mean = np.mean(wers)
    fig, axs = plt.subplots(1, 2, figsize=(20, 4))
    axs[0].hist(cers, bins=100)
    axs[0].axvline(x=cer_mean, color='orange')
    axs[0].text(cer_mean,5.0, f'mean={cer_mean:.2f}')
    axs[0].set_title(f'CER')
    axs[1].hist(wers, bins=100)
    axs[1].set_title('WER')
    axs[1].text(wer_mean,5.0, f'mean={wer_mean:.2f}')
    axs[1].axvline(x=wer_mean, color='orange')
    if path != None:
        plt.savefig(path)
    plt.show()


# -

len(PegonAnnotatedDataset('/workspace/Dataset/pegon-ocr-patched'))


