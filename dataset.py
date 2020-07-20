import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import re
from tqdm import tqdm

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


class DataSet(Dataset):
    def __init__(self, path):
        super(DataSet, self).__init__()
        self.file = open(path, 'r')
        self.labels = []
        self.sentences = []

        for line in self.file:
            label = int(re.findall('__label__(\d*)', line)[0])
            sentence = re.findall('__label__\d* (.*)', line)[0]
            self.labels.append(label-1)
            self.sentences.append(sentence)

        self.input_ids = []
        self.attention_masks = []

        self.sentences = self.sentences[:200]
        self.labels = self.labels[:200]
        for s in tqdm(self.sentences, desc='Encoding dataset'):
            encoded_dict = tokenizer.encode_plus(
                s,
                add_special_tokens=True,
                max_length=64,
                pad_to_max_length=True,
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            self.input_ids.append(encoded_dict['input_ids'])
            self.attention_masks.append(encoded_dict['attention_mask'])

        self.input_ids = torch.cat(self.input_ids, dim=0)
        self.attention_masks = torch.cat(self.attention_masks, dim=0)
        self.labels = torch.tensor(self.labels)

    def __getitem__(self, index):
        return self.labels[index], self.attention_masks[index], self.input_ids[index]

    def __len__(self):
        return len(self.labels)
