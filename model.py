import torch.nn as nn
from transformers import BertModel


class Bert(nn.Module):
    def __init__(self):
        super(Bert, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(768, 14)

    def forward(self, text):
        _, pooled = self.bert(text)
        out = self.fc(pooled)
        return out
