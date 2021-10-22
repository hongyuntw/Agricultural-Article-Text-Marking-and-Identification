from typing import Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn


def compute_acc(pred, label):
    pred = pred.detach()
    pred = F.softmax(pred, dim=1)
    pred = pred.argmax(dim=-1)
    return (pred == label).float().mean()