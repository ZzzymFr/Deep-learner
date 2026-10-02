
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNClassifier, self).__init__()