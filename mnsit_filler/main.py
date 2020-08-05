import torch
import torchvision

from mnsit_filler.attention_rnn import AttentionRNN
from sklearn.model_selection import train_test_split
from mnsit_filler.train import torch_train_loop
from torch import nn
from mnsit_filler.utils import one_hot, LabelSmoothLoss
import numpy as np

mnist = torchvision.datasets.MNIST('./', download=False)
mnist_target = mnist.targets
mnist = mnist.data.long()
mnist = mnist // 32
print(mnist.shape)



data_train, data_test = train_test_split(mnist.numpy(), random_state=42, test_size=0.1, stratify=mnist_target.numpy())
del mnist
del mnist_target

model = AttentionRNN(8, 8, 28*28, project_size=28, encoder_hidden_size=64, encoder_num_layers=1, decoder_hidden_size=64,
                     decoder_num_layers=1, dropout=0.1)

print(model)

model = torch_train_loop(model, data_train, data_test, batch_size=4, num_epochs=60, criterion=LabelSmoothLoss(0.02),
                         print_every=1, lr=0.001)

torch.save(model, 'model.pt')





