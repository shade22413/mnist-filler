import copy
import random

import torch
from torch import nn
import matplotlib.pyplot as plt

import numpy as np
from mnsit_filler.utils import shuffle_jointly, hide_part


def torch_train_loop(model, data_train, data_test, batch_size=32, num_epochs=60,
                     criterion=nn.MSELoss(),
                     optimizer=torch.optim.Adam, weight_decay=0, lr=0.001,
                     valid_check=3, anneal_coef=0.6, max_not_improved=5, print_every=100000000, cuda=False,
                     seed=12362736):
    random.seed(seed, version=2)
    torch.manual_seed(seed)
    if cuda:
        model = model.cuda()

    optimizer = optimizer(model.parameters(), lr, weight_decay=weight_decay)
    best_loss = float('inf')
    not_improved = 0
    best_model = None
    for epoch in range(num_epochs):
        print("Epoch", epoch)
        total_loss = 0
        model.train()

        data_train = shuffle_jointly(data_train)
        for i in range(0, data_train.shape[0], batch_size):
            images = data_train[i:i+batch_size]
            inputs, targets = [], []
            for i in range(images.shape[0]):
                img, img_org = hide_part(images[i], part=0.5)
                inputs.append(img)
                targets.append(img_org)
                """plt.subplot(1,2,1)
                plt.imshow(img_org)
                plt.subplot(1,2,2)
                plt.imshow(img)
                plt.show()"""
            inputs = np.stack(inputs, axis=0)
            targets = np.stack(targets, axis=0)
            targets = targets.argmax(axis=-1)

            inputs, targets = torch.from_numpy(inputs).float(), torch.from_numpy(targets).float()
            inputs, targets = inputs.view(*inputs.shape[:-3], -1, inputs.shape[-1]), \
                              targets.view(*targets.shape[:-3], -1, targets.shape[-1])
            if cuda:
                inputs, targets = inputs.cuda(), targets.cuda()


            out = model(inputs)
            loss = criterion(out, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if (i // batch_size + 1) % print_every == 0:
                print('Loss :', total_loss / (i // batch_size + 1))
        print('Training Loss', total_loss / (i // batch_size + 1))
        total_loss = 0
        model.eval()

        for i in range(0, data_test.shape[0], batch_size):
            images = data_test[i:i + batch_size]
            inputs, targets = [], []
            for i in range(images.shape[0]):
                img, img_org = hide_part(images[i])
                inputs.append(img)
                targets.append(img_org)
            inputs = np.stack(inputs, axis=0)
            targets = np.stack(targets, axis=0)

            inputs, targets = torch.from_numpy(inputs).float(), torch.from_numpy(targets).float()
            if cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            out = model(inputs)
            loss = criterion(out, targets)
            loss.backward()
            total_loss += loss.item()

        total_loss = total_loss / (i // batch_size + 1)
        if best_loss > total_loss:
            best_loss = total_loss
            print('New best reached!')
            not_improved = 0
            best_model = copy.deepcopy(model)
        else:
            not_improved += 1

        if (not_improved + 1) % valid_check == 0:
            print('Learning rate decreased... lr=', end='')
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * anneal_coef
                print(param_group['lr'], end=' ')
            print()

        if not_improved > max_not_improved:
            break

        print('Validation Loss', total_loss)
        print('\n')

    print('Best Validation Loss :', best_loss)
    return best_model