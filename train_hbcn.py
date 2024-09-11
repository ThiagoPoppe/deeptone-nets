import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from torch.utils.data import DataLoader
from source.data.constants import RANDOM_SEED

from source.models.chroma import HarmonyBassChromaNetwork
from source.data.datasets import HarmonyBassChromaDataset

BATCH_SIZE = 256
NUM_EPOCHS = 100
WEIGHT_DECAY = 1e-4
LEARNING_RATE = 1e-4

device = torch.device('cuda' if torch.cuda.is_availabel() else 'cpu')
print('Using device:', device)


def fix_random_seeds():  
    """
    Auxiliary function to fix seed and ensure reproducibility.
    """
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)


def parse_command_line_arguments():
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name to train model (must exist within datasets folder)')

    args = parser.parse_args()
    return args


def train(model, train_dataloader, criterion, optimizer):
    train_loss = 0
    bass_train_loss = 0
    harmony_train_loss = 0
    
    model.train()
    for spec, bass_label, harmony_label in train_dataloader:
        spec = spec.to(device)
        bass_label = bass_label.to(device)
        harmony_label = harmony_label.to(device)

        bass_out, harmony_out = model(spec)
        bass_loss = criterion(bass_out, bass_label)
        harmony_loss = criterion(harmony_out, harmony_label)
        total_loss = (bass_loss + harmony_loss) / 2
        
        train_loss += total_loss.item()
        bass_train_loss += bass_loss.item()
        harmony_train_loss += harmony_loss.item()
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    train_loss /= len(train_dataloader)
    bass_train_loss /= len(train_dataloader)
    harmony_train_loss /= len(train_dataloader)
    
    return train_loss, bass_train_loss, harmony_train_loss


def validate(model, valid_dataloader, criterion):
    valid_loss = 0
    bass_valid_loss = 0
    harmony_valid_loss = 0
    
    model.eval()
    with torch.no_grad():
        for spec, bass_label, harmony_label in valid_dataloader:
            spec = spec.to(device)
            bass_label = bass_label.to(device)
            harmony_label = harmony_label.to(device)
    
            bass_out, harmony_out = model(spec)
            bass_loss = criterion(bass_out, bass_label)
            harmony_loss = criterion(harmony_out, harmony_label)
            total_loss = (bass_loss + harmony_loss) / 2
    
            valid_loss += total_loss.item()
            bass_valid_loss += bass_loss.item()
            harmony_valid_loss += harmony_loss.item()
    
    valid_loss /= len(valid_dataloader)
    bass_valid_loss /= len(valid_dataloader)
    harmony_valid_loss /= len(valid_dataloader)
    
    return valid_loss, bass_valid_loss, harmony_valid_loss


def plot_losses(train_losses, valid_losses, title, figname):
    plt.title(title)
    plt.plot(train_losses, label='train')
    plt.plot(valid_losses, label='valid')

    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')

    plt.legend()
    plt.savefig(figname)
    plt.close()


if __name__ == '__main__':
    fix_random_seeds()
    args = parse_command_line_arguments()

    experiment_folder = f'experiments/harmonybass_chroma_network_{args.dataset}/'
    os.makedirs(experiment_folder, exist_ok=True)

    datapath = os.path.join('datasets/', args.dataset)
    train_dataset = HarmonyBassChromaDataset(datapath, split='train')
    valid_dataset = HarmonyBassChromaDataset(datapath, split='valid')
    test_dataset = HarmonyBassChromaDataset(datapath, split='test')

    print('----------')
    print(f'Size of train dataset: {len(train_dataset)}')
    print(f'Size of valid dataset: {len(valid_dataset)}')
    print(f'Size of test dataset: {len(test_dataset)}')
    print('----------')

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = HarmonyBassChromaNetwork()
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    train_losses = []
    bass_train_losses = []
    harmony_train_losses = []
    
    valid_losses = []
    bass_valid_losses = []
    harmony_valid_losses = []

    best_valid_loss = torch.inf
    
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, bass_train_loss, harmony_train_loss = train(model, train_dataloader, criterion, optimizer)
        valid_loss, bass_valid_loss, harmony_valid_loss = validate(model, valid_dataloader, criterion)
    
        train_losses.append(train_loss)
        bass_train_losses.append(bass_train_loss)
        harmony_train_losses.append(harmony_train_loss)
    
        valid_losses.append(valid_loss)
        bass_valid_losses.append(bass_valid_loss)
        harmony_valid_losses.append(harmony_valid_loss)
    
        print(f'Epoch [{epoch}/{NUM_EPOCHS}] => train loss: {train_loss:.5f} -- valid loss: {valid_loss:.5f}')

        if valid_loss < best_valid_loss:
            print(f'Saving best valid loss model at epoch {epoch} [{best_valid_loss:.5f} -> {valid_loss:.5f}]')
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), os.path.join(experiment_folder, 'best_valid_model.pth'))

    # Saving train vs validation losses
    figname = os.path.join(experiment_folder, 'losses.png')
    plot_losses(train_losses, valid_losses, 'HarmonyBass Chroma Network Losses', figname)

    figname = os.path.join(experiment_folder, 'bass_losses.png')
    plot_losses(bass_train_losses, bass_valid_losses, 'HarmonyBass Chroma Network Losses (Bass Only)', figname)
    
    figname = os.path.join(experiment_folder, 'harmony_losses.png')
    plot_losses(harmony_train_losses, harmony_valid_losses, 'HarmonyBass Chroma Network Losses (Harmony Only)', figname)

    print('Training Finished!')
