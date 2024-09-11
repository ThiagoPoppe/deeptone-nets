import os
import json
import torch
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from collections import defaultdict
from torch.utils.data import DataLoader

from source.models.dtns import DeepToneNets
from source.loss import MultiTaskCrossEntropyLoss
from source.data.datasets import FunctionalHarmonyDataset
from source.data.constants import RANDOM_SEED, LABEL_NAMES, LABEL_SIZES

BATCH_SIZE = 128
NUM_EPOCHS = 100
WEIGHT_DECAY = 1e-4
LEARNING_RATE = 1e-4

device = torch.device('cuda' if torch.cuda.is_availabel() else 'cpu')
print('Using device:', device)


def parse_command_line_arguments():
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name to train model (must exist within datasets folder)')
    parser.add_argument('--use-harmony-only-chroma', action='store_true', help='Flag to only use harmony related chroma features')
    parser.add_argument('--no-keysim-bias', action='store_true', help='Flag to deactivate the usage of key similarity bias method')

    args = parser.parse_args()
    return args


def fix_random_seeds():  
    """
    Auxiliary function to fix seed and ensure reproducibility.
    """
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)


if __name__ == '__main__':
    fix_random_seeds()
    args = parse_command_line_arguments()

    experiment_folder = f'experiments/harmonybass_chroma_network_{args.dataset}/'
    os.makedirs(experiment_folder, exist_ok=True)


def train(model, train_dataloader, criterion, optimizer, use_keysim_bias):
    train_loss = 0
    
    model.train()
    for features, targets_indices, targets_one_hot, tivs_information in train_dataloader:
        features = features.to(device)
        targets_indices = targets_indices.to(device)
        targets_one_hot = [target.to(device) for target in targets_one_hot]

        if use_keysim_bias:
            key_similarities = tivs_information['harmonybass_key_similarities'].to(device)
            outputs = model(features, is_train=True, key_similarities=key_similarities, targets=targets_one_hot)
        else:
            outputs = model(features, is_train=True, targets=targets_one_hot)
        
        loss = criterion(outputs, targets_indices)
        train_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(train_dataloader)
    return train_loss


def validate(model, valid_dataloader, criterion, use_keysim_bias):
    valid_loss = 0
    
    model.eval()
    with torch.no_grad():
        for features, targets_indices, targets_one_hot, tivs_information in valid_dataloader:
            features = features.to(device)
            targets_indices = targets_indices.to(device)

            if use_keysim_bias:
                key_similarities = tivs_information['harmonybass_key_similarities'].to(device)
                outputs = model(features, is_train=False, key_similarities=key_similarities)
            else:
                outputs = model(features, is_train=False)

            loss = criterion(outputs, targets_indices)
            valid_loss += loss.item()
    
    valid_loss /= len(valid_dataloader)    
    return valid_loss


def compute_accuracies(model, dataloader, use_keysim_bias):
    accs = defaultdict(int)

    model.eval()
    with torch.no_grad():
        for features, label_indices, targets_one_hot, tivs_information in dataloader:
            features = features.to(device)
            label_indices = label_indices.to(device)

            if use_keysim_bias:
                key_similarities = tivs_information['harmonybass_key_similarities'].to(device)
                outputs = model(features, is_train=False, key_similarities=key_similarities)
            else:
                outputs = model(features, is_train=False)
            
            padding_idxs = label_indices[:, 0] == -1
            labels = {LABEL_NAMES[i]: label_indices[:, i] for i in range(len(outputs))}
            preds = {LABEL_NAMES[i]: outputs[i].argmax(dim=-1) for i in range(len(outputs))}
            
            for name in LABEL_NAMES:
                correct = (preds[name][~padding_idxs] == labels[name][~padding_idxs]).sum()
                accs[name] += (correct / labels[name][~padding_idxs].numel())
                
            # Computing RN analysis accuracy w/o key
            correct = None
            for name in ['secondary degree', 'primary degree', 'triad quality', 'inversion']:
                padding_idxs = labels[name] == -1

                if correct is None:
                    correct = (preds[name][~padding_idxs] == labels[name][~padding_idxs])
                else:
                    correct &= (preds[name][~padding_idxs] == labels[name][~padding_idxs])

            accs['RN w/o key'] += (correct.sum() / labels['key'][~padding_idxs].numel())

            # Computing RN analysis accuracy with key
            padding_idxs = labels['key'] == -1
            correct &= (preds['key'][~padding_idxs] == labels['key'][~padding_idxs])
            accs['RN with key'] += (correct.sum() / labels['key'][~padding_idxs].numel())

    accs = {key: value.item() / len(dataloader) for key, value in accs.items()}
    return accs


if __name__ == '__main__':
    fix_random_seeds()
    args = parse_command_line_arguments()

    use_keysim_bias = (args.no_keysim_bias == False)
    use_harmonybass_chroma = (args.use_harmony_only_chroma == False)

    suffixes = 'harmonybass' if use_harmonybass_chroma else 'harmony' + '_' + \
               'keysim' if use_keysim_bias else ''

    experiment_folder = f'experiments/dtns_{suffixes}_{args.dataset}/'
    os.makedirs(experiment_folder, exist_ok=True)

    datapath = os.path.join('datasets/', args.dataset)
    train_dataset = FunctionalHarmonyDataset(datapath, split='train', use_harmony_only_chroma=args.use_harmony_only_chroma)
    valid_dataset = FunctionalHarmonyDataset(datapath, split='valid', use_harmony_only_chroma=args.use_harmony_only_chroma)
    test_dataset = FunctionalHarmonyDataset(datapath, split='test', use_harmony_only_chroma=args.use_harmony_only_chroma)

    print('----------')
    print(f'Size of train dataset: {len(train_dataset)}')
    print(f'Size of valid dataset: {len(valid_dataset)}')
    print(f'Size of test dataset: {len(test_dataset)}')
    print('----------')

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    in_features = 12 if args.use_harmony_only_chroma else 24
    model = DeepToneNets(in_features=in_features, label_sizes=LABEL_SIZES)
    model = model.to(device)

    criterion = MultiTaskCrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    train_losses = []
    valid_losses = []
    best_valid_loss = torch.inf

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train(model, train_dataloader, criterion, optimizer, use_keysim_bias=use_keysim_bias)
        valid_loss = validate(model, valid_dataloader, criterion, use_keysim_bias=use_keysim_bias)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print(f'Epoch [{epoch}/{NUM_EPOCHS}] => train loss: {train_loss:.5f} -- valid loss: {valid_loss:.5f}')

        if valid_loss < best_valid_loss:
            print(f'Saving best valid loss model at epoch {epoch} [{best_valid_loss:.5f} -> {valid_loss:.5f}]')
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), os.path.join(experiment_folder, 'best_valid_model.pth'))

    # Saving train vs validation losses
    plt.title(f'dtns_{suffixes}_{args.dataset} losses')
    plt.plot(train_losses, label='train')
    plt.plot(valid_losses, label='valid')

    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')

    plt.legend()
    plt.savefig(os.path.join(experiment_folder, 'losses.png'))
    
    # Loading best validation model
    modelpath = os.path.join(experiment_folder, 'best_valid_model.pth')
    model.load_state_dict(torch.load(modelpath))

    # Computing accuracies and saving JSON file
    print('Computing accuracies...')
    train_accs = compute_accuracies(model, train_dataloader, use_keysim_bias=use_keysim_bias)
    valid_accs = compute_accuracies(model, valid_dataloader, use_keysim_bias=use_keysim_bias)
    test_accs = compute_accuracies(model, test_dataloader, use_keysim_bias=use_keysim_bias)

    with open(os.path.join(experiment_folder, 'accuracies.json'), 'w') as fp:
        accuracies = {'train': train_accs, 'valid': valid_accs, 'test': test_accs}
        json.dump(accuracies, fp=fp, indent=2)

    print('Training Finished!')
