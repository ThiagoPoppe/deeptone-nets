# DeepToneNets
Repository for DeepToneNets, a deep learning model for functional harmony recognition directly on audio.

## Requirements

Make sure to install the required dependencies specified in the `environment.yml` file, or set up a suitable Python environment using conda or pip.

### Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/ThiagoPoppe/deeptone-nets.git
   cd deeptone-nets
   ```

2. Create and activate the conda environment (anaconda required):
   ```bash
   conda env create -f environment.yml
   conda activate dtns
   ```

### Defining data constants
Under `source/data/constants.py` you can change `AUDIOS_DATAPATH` and `PROCESSED_HOOKTHEORY_DATAPATH ` values to reflect the datapath in which your audios are stored (needs to be in H5PY format) and the processed HookTheory JSON file.

## Model pre-trained weights

The model weights with results reported in the paper can be found at `pretrained/` folder.

```python
# Loading example using PyTorch
model = HarmonyBassChromaNetworks()
model.load_state_dict(torch.load('pretrained/harmonybass_chroma_networks_weights.pth'))
```

## Dataset Creation
In order to train your own models, you need to create a dataset for each specific task with the `create_harmonybass_dataset.py` and `create_functional_harmony_dataset.py` scripts. Next, we give a usage example for each one and their command line arguments.

### HarmonyBass Chroma Network (HBCN)

```bash
# Usage example
python create_harmonybass_dataset.py --artist <ARTIST> [--no-split]
```

| Parameter         | Type     | Requirement | Description                                     |
|-------------------|----------|-------------|-------------------------------------------------|
| `--artist`        | `str`    | Required    | The artist to create the dataset for (e.g. `top100`, `classical`, `the-beatles`) |
| `--no-split`      | `flag`   | Optional    | Flag to avoid splitting the dataset into train/valid/test. |

### DeepToneNets (DTNs)

```bash
# Usage example
python create_functional_harmony_dataset.py --artist <ARTIST> [--no-split]
```

| Parameter         | Type     | Requirement | Description                                     |
|-------------------|----------|-------------|-------------------------------------------------|
| `--artist`        | `str`    | Required    | The artist to create the dataset for (e.g. `top100`, `classical`, `the-beatles`) |
| `--hbcn-checkpoint` | `str`    | Required    | Path to the HBCN checkpoint file with the weights |
| `--no-split`      | `flag`   | Optional    | Flag to avoid splitting the dataset into train/valid/test. |

## Training Models
### HarmonyBass Chroma Network (HBCN)
You can re-train the HarmonyBass Chroma Network (HBCN) using the `train_hbcn.py` script. This script only requires the dataset name as an argument.

```bash
# Usage example
python train_hbcn.py --dataset <DATASET_NAME>
```

| Parameter         | Type     | Requirement | Description                                     |
|-------------------|----------|-------------|-------------------------------------------------|
| `--dataset`       | `str`    | Required    | Name of the dataset to train the model on (must exist within datasets folder)|


### DeepToneNets (DTNs)

You can also re-train the DeepToneNets (DTNS) model using the `train_dtns.py` script. This script allows you to specify the dataset and variations of the DTNs for training.

```bash
# Usage example
python train_dtns.py --dataset <DATASET_NAME> [--use-harmony-only-chroma] [--no-keysim-bias]
```

| Parameter         | Type     | Requirement | Description                                     |
|-------------------|----------|-------------|-------------------------------------------------|
| `--dataset`       | `str`    | Required    | Name of the dataset to train the model on        |
| `--use-harmony-only-chroma` | `flag`   | Optional    | Flag to use only harmony-related chroma features |
| `--no-keysim-bias`| `flag`   | Optional    | Disable key similarity bias during training      |
