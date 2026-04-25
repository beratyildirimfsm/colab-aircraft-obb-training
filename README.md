# Gazebo Aircraft OBB Fine-Tune

This repository contains a Google Colab notebook and a trained model for fine-tuning an aircraft detector on a Gazebo-based dataset.

## Project Goal

The goal of this project is to fine-tune a pretrained Ultralytics YOLO OBB model on a custom Gazebo aircraft dataset containing roughly 7000 images.

This repository is focused on:

- transfer learning / fine-tuning
- Gazebo aircraft detection
- Google Colab training workflow
- local inference with the trained model

## What Is Included

- `colab_gazebo_obb_finetune.ipynb`: Google Colab notebook for training, validation, prediction, and manual image testing
- `models/best_aircraft_obb.pt`: trained OBB aircraft model

## Model Choice

This project uses:

```text
yolo11l-obb.pt
```

Why this model was chosen:

- the dataset uses OBB style labels
- the target task is oriented object detection
- a larger model was preferred for stronger performance
- the training environment uses a strong GPU, so a larger backbone is practical

## Repository Structure

```text
.
├── README.md
├── colab_gazebo_obb_finetune.ipynb
└── models
    └── best_aircraft_obb.pt
```

## Open In Google Colab

After pushing this repository to GitHub, the notebook can be opened with:

```text
https://colab.research.google.com/github/beratyildirimfsm/colab-aircraft-obb-training/blob/main/colab_gazebo_obb_finetune.ipynb
```

If the repository is private, only invited collaborators can open it.

## Dataset Location Used By The Notebook

The notebook expects the prepared dataset here:

```text
/content/drive/MyDrive/pnpegitme/Data/merged_dataset
```

It also saves outputs to:

```text
/content/drive/MyDrive/pnpegitme/Data/models
/content/drive/MyDrive/pnpegitme/Data/training_outputs
/content/drive/MyDrive/pnpegitme/Data/manual_tests
```

## Part 1: How To Run The Notebook In Colab

### Step 1. Open Colab

Open:

```text
https://colab.research.google.com/
```

Then either:

- open the GitHub notebook link
- or upload `colab_gazebo_obb_finetune.ipynb` manually

### Step 2. Enable GPU

In Colab:

1. click `Runtime`
2. click `Change runtime type`
3. select `GPU`
4. save

### Step 3. Mount Google Drive

The notebook already contains the correct mount cell:

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 4. Make Sure The Dataset Exists

Expected path:

```text
/content/drive/MyDrive/pnpegitme/Data/merged_dataset
```

The notebook checks:

- whether the dataset exists
- whether `data.yaml` exists
- whether the dataset is copied into Colab local storage

### Step 5. Run Fine-Tuning

The notebook uses the following training flow:

```python
from ultralytics import YOLO

model = YOLO("yolo11l-obb.pt")

results = model.train(
    data="/content/merged_dataset/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    device=0,
    pretrained=True,
    project="/content/yolo_runs",
    name="gazebo_aircraft_obb_finetune"
)
```

If GPU memory becomes a problem, the notebook also includes a backup training cell with:

```python
batch=8
```

### Step 6. Validate The Model

Validation is included in the notebook:

```python
metrics = model.val(data=str(DATA_YAML), split='val')
print(metrics)
```

### Step 7. Test On The Dataset Test Split

The notebook predicts on:

```text
/content/merged_dataset/images/test
```

with:

```python
pred = model.predict(
    source='/content/merged_dataset/images/test',
    conf=0.25,
    save=True,
    project='/content/yolo_runs',
    name='gazebo_test_preds'
)
```

### Step 8. Save The Trained Model To Google Drive

The notebook saves:

- `best.pt`
- `last.pt`
- the whole training output folder

into the configured Google Drive directories.

### Step 9. Test With Your Own Image

The notebook also supports:

1. uploading a custom image from your computer
2. loading the trained model
3. running prediction
4. displaying the saved result
5. saving the result back to Google Drive

## Part 2: How To Use The Trained Model On Your Computer

This section explains how another user can run the trained model locally.

## Requirements

- Python 3.10 or newer
- Git
- pip

Optional:

- NVIDIA GPU

## Step 1. Install Git

### Ubuntu / Debian

```bash
sudo apt update
sudo apt install -y git
```

### Windows

Download Git from:

```text
https://git-scm.com/downloads
```

### macOS

```bash
brew install git
```

## Step 2. Install Python

Download from:

```text
https://www.python.org/downloads/
```

Check installation:

```bash
python3 --version
pip --version
```

## Step 3. Clone The Repository

```bash
git clone https://github.com/beratyildirimfsm/colab-aircraft-obb-training.git
cd colab-aircraft-obb-training
```

If the repository is private, the user must be invited first.

## Step 4. Create A Virtual Environment

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

## Step 5. Install Required Packages

```bash
pip install --upgrade pip
pip install ultralytics opencv-python pillow
```

## Step 6. Run Inference On A Single Image

```bash
yolo predict model=models/best_aircraft_obb.pt source=path/to/image.jpg conf=0.25
```

Prediction outputs are usually saved inside:

```text
runs/obb/predict
```

## Step 7. Run Inference On A Folder

```bash
yolo predict model=models/best_aircraft_obb.pt source=path/to/folder conf=0.25
```

## Step 8. Use The Model From Python

```python
from ultralytics import YOLO

model = YOLO("models/best_aircraft_obb.pt")

results = model.predict(
    source="path/to/image.jpg",
    conf=0.25,
    save=True
)

print(results)
```

## Step 9. Validate If You Have A Matching Dataset

```bash
yolo val model=models/best_aircraft_obb.pt data=path/to/data.yaml
```

## What This Project Solves

This project is useful when:

- a base aircraft detector needs to be adapted to Gazebo imagery
- a synthetic aircraft dataset is available
- transfer learning is preferred over training completely from scratch
- Google Colab is used for a practical notebook-driven workflow

## Notes

- This repository includes trained model weights directly.
- The dataset itself is not included in this repository.
- This project uses OBB detection, not standard axis-aligned bbox detection.
- The notebook is intended for Colab and Google Drive based execution.
