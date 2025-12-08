# Looks Maximizer - Model Training on Google Colab

This notebook trains the multi-task face analysis model using the CelebA dataset.

## 1. Setup & Install Dependencies

```python
!pip install kaggle torch torchvision pandas opencv-python scikit-learn tqdm
```

## 2. Download Dataset (Kaggle)

**IMPORTANT**: You need to upload your `kaggle.json` file to the Colab runtime before running this cell.

```python
import os
from google.colab import files

# Upload kaggle.json
if not os.path.exists('kaggle.json'):
    print("Please upload your kaggle.json file")
    files.upload()

# Setup Kaggle Config
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Download CelebA
!kaggle datasets download -d jessicali9530/celeba-dataset
!unzip -q celeba-dataset.zip -d celeba
```

## 3. Training Script

Copy and paste the entire content of `train_celeba.py` into a cell here, or upload the file.

If pasting, make sure to adjust the `dataset_path` argument in the `main` function call or CLI arguments.

```python
# ... [Paste content of train_celeba.py here] ...
```

## 4. Run Training

```python
# Run the script directly if you pasted it into a file named train_celeba.py
!python train_celeba.py --dataset_path celeba --epochs 10
```

## 5. Download Trained Model

```python
from google.colab import files
files.download('export/celeba_multitask.onnx')
files.download('models/best_model.pth')
```
