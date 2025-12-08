# Google Colab Training Guide (SCUT-FBP5500 Dataset)

This guide explains how to train the **Beauty Scoring Model** using the SCUT-FBP5500 dataset on Google Colab.

## 1. Prepare Files
1.  **Training Script**: `d:\Looks Maximizer\train_scut.py`
2.  **Dataset**: Go to `d:\Looks Maximizer\ml_pipeline\Scut`.
    *   **Zip the folder** `SCUT-FBP5500_v2` into a file named `SCUT.zip`.
    *   (Uploading a single zip file is much faster than 5000 images).

## 2. Open Google Colab
1.  Go to [colab.research.google.com](https://colab.research.google.com/).
2.  Create a **New Notebook**.
3.  **Enable GPU**: Runtime > Change runtime type > T4 GPU.

## 3. Upload Files
1.  **Upload** `train_scut.py` to the main area.
2.  **Upload** your `SCUT.zip` file (or whatever you named it) to the main area.

## 4. Install Dependencies & Unzip
Run this in the first cell:

```python
!pip install torch torchvision opencv-python tqdm onnx onnxscript pandas

# Unzip the dataset (Replace SCUT.zip with your actual filename)
!unzip -q SCUT.zip -d .
```

## 5. Run Training
Run this in the second cell:

```python
# The unzip command usually creates a folder named SCUT-FBP5500_v2
# If your zip created a different folder, change the name below.
!python train_scut.py --data_dir SCUT-FBP5500_v2 --epochs 25
```

## 6. Download Model
Run this in a new cell to download your trained files:

```python
from google.colab import files

# Download the ONNX model (for the app)
files.download('scut_model.onnx')

# Download the PyTorch weights (for future training)
files.download('models_scut/best_model_scut.pth')
```
