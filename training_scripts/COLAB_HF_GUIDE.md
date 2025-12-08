# Google Colab Training Guide (Hugging Face Dataset)

This guide explains how to train the Age/Gender/Race model using your Parquet dataset on Google Colab.

## 1. Prepare Files
You need the following files from your local computer:
1.  **Training Script**: `d:\Looks Maximizer\train_hf.py`
2.  **Dataset Files**: All files inside `d:\Looks Maximizer\ml_pipeline\HuggingFace\0.25\`
    *   `train-00000-of-00002-....parquet`
    *   `train-00001-of-00002-....parquet`
    *   `validation-00000-of-00001-....parquet`

## 2. Open Google Colab
1.  Go to [colab.research.google.com](https://colab.research.google.com/).
2.  Create a **New Notebook**.
3.  **Enable GPU**: Runtime > Change runtime type > T4 GPU.

## 3. Upload Files (Direct Upload)
1.  **Upload** `train_hf.py` to the main area.
2.  **Upload** ALL your `.parquet` files (from both `0.25` and `1.25` folders) directly to the main area.
    *   Do not put them in folders. Just drag and drop them all.

## 4. Install Dependencies
Run this in a new cell:

```python
!pip install pandas pyarrow fastparquet torch torchvision opencv-python tqdm onnx onnxscript
```

## 5. Run Training
Since your files are in the main area (root), use `.` as the data directory:

```python
!python train_hf.py --data_dir . --epochs 25
```

## 6. Download Model
After training finishes, download the model:

```python
from google.colab import files
files.download('hf_model.onnx')
files.download('models_hf/best_model_hf.pth')
```
