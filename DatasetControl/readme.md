# DatasetControl

A Python tool for loading, preprocessing, balancing, and exporting physical image datasets stored in `.npz` format. It is designed to streamline Machine Learning workflows, supporting direct export to PyTorch `DataLoaders`, TensorFlow `tf.data.Dataset`, and standard NumPy arrays.

---

## Key Features

* **Efficient Loading:** Extracts raw images and physical parameters (`Temperature`, `Jex2`, `KDM`) directly from `.npz` files.
* **Domain Filtering:** Allows selecting specific data subsets (`kdm`, `jex`, or `unified`).
* **Data Balancing:** Uses `MiniBatchKMeans` to extract intelligently balanced subsets based on parameter distribution.
* **Preprocessing Pipeline:** Automates image resizing to 224x224 (RGB) and scales parameters using `MinMaxScaler`.
* **Multi-Framework Support:** Exports data ready for training in **PyTorch**, **TensorFlow**, or as flat **NumPy** arrays.

---

## Requirements

Ensure you have the following dependencies installed in your environment:

* `numpy`
* `scikit-learn`
* `tensorflow`
* `torch`

---

## Basic Workflow (Usage)

Using `DatasetControl` follows a strict sequential pipeline: initialize, load, split, preprocess, and export.

```python
from DatasetControl import DatasetControl

# 1. Initialize and load
dataset = DatasetControl('path/to/your/file.npz')
dataset.load()

# 2. (Optional) Filter and view summary
dataset.select('unified')
dataset.summary()

# 3. (Optional) Get a balanced subset
subset = dataset.get_balanced_subset(n_samples=10000)

# 4. Split into Train, Validation, Test
subset.split(val_size=0.15, test_size=0.15)

# 5. Preprocess images and parameters
subset.preprocess_images()
subset.preprocess_params(scaler_type='minmax')

# 6. Export to your framework of choice 
train_loader, val_loader, test_loader = subset.get_loaders_torch(batch_size=32) #(pytorch example)
train_ds, val_ds, test_ds = subset.get_loaders_tf(batch_size=32) #(tf example)
```
---

## Importing the Dataset from Kaggle (Google Colab)

If you are working in Google Colab, you can seamlessly download the dataset directly from Kaggle. 

**Prerequisite:** Ensure you have uploaded your Kaggle API token (`kaggle.json`) to the root `/content/` directory in Colab. Then, run the following snippet:

```python
import os, shutil, json
from DatasetControl import DatasetControl

# Place kaggle.json BEFORE importing kaggle
kaggle_dir = os.path.expanduser('~/.config/kaggle')
os.makedirs(kaggle_dir, exist_ok=True)
shutil.copy('/content/kaggle.json', f'{kaggle_dir}/kaggle.json')
os.chmod(f'{kaggle_dir}/kaggle.json', 0o600)

with open(f'{kaggle_dir}/kaggle.json') as f:
    creds = json.load(f)
assert 'username' in creds and 'key' in creds, \
    'kaggle.json must contain username and key'
print(f'Credentials OK — user: {creds["username"]}')

os.environ['KAGGLE_USERNAME'] = creds['username']
os.environ['KAGGLE_KEY']      = creds['key']

# Download and unzip the dataset
!kaggle datasets download \
    -d carloscanamejoy/dataset-spines-united \
    -p /content/ --unzip

files = os.listdir('/content/')
print('Files in /content/:', files)
assert 'dataset-united.npz' in files, 'dataset-united.npz not found'
print('dataset-united.npz ready.')

# Initialize DatasetControl with the downloaded file
ds = DatasetControl('/content/dataset-united.npz')
ds.load()
```
---