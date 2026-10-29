# COCO Dataset Creation Tool 🧩

This project provides a Python utility for creating and managing datasets in **COCO (Common Objects in Context)** format.  
It includes functionality to generate COCO-style annotations from custom datasets and supports automated testing using **pytest**.

---

## 📂 Project Structure
```
Wild-Life Detection Project/
│
├── coco_dataset_creation/
│── coco_dataset_generator.py
│
├── tests/
│ └── test_coco_dataset_generator.py
│
├── .gitignore
├── requirements.txt
└── README.md
```



## ⚙️ Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/HarishSingaravelan/DSCI-601-Wildlife.git
   cd coco-dataset-creation

2. (Optional) Create a virtual environment:
    ```
    conda create -n dsci601 python=3.10
    conda activate dsci601
    ```

3. Install dependencies:

    `pip install -r requirements.txt`

## 🚀 Usage

The main script for generating a COCO-style dataset is:
```coco_dataset_creation/coco_dataset_generator.py```

Example usage inside Python:

from coco_dataset_creation.coco_dataset_generator import generate_coco_json

#### Example paths
```image_dir = "path/to/images"
output_json = "path/to/output/coco_annotations.json"

generate_coco_json(image_dir, output_json) 
```

This will generate a COCO-style JSON annotation file based on your input dataset.

## 🧪 Running Tests

All unit tests are located in the tests/ directory.
To run tests, execute:

`pytest -v`

## 🧱 Features

- ✅ Generate COCO-style annotation JSON automatically

- 🖼️ Supports multiple image formats

- 🧪 Includes Pytest-based unit tests

- 🧰 Modular design for dataset manipulation and extension

## 🧾 Example COCO JSON Output
```.json
{
  "images": [
    {"id": 1, "file_name": "image1.jpg", "height": 480, "width": 640}
  ],
  "annotations": [
    {"id": 1, "image_id": 1, "category_id": 3, "bbox": [100, 120, 50, 60], "area": 3000}
  ],
  "categories": [
    {"id": 3, "name": "person"}
  ]
}
```
## 🚀 Data Splitting and Execution

The final part of your utility is the script used to split the generated COCO dataset into training, validation, and test subsets based on specified ratios.

The script is assumed to be located at: `data/split_dataset.py`

### 🖥️ Execution

Use the following commands to execute the splitting script, modifying the path arguments (`--source_root`, `--annotation_file`, etc.) to match your specific environment.

#### 1. Linux / macOS (Bash)

Use the backslash (`\`) to continue the command across multiple lines for better readability.
Here the location file is for the Remote system. Add you local system or replace it with windows folder location

```bash
python split_dataset.py \
  --source_root "../../../../shared/rc/turbine/resized_dataset_new/Test_2021_SY" \
  --annotation_file "../../../../shared/rc/turbine/resized_dataset_new/coco_annotations_new.json" \
  --output_root "../../../../shared/rc/turbine/resized_dataset_new/turbine_split_stratified" \
  --min_images 3 \
  --bucket_name "unknown" \
  --train_ratio 0.7 \
  --val_ratio 0.15 \
  --test_ratio 0.15
```

#### 2. Windows (PowerShell or Command Prompt)

Use the backtick (`` ` ``) for line continuation in PowerShell, or combine the arguments onto a single line for Command Prompt.

**PowerShell:**

```powershell
python split_dataset_stratified.py `
  --source_root "../../../../shared/rc/turbine/resized_dataset_new/Test_2021_SY" `
  --annotation_file "../../../../shared/rc/turbine/resized_dataset_new/coco_annotations_new.json" `
  --output_root "../../../../shared/rc/turbine/resized_dataset_new/turbine_split_stratified" `
  --min_images 3 `
  --bucket_name "unknown" `
  --train_ratio 0.8 `
  --val_ratio 0.1 `
  --test_ratio 0.1 `
```

**Command Prompt:**

```cmd
python data/split_dataset.py --source_root . --annotation_file coco_annotations.json --output_root turbine_split --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15
```

### 📚 Argument Descriptions

| Argument | Description | Example Value |
|----------|-------------|---------------|
| `--source_root` | The root directory containing all image files referenced in the annotation file. | `../../../shared/rc/turbine` |
| `--annotation_file` | The path to the complete COCO-style JSON annotation file. | `coco_annotations_updated.json` |
| `--output_root` | The directory where the `train/`, `val/`, and `test/` annotation files and image symlinks will be saved. | `turbine_split` |
| `--train_ratio` | The fraction of the dataset to allocate for training. | `0.7` |
| `--val_ratio` | The fraction of the dataset to allocate for validation. | `0.15` |
| `--test_ratio` | The fraction of the dataset to allocate for testing. | `0.15` |

### 📝 Notes

- Ensure that the ratios (`--train_ratio`, `--val_ratio`, `--test_ratio`) sum to 1.0.
- The script will create subdirectories within `--output_root` for each split (train, val, test).
- Image files are typically symlinked rather than copied to save disk space.
