# Brain Tumor Analysis

A Python-based brain MRI image classification project. It includes a TensorFlow/Keras
convolutional neural network, a Tkinter desktop interface, batch prediction, tumor
visualization, and PDF report generation.

> **Medical disclaimer:** This project is for education and research only. Its output is
> not a medical diagnosis and must not be used to make clinical decisions. Always consult
> a qualified healthcare professional.

## Features

- Classifies MRI images into:
  - `glioma_tumor`
  - `meningioma_tumor`
  - `no_tumor`
  - `pituitary_tumor`
- Loads either a `.keras` or `.h5` model.
- Provides single-image and batch analysis through a desktop GUI.
- Displays the predicted class and confidence score.
- Generates PDF reports in `AI_Reports/`.
- Produces tumor-region visualizations and a computer-generated growth simulation.

## Project files

| File | Description |
| --- | --- |
| `brain_tumor_gui.py` | Tkinter desktop application for interactive analysis |
| `model_training.py` | Loads training images, trains the CNN, evaluates it, and saves the model |
| `full_ai_report_generator.py` | Command-line batch prediction and PDF report generator |
| `Brain_Tumor_Detection_Model.keras` | Saved Keras model used by default |
| `Brain_Tumor_Detection_Model.h5` | Alternative saved model format |
| `brain tumor dataset/` | Training and testing image folders |
| `AI_Reports/` | Generated PDF reports |

## Requirements

- Python 3.9 or newer
- TensorFlow
- OpenCV
- NumPy
- Pillow
- Matplotlib
- ReportLab
- scikit-learn
- tqdm

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv venv
venv\Scripts\activate
python -m pip install tensorflow opencv-python numpy Pillow matplotlib reportlab scikit-learn tqdm
```

On Windows, Tkinter is normally included with the standard Python installer.

## Run the desktop application

From the project directory:

```bash
venv\Scripts\activate
python brain_tumor_gui.py
```

The application automatically looks for `Brain_Tumor_Detection_Model.keras` in the
project directory. Use **Browse Model** to select another `.keras` or `.h5` file.

Workflow:

1. Select an MRI image with **Select Image**.
2. Click **Process Image** to classify it.
3. Optionally click **Generate Report** to create a PDF.
4. Use **Batch Process** to process multiple images from a selected folder.

## Train the model

The training script expects the dataset in this layout:

```text
brain tumor dataset/
├── Training/
│   ├── glioma_tumor/
│   ├── meningioma_tumor/
│   ├── no_tumor/
│   └── pituitary_tumor/
└── Testing/
    ├── glioma_tumor/
    ├── meningioma_tumor/
    ├── no_tumor/
    └── pituitary_tumor/
```

Run:

```bash
python model_training.py
```

The script resizes images to `128x128`, normalizes pixel values, creates an 80/10/10
train/validation/test split, trains for 10 epochs, and saves
`Brain_Tumor_Detection_Model.keras`.

Before training on another computer, update `DATASET_PATH` in `model_training.py` to
the local `Training` directory. The current value is an absolute Windows path.

## Generate reports from the command line

For the dataset included in this repository, pass the actual test-folder name
explicitly:

```bash
python full_ai_report_generator.py `
  --model Brain_Tumor_Detection_Model.keras `
  --test-folder "brain tumor dataset\Testing" `
  --output AI_Reports `
  --batch
```

Useful options:

```text
--model       Path to a .keras or .h5 model
--test-folder Folder containing the four category folders
--output      Destination directory for PDF reports
--img-size    Model input size (default: 128)
--batch       Skip the confirmation prompt
```

The script writes processing logs to `tumor_detection.log`. The GUI writes logs to
`tumor_detection_gui.log`.

## Notes and limitations

- The classifier predicts one of four categories; confidence is the model's softmax
  score, not a clinically validated probability.
- The tumor segmentation and 30-day growth projection are algorithmic visualizations,
  not measurements of tumor size or future patient outcomes.
- The report generator's default test-folder value is `BrainTumorDataset/Testing`;
  use `--test-folder "brain tumor dataset\Testing"` for this repository.
- Model files can be large, and inference performance depends on the local TensorFlow
  and hardware installation.