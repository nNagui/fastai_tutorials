
# Bird vs. Forest Image Classifier

A simple image classification project built with [fast.ai](https://docs.fast.ai/) and a pretrained ResNet18 model.

The goal of this project is to build a binary image classifier that can distinguish between:

- Bird images
- Forest images

The project demonstrates the basic fast.ai workflow for image classification, including dataset collection, image validation, data preparation, transfer learning, model training, and inference.

---

## Project Structure

```text
1_bird-forest-classifier/
│
├── README.md
├── requirements.txt
│
├── src/
│   └── classifier.py
│
├── data/
│   └── .gitkeep
│
└── models/
    └── .gitkeep
````

### Directory description

| Path                | Description                                            |
| ------------------- | ------------------------------------------------------ |
| `README.md`         | Documentation for this project                         |
| `requirements.txt`  | Python dependencies required to run the project        |
| `src/classifier.py` | Main Python script containing the complete ML pipeline |
| `data/`             | Downloaded and processed image dataset                 |
| `models/`           | Trained model artifacts                                |

The `data/` and `models/` directories are intentionally kept in the repository structure, but generated files inside them are ignored by Git.

---

## What This Project Does

The script performs the following pipeline:

```text
Image Search
     │
     ▼
Download Images
     │
     ▼
Organize Dataset
     │
     ▼
Validate Images
     │
     ▼
Resize Images
     │
     ▼
Create fast.ai DataLoaders
     │
     ▼
Train ResNet18
     │
     ▼
Fine-tune Model
     │
     ▼
Predict New Image
```

---

## Technologies Used

* Python
* fast.ai
* PyTorch (through fast.ai)
* ResNet18
* DDGS for image search
* FastDownload for downloading images

fast.ai provides the high-level API used to prepare the dataset, create the DataLoaders, train the model, and perform inference.

---

## Requirements

* Python 3.9+
* Internet connection
* A machine capable of running PyTorch/fast.ai

A GPU is **not required** for this small example, although training will generally be faster with one.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/nNagui/fastai_tutorials.git
```

Navigate to this project:

```bash
cd fastai_tutorials/1_bird-forest-classifier
```

It is recommended to create a virtual environment:

```bash
python -m venv .venv
```

Activate it.

### Linux / macOS

```bash
source .venv/bin/activate
```

### Windows

```powershell
.venv\Scripts\activate
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

From the `1_bird-forest-classifier` directory:

```bash
python src/classifier.py
```

The script will:

1. Search for bird and forest images.
2. Download images for the training dataset.
3. Validate the downloaded images.
4. Remove invalid images.
5. Resize the images.
6. Create fast.ai DataLoaders.
7. Create a pretrained ResNet18 classifier.
8. Fine-tune the model.
9. Download example bird and forest images.
10. Run a prediction on the example bird image.

---

## Dataset

The dataset is generated automatically using image search queries.

The resulting dataset follows this structure:

```text
data/
└── bird_or_not/
    ├── bird/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    │
    └── forest/
        ├── image1.jpg
        ├── image2.jpg
        └── ...
```

The parent directory is used as the image label:

```text
bird_or_not/bird/image.jpg
              │
              └── label = bird
```

and:

```text
bird_or_not/forest/image.jpg
              │
              └── label = forest
```

### Important

The images are downloaded automatically from the internet and are **not committed to this repository**.

The downloaded images may be subject to their respective copyright and licensing conditions. This project is intended for educational purposes.

---

## Model

The classifier uses **ResNet18** with transfer learning.

Instead of training a convolutional neural network completely from scratch, the project starts with a ResNet18 model pretrained on ImageNet.

fast.ai then adapts the final classification layer to the two classes in this project:

```text
bird
forest
```

The model is fine-tuned for a small number of epochs.

---

## Data Splitting

The dataset is divided into:

```text
80% → Training
20% → Validation
```

A fixed random seed is used so that the split is reproducible.

```python
RandomSplitter(
    valid_pct=0.2,
    seed=42,
)
```

---

## Image Preprocessing

Images are resized before being passed to the model.

This reduces computational requirements and ensures that the model receives consistently sized inputs.

---

## Image Validation

Images downloaded from internet searches are not guaranteed to be valid image files.

The project uses fast.ai's image verification functionality to identify files that cannot be opened correctly.

Invalid images are removed before training.

---

## Example Output

A successful run produces output similar to:

```text
Prediction for: bird.jpg
Class: bird

Class probabilities:
  bird: 98.23%
  forest: 1.77%
```

The exact probabilities will vary depending on the images downloaded and the training environment.

---

## Limitations

This is an educational example rather than a production-ready classifier.

Potential limitations include:

* The dataset is collected automatically from internet image searches.
* Image quality and relevance are not guaranteed.
* The dataset may contain mislabeled images.
* The number of training images can vary between runs.
* Training results are therefore not necessarily reproducible across different runs.
* The model is trained for only a small number of epochs.
* The classifier only recognizes the two categories used during training.
* A bird image containing a forest may still be classified according to the visual patterns learned by the model.

---

## Possible Improvements

Some potential improvements include:

* Use a manually curated dataset.
* Increase the dataset size.
* Add data augmentation.
* Experiment with different pretrained architectures.
* Tune the learning rate.
* Increase the number of training epochs.
* Add confusion matrices and additional evaluation metrics.
* Save and export the trained model.
* Add a command-line interface for training and prediction.
* Separate data collection, training, and inference into independent commands.
* Add automated tests.
* Add experiment tracking.

---

## Learning Objectives

This project demonstrates several fundamental concepts in deep learning:

1. Collecting an image dataset.
2. Organizing images by class.
3. Validating downloaded data.
4. Preparing image DataLoaders.
5. Using transfer learning.
6. Fine-tuning a pretrained CNN.
7. Evaluating predictions.
8. Performing inference on new images.

---

## References

* [fast.ai Book](https://course.fast.ai/Resources/book.html)
* [ResNet Paper](https://arxiv.org/abs/1512.03385)

---

## License

This project is part of my personal fast.ai learning/tutorial repository.

See the repository-level license for the terms that apply to this project.

---