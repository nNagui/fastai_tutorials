# fast.ai Tutorials

A collection of hands-on tutorials and projects for learning **fast.ai**, **PyTorch**, and practical deep learning.

The repository is organized as a series of independent projects. Each project focuses on a specific concept or technique and contains its own documentation, dependencies, source code, data directory, and model directory.

---

## Repository Structure

```text
fastai_tutorials/
│
├── .gitignore
├── LICENSE
├── README.md
│
├── 1_bird-forest-classifier/
│   ├── README.md
│   ├── requirements.txt
│   ├── src/
│   │   └── classifier.py
│   ├── data/
│   │   └── .gitkeep
│   └── models/
│       └── .gitkeep
│
├── 2_...
│   ├── README.md
│   ├── requirements.txt
│   ├── src/
│   ├── data/
│   └── models/
│
└── ...
````

Each numbered directory represents a standalone learning project.

---

## Projects

| # | Project                                                         | Main Concepts                                     | Status         |
| - | --------------------------------------------------------------- | ------------------------------------------------- | -------------- |
| 1 | [Bird vs. Forest Image Classifier](./1_bird-forest-classifier/) | Image classification, transfer learning, ResNet18 | In progress |

More projects will be added as the learning journey progresses.

---

## Learning Goals

The main goal of this repository is to build a practical understanding of modern deep learning through small, focused projects.

Topics covered or planned include:

* fast.ai fundamentals
* PyTorch fundamentals
* Computer vision
* Image classification
* Transfer learning
* Convolutional neural networks
* Data preparation
* Data augmentation
* Model training
* Model evaluation
* Inference
* Model export and deployment
* Natural language processing
* Tabular data
* Recommendation systems
* Experiment tracking
* Practical machine learning workflows

---

## Project Philosophy

The projects in this repository are intentionally kept relatively small.

The goal is not to build production-ready systems immediately, but to understand the underlying concepts by implementing complete end-to-end workflows.

Each project generally follows a structure similar to:

```text
Data
  │
  ▼
Preparation
  │
  ▼
Training
  │
  ▼
Evaluation
  │
  ▼
Inference
```

As the projects become more advanced, the implementations will gradually introduce better engineering practices such as:

* Reproducible environments
* Configuration management
* Better dataset management
* Model persistence
* Testing
* Logging
* Command-line interfaces
* Experiment tracking
* Deployment

---

## Getting Started

Clone the repository:

```bash
git clone https://github.com/nNagui/fastai_tutorials.git
```

Navigate to the project you want to explore:

```bash
cd fastai_tutorials/1_bird-forest-classifier
```

Each project contains its own `README.md` and `requirements.txt`.

Install the dependencies for the selected project:

```bash
pip install -r requirements.txt
```

Then follow the instructions in that project's README.

---

## Environment Recommendation

It is recommended to use a separate Python virtual environment for each project.

For example:

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

Then install the project's dependencies:

```bash
pip install -r requirements.txt
```

---

## Why Separate Requirements Files?

Each project has its own `requirements.txt` intentionally.

Different tutorials may use:

* Different fast.ai versions
* Different Python packages
* Different libraries
* Different model architectures
* Different experimental dependencies

Keeping dependencies at the project level makes each tutorial more independent and easier to reproduce.

---

## Data and Models

Generated datasets and trained models are generally **not committed to this repository**.

Each project contains:

```text
data/
models/
```

directories to provide a consistent project structure.

Generated files inside these directories are excluded through the repository-level `.gitignore`.

This keeps the Git repository lightweight while preserving a clear location for project artifacts.

---

## Reproducibility

Machine learning experiments are not always perfectly reproducible.

Results can vary because of:

* Random initialization
* Dataset changes
* Random train/validation splits
* Library versions
* Operating system
* CPU/GPU hardware
* CUDA versions
* Different downloaded data

However, projects use fixed random seeds and documented dependencies to improve reproducibility.

---

## Learning Resources

The main resources used throughout this repository include:

* [fast.ai Documentation](https://docs.fast.ai/)
* [fast.ai Course](https://course.fast.ai/)

---

## Disclaimer

This repository is primarily an educational and experimental collection.

The projects are intended to demonstrate machine learning concepts and practical workflows. They should not automatically be considered production-ready implementations.

Some projects may download datasets or other resources from external sources. Users are responsible for complying with the licensing and terms of use associated with those resources.

---

## License

The source code in this repository is licensed under the [MIT License](./LICENSE).

Individual datasets, pretrained models, images, or other third-party resources used by individual projects may have their own licenses and terms of use.

---

## Author

**Nagui**

GitHub: [@nNagui](https://github.com/nNagui)

---

⭐ If you find this repository useful for learning fast.ai or deep learning, feel free to explore the projects and follow along with the progression from simple examples to more advanced workflows.

---
