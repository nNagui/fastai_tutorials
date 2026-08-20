"""
Bird vs. Forest Image Classifier
================================

This script:

1. Searches the web for bird and forest images.
2. Downloads the images into a local dataset.
3. Validates and resizes the downloaded images.
4. Creates a fastai DataLoaders object.
5. Fine-tunes a ResNet18 image classifier.
6. Uses the trained model to classify a test image.

The resulting dataset is stored in the ``bird_or_not`` directory.

Prerequisites
-------------
Install the dependencies with:

    pip install -r requirements.txt

Example requirements.txt:

    ddgs
    fastai
    fastcore
    fastdownload
"""

# This enables lazy evaluation of type hints so list[str] works even on older python versions without error
from __future__ import annotations

import logging
import time
from pathlib import Path

from ddgs import DDGS
from fastai.vision.all import (
    CategoryBlock,
    DataBlock,
    ImageBlock,
    PILImage,
    RandomSplitter,
    Resize,
    error_rate,
    get_image_files,
    parent_label,
    resnet18,
    verify_images,
    vision_learner,
    download_images,
    resize_images,
)
from fastdownload import download_url


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

# Gets the folder two levels up from this script
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = DATA_DIR / "bird_or_not"
MODELS_DIR = PROJECT_ROOT / "models"

TEST_BIRD_IMAGE = DATA_DIR / "bird.jpg"
TEST_FOREST_IMAGE = DATA_DIR / "forest.jpg"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

IMAGE_SIZE = 400
VALIDATION_PERCENTAGE = 0.20
RANDOM_SEED = 42
EPOCHS = 3

# Delay between search requests.
# This helps avoid sending requests too quickly to the search service.
REQUEST_DELAY_SECONDS = 10

# Number of images to retrieve for each search query.
MAX_IMAGES_PER_SEARCH = 200

# The categories used to train the classifier.
CATEGORIES = ("forest", "bird")

# Dictionary for different search queries to increase dataset diversity.
SEARCH_QUERIES = {
    "bird": (
        "bird photo",
        "bird sun photo",
        "bird shade photo",
    ),
    "forest": (
        "forest photo",
        "forest sun photo",
        "forest shade photo",
    ),
}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Image search
# ---------------------------------------------------------------------------

def search_images(
    query: str,
    max_images: int = MAX_IMAGES_PER_SEARCH,
) -> list[str]:
    """
    Search for images matching ``query``.

    Parameters
    ----------
    query:
        Search phrase sent to the image search engine.

    max_images:
        Maximum number of image URLs to retrieve.

    Returns
    -------
    list[str]
        A list containing direct image URLs.
    """
    logger.info("Searching for images: %s", query)

    results = DDGS().images(
        query,
        max_results=max_images,
    )

    # DDGS returns dictionaries containing metadata for each image.
    # We only need the direct image URL for downloading.
    urls = [result["image"] for result in results if result.get("image")]

    logger.info("Found %d image URLs for '%s'.", len(urls), query)

    return urls


# ---------------------------------------------------------------------------
# Downloading example/test images
# ---------------------------------------------------------------------------

def download_example_images() -> None:
    """
    Download one bird image and one forest image.

    These images are used after training to demonstrate model prediction.
    """
    logger.info("Downloading example images...")

    bird_urls = search_images("bird photos", max_images=1)

    if bird_urls:
        download_url(
            bird_urls[0],
            TEST_BIRD_IMAGE,
            show_progress=False,
        )
        logger.info("Saved example bird image to %s", TEST_BIRD_IMAGE)

    forest_urls = search_images("forest photos", max_images=1)

    if forest_urls:
        download_url(
            forest_urls[0],
            TEST_FOREST_IMAGE,
            show_progress=False,
        )
        logger.info("Saved example forest image to %s", TEST_FOREST_IMAGE)


# ---------------------------------------------------------------------------
# Dataset creation
# ---------------------------------------------------------------------------

def download_dataset() -> None:
    """
    Build the training dataset by downloading images for each category.

    Images are organized using the following structure:

        bird_or_not/
        ├── bird/
        │   ├── image_001.jpg
        │   ├── image_002.jpg
        │   └── ...
        └── forest/
            ├── image_001.jpg
            ├── image_002.jpg
            └── ...

    The directory structure is important because fastai uses the parent
    directory name as the image's class label.
    """
    logger.info("Creating dataset in %s", DATASET_DIR)

    for category in CATEGORIES:
        category_dir = DATASET_DIR / category
        category_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Collecting images for category: %s", category)

        for query in SEARCH_QUERIES[category]:
            logger.info("Processing search query: '%s'", query)

            urls = search_images(query)

            # fastai's download_images handles downloading multiple images
            # and automatically gives them unique filenames.
            download_images(
                category_dir,
                urls=urls,
            )

            logger.info(
                "Downloaded images for query '%s'.",
                query,
            )

            # Give the search/download service a short break before
            # sending another request.
            logger.info(
                "Waiting %d seconds before the next search...",
                REQUEST_DELAY_SECONDS,
            )

            time.sleep(REQUEST_DELAY_SECONDS)

        # Resize images after all searches for this category are complete.
        #
        # Resizing reduces disk usage and makes subsequent model training
        # significantly faster.
        resize_images(
            category_dir,
            max_size=IMAGE_SIZE,
            dest=category_dir,
        )


# ---------------------------------------------------------------------------
# Dataset validation
# ---------------------------------------------------------------------------

def clean_dataset() -> None:
    """
    Verify downloaded images and remove files that cannot be opened.

    Web image searches can return broken URLs, unsupported files, corrupted
    images, or HTML pages masquerading as images. fastai's ``verify_images``
    identifies files that cannot be successfully opened.
    """
    logger.info("Validating downloaded images...")

    image_files = get_image_files(DATASET_DIR)
    failed_files = verify_images(image_files)

    if not failed_files:
        logger.info("All images passed validation.")
        return

    logger.warning(
        "Found %d invalid images. Removing them...",
        len(failed_files),
    )

    for file_path in failed_files:
        logger.warning("Removing invalid image: %s", file_path)
        file_path.unlink(missing_ok=True)

    logger.info("Dataset cleanup complete.")


# ---------------------------------------------------------------------------
# Model creation
# ---------------------------------------------------------------------------

def create_dataloaders():
    """
    Create fastai DataLoaders for the image classification task.

    The dataset is split into:

    - 80% training data
    - 20% validation data

    ``parent_label`` extracts the class name from the image's parent
    directory. For example:

        bird_or_not/bird/example.jpg
                         ^^^^
                         label

    becomes:

        label = "bird"
    """
    logger.info("Creating DataLoaders...")

    data_block = DataBlock(
        blocks=(
            ImageBlock,
            CategoryBlock,
        ),
        get_items=get_image_files,
        splitter=RandomSplitter(
            valid_pct=VALIDATION_PERCENTAGE,
            seed=RANDOM_SEED,
        ),
        get_y=parent_label,
        item_tfms=Resize(
            IMAGE_SIZE // 2,
            method="squish",
        ),
    )

    dataloaders = data_block.dataloaders(DATASET_DIR)

    logger.info(
        "DataLoaders created with %d training images and "
        "%d validation images.",
        len(dataloaders.train_ds),
        len(dataloaders.valid_ds),
    )

    return dataloaders


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_model(dataloaders):
    """
    Create and fine-tune a ResNet18 image classifier.

    ResNet18 is used as a pretrained convolutional neural network.
    fastai replaces the final classification layer so that the network
    predicts our two classes:

        bird
        forest

    ``fine_tune`` first trains the newly added classification layers and
    then fine-tunes the pretrained network for the specified number of
    epochs.
    """
    logger.info("Creating ResNet18 learner...")

    learner = vision_learner(
        dataloaders,
        resnet18,
        metrics=error_rate,
    )

    logger.info(
        "Starting model training for %d epochs...",
        EPOCHS,
    )

    learner.fine_tune(EPOCHS)

    logger.info("Model training complete.")

    return learner


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_image(learner, image_path: Path) -> None:
    """
    Classify a single image using the trained model.

    Parameters
    ----------
    learner:
        Trained fastai learner.

    image_path:
        Path to the image that should be classified.

    Notes
    -----
    We deliberately obtain the probability using the predicted class index
    rather than assuming that ``probs[0]`` is always the bird probability.

    This is safer because class ordering is determined by the DataLoaders.
    """
    logger.info("Classifying image: %s", image_path)

    image = PILImage.create(image_path)

    predicted_class, _, probabilities = learner.predict(image)

    # fastai stores the class names in ``learner.dls.vocab``.
    class_names = learner.dls.vocab

    logger.info("Model prediction: %s", predicted_class)

    print(f"\nPrediction for: {image_path}")
    print(f"Class: {predicted_class}")

    print("\nClass probabilities:")

    for class_name, probability in zip(class_names, probabilities):
        print(f"  {class_name}: {probability:.2%}")


# ---------------------------------------------------------------------------
# Application entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Execute the complete image-classification pipeline.

    The workflow is intentionally kept in separate stages so that each
    stage can later be modified independently.
    """
    logger.info("Starting bird-vs-forest image classifier.")

    # Stage 1: Download a couple of images that will later be used
    # to demonstrate the trained model.
    download_example_images()

    # Stage 2: Build the training dataset by searching and downloading
    # images for each class.
    download_dataset()

    # Stage 3: Remove corrupted or unsupported image files.
    clean_dataset()

    # Stage 4: Convert the dataset into fastai DataLoaders.
    dataloaders = create_dataloaders()

    # Stage 5: Train the image classifier using transfer learning.
    learner = train_model(dataloaders)

    # Stage 6: Run the trained model against an image that was not used
    # as part of the training dataset.
    predict_image(
        learner,
        TEST_BIRD_IMAGE,
    )

    logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
