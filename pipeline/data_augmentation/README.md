# Data Augmentation


## Description
This project implements various data augmentation techniques using TensorFlow. Data augmentation is a strategy that enables practitioners to significantly increase the diversity of data available for training models, without actually collecting new data. The project includes implementations of common augmentation techniques such as flipping, rotating, shearing, and adjusting brightness and hue of images.

## Learning Objectives
At the end of this project, you will be able to explain:
- What data augmentation is and its importance
- When to perform data augmentation
- Benefits of using data augmentation
- Various ways to perform data augmentation
- How to use ML to automate data augmentation

## Requirements
### General
- Allowed editors: `vi`, `vim`, `emacs`
- Intended Python version: 3.6.12
- Operating System: Ubuntu 16.04 LTS
- Required packages:
  - tensorflow (version 1.15)
  - numpy (version 1.16)
  - tensorflow-datasets
- Style guide: `pycodestyle` (version 2.4)
- All files must be executable
- All modules, classes, and functions must be documented

## Installation
```bash
# Install required tensorflow-datasets
pip install --user tensorflow-datasets
```

## File Descriptions

### 0-flip.py
Contains the `flip_image` function that flips an image horizontally.
```python
def flip_image(image):
    """
    Flips an image horizontally
    Args:
        image: 3D tf.Tensor containing the image to flip
    Returns:
        The flipped image
    """
```

### 1-crop.py
Contains the `crop_image` function that performs a random crop of an image.
```python
def crop_image(image, size):
    """
    Performs a random crop of an image
    Args:
        image: 3D tf.Tensor containing the image to crop
        size: tuple containing the size of the crop
    Returns:
        The cropped image
    """
```

### 2-rotate.py
Contains the `rotate_image` function that rotates an image 90 degrees counter-clockwise.
```python
def rotate_image(image):
    """
    Rotates an image by 90 degrees counter-clockwise
    Args:
        image: 3D tf.Tensor containing the image to rotate
    Returns:
        The rotated image
    """
```

### 3-shear.py
Contains the `shear_image` function that randomly shears an image.
```python
def shear_image(image, intensity):
    """
    Randomly shears an image
    Args:
        image: 3D tf.Tensor containing the image to shear
        intensity: intensity with which the image should be sheared
    Returns:
        The sheared image
    """
```

### 4-brightness.py
Contains the `change_brightness` function that randomly changes the brightness of an image.
```python
def change_brightness(image, max_delta):
    """
    Randomly changes the brightness of an image
    Args:
        image: 3D tf.Tensor containing the image to change
        max_delta: maximum amount the image should be brightened (or darkened)
    Returns:
        The altered image
    """
```

### 5-hue.py
Contains the `change_hue` function that changes the hue of an image.
```python
def change_hue(image, delta):
    """
    Changes the hue of an image
    Args:
        image: 3D tf.Tensor containing the image to change
        delta: amount the hue should change
    Returns:
        The altered image
    """
```

## Usage Examples
Each function can be tested using its corresponding main file. For example:

```bash
# Test flip_image function
./0-main.py

# Test crop_image function
./1-main.py

# Test rotate_image function
./2-main.py

# Test shear_image function
./3-main.py

# Test change_brightness function
./4-main.py

# Test change_hue function
./5-main.py
```