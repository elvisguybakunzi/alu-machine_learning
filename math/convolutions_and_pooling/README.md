# Convolution and Pooling Operations

## Overview

This project demonstrates the implementation of basic convolution and pooling operations using numpy. The key concepts covered include:
- Convolution
- Max Pooling
- Average Pooling
- Kernel/Filter
- Padding (Same and Valid)
- Stride
- Channels

## Requirements

- Python 3.5
- numpy 1.15

## Usage

The main script convolution_pooling.py includes functions to perform convolution, max pooling, and average pooling on an image. Example usage is provided in the script.

## Functions

### convolution(image, kernel, stride=1, padding='valid')

Performs a convolution on an image using a specified kernel and stride.

### max_pooling(image, size, stride=1)

Performs max pooling on an image.

### average_pooling(image, size, stride=1)

Performs average pooling on an image.

## Execution

To run the script, make sure it is executable:

```bash
chmod +x convolution_pooling.py 