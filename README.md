# Video Classification System using CNN-LSTM

This project implements a video classification system leveraging a combination of Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks, focusing on the classification of video data into predefined categories.

## Getting Started

### Requirements

```bash
pip install -r requirements
```

### Configuration

Edit the `params.json` file to specify the model and data processing parameters. The parameters include paths to data, model configurations, and training settings.

![Alt text](/resources/image.png)

## Data Organization

The project expects video data to be organized in a specific structure:

- The data directory should contain subdirectories, each representing a video class.
- Each subdirectory should contain `.avi` files corresponding to that class.
- The name of each subdirectory is used as the label for the videos inside it.

### Example Structure:

![Alt text](/resources/image-1.png)

## Usage

To run the video classification system:

1. Set up the `params.json` with the desired configuration.
2. Organize your video data as described above.
3. Execute the main script to start the training and classification process.

```bash
python main.py
````
## Demo

You can use the `demo.py` script to see the system in action. Simply modify the labels in the `activity_labels` with your own data and provide an instance like `demo.mpt` (which is me doing jumping jacks, and as you can see, the model correctly predicts it).

For more information about the model training process and results, please refer to the `report.pdf`.
