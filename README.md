# Aerial Tree Detection Models Comparison

This project compares the performance of different object detection models for detecting trees in aerial images. The models evaluated are YOLOv5, YOLOv8, DETR, and YOLO-NAS. The evaluation metrics used include precision, recall, and mean Average Precision (mAP).

## Folder Structure

The project has the following folder structure:

```
root:/
|_docs/                    # Documentation files
|_models/                  # Models' folders
  |_yolov5/
    |_train/              # Training notebook and related files
    |_model/              # Best trained model weight and loader module for testing
  |_yolov8/
    |_train/
    |_model/
  |_detr/
    |_train/
    |_model/
  |_yolo-nas/
    |_train/
    |_model/
|_data/                    # Aerial images dataset
|_results/                 # Results of model evaluation
```

## Usage

To use the models, follow these steps:

1. Clone the repository to your local machine.
2. Download the aerial images dataset.
3. Open the training notebook for the desired model in the corresponding train folder.
4. Run the notebook to train the model on the dataset.
5. Once training is complete, the best trained model weight will be saved in the model folder.
6. To test the model, open the loader module in the model folder and run it.
7. Evaluate the model performance using the evaluation metrics and save the results in the results folder.

Note: The weight files are large and are therefore ignored in the repository. You will need to train the models yourself to generate the weight files.
!TODO: Add links for my trained model weights (wait for it).

## Requirements

- Python 3.6 or later
- Pytorch
- OpenCV
- Matplotlib

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Contributors

- [Pouria Zarei](https://github.com/porya-zarei)