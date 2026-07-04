# Real-time yoga pose estimation with a hybrid MoveNet architecture

Code, trained model and evaluation data behind my first-author paper in **Sādhanā (Indian Academy of Sciences, Springer, 2025)**: [doi.org/10.1007/s12046-025-02788-w](https://doi.org/10.1007/s12046-025-02788-w)

![Pose estimation overlay](assets/animation.gif)

**Try it live:** the trained classifier from this repo runs in your browser at [axwolf13.github.io/demo](https://axwolf13.github.io/demo/), MoveNet Lightning tracks your webcam and the 28 KB head below names the asana.

## Results

| Metric | Value |
|---|---|
| Test accuracy (Keras, 571 held-out samples) | **98.77%** |
| Weighted avg precision/recall/F1 | 0.99 |
| Test accuracy after TFLite conversion | **98.42%** (562/571) |
| Model size (TFLite, dynamic-range quantized) | 28 KB |
| Classes | 11 asanas |
| Exploratory 49-class run | 90.0% (961 samples) |

Verify the headline number yourself in two commands, no GPU needed:

```
pip install -r requirements.txt
python verify_model.py
```

Expected output ends with `562/571 = 0.9842`.

## Architecture

The design splits training and inference deliberately:

1. **Training:** MoveNet **Thunder** (the accurate variant) extracts 17 COCO keypoints from each image, with a 3-pass crop-refinement loop for stability. Only the keypoints are kept.
2. **Classifier head:** 51 inputs (17 keypoints × x, y, score) → hip-centered, torso-scaled normalization baked into the graph → Dense 128 (ReLU6) → Dense 64 (ReLU6) → 11-way softmax. Trained on keypoints, never on pixels.
3. **Inference:** the backbone swaps to MoveNet **Lightning** for 30+ FPS on consumer hardware. Accuracy where the model learns, speed where the user waits.

Because the head only ever sees normalized keypoints, it is camera-resolution independent and small enough to run anywhere, including as plain JavaScript in the browser demo.

## Repository contents

| Path | What it is |
|---|---|
| `notebooks/pose_classifier_training.ipynb` | The paper's training run (April 2024): preprocessing, training, confusion matrix, misclassification analysis, TFLite export. Outputs preserved. |
| `notebooks/pose_classifier_49class_experiment.ipynb` | Earlier exploratory run on a harder 49-class set (90%). Kept for honesty about the path to the final scope. |
| `notebooks/movenet_realtime_inference.ipynb` | MoveNet Lightning real-time webcam inference with keypoint/skeleton drawing. Outputs cleared (they contained webcam frames). |
| `model/pose_classifier.tflite` | The trained 11-class head, byte-for-byte the artifact evaluated in the paper. |
| `model/pose_labels.txt` | The 11 class labels. |
| `data/test2_data.csv` | The paper's held-out test set: Thunder keypoints for 571 samples, 11 classes. Feeds `verify_model.py`. |
| `data/train_data.csv`, `data/test_data.csv` | Keypoint CSVs from the 49-class exploratory run. |
| `verify_model.py` | Standalone script reproducing the 98.42% TFLite accuracy from the shipped model and CSV. |

## Data

The underlying 3,435 images (11 asanas, curated from public Kaggle yoga datasets and [Yoga-82](https://sites.google.com/view/yoga-82/home)) are **not redistributed here**; image licensing is the dataset owners' call, not mine. What this repo ships instead is the extracted keypoint representation, which is all the classifier ever trained on. That is enough to reproduce every number above.

## Honest limitations

- **Closed-set classifier.** It answers "which of the 11 poses is this most like?", never "is this a yoga pose at all?". Anything outside the set maps confidently to the nearest class. The browser demo adds a geometric gate for neutral standing precisely because of this.
- **2D, single person.** Keypoints are image-plane coordinates from one person; camera angle changes the geometry.
- **Class imbalance in the test set** (Tree 116 samples, Child 24), so per-class numbers vary more than the headline.
- Test images come from the same source distribution as training. Field robustness (odd angles, occlusion, low light) was not part of the paper's evaluation.

## Attribution

The training pipeline builds on Google's [TensorFlow Lite pose classification tutorial](https://www.tensorflow.org/lite/tutorials/pose_classification) (Apache 2.0). MoveNet models are from [TensorFlow Hub](https://www.kaggle.com/models/google/movenet). The dataset curation, 11-class scope, training runs, evaluation and the paper itself are my own work with my co-authors at NIT Puducherry.

## Links

- Paper (DOI): https://doi.org/10.1007/s12046-025-02788-w
- Live browser demo: https://axwolf13.github.io/demo/
- Portfolio: https://axwolf13.github.io
