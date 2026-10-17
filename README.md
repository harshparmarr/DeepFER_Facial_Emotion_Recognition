# 😀 DeepFER: Facial Emotion Recognition Using Deep Learning

**Author:** Harsh Parmar

DeepFER classifies a face image into one of seven expressions: **angry, disgust, fear, happy, neutral, sad, surprise**. It compares a Custom CNN with MobileNetV2 transfer learning and serves the better model in a Streamlit app.

> **Honest summary:** the final model reaches **47.63% test accuracy** (macro F1 0.40) on 7 classes. That is about 1.9x the 24.9% you get by always guessing "happy", but below the roughly 65% estimated for human labelers on this dataset. This is a learning project and a demo, not a production-grade classifier.

---

## Results

| Metric | Custom CNN | MobileNetV2 (final) |
|---|---|---|
| Test accuracy | 38.61% | **47.63%** |
| Macro F1 | 0.21 | **0.40** |
| Weighted F1 | 0.31 | 0.46 |
| Classes never predicted correctly | angry, disgust, surprise (fear almost never) | disgust (recall 0.03) |

Per-class F1 (test set, 5,765 images):

| Class | Custom CNN | MobileNetV2 | Test images |
|---|---|---|---|
| angry | 0.00 | 0.35 | 799 |
| disgust | 0.00 | 0.07 | 87 |
| fear | 0.01 | 0.32 | 821 |
| happy | 0.76 | 0.64 | 1,433 |
| neutral | 0.43 | 0.43 | 996 |
| sad | 0.25 | 0.37 | 988 |
| surprise | 0.00 | 0.60 | 641 |

**How to read this**
- Macro F1 is the fair headline number. Weighted F1 is flattered by the large Happy class.
- Transfer learning helped (Angry, Fear and Surprise went from near-zero to usable), but the comparison is **not like-for-like**: the CNN sat at about 25% accuracy for its first ~9 epochs and was still improving when it stopped, and MobileNetV2 trained for only 10 epochs with a frozen base (train 51%, validation 47%, both still rising).
- Disgust is effectively not detected: only 436 training images (1.5% of the data).

---

## Dataset

- **Source:** FER-2013, image-folder version (`train/` and `validation/`, one sub-folder per emotion).
- **Used in the notebook:** the `train` folder only, 28,821 images. The `validation` folder (7,066 images) was not used.
- **Format:** 48x48 pixel **grayscale** images. OpenCV loads them as 3 identical channels, which is why the arrays are `(48, 48, 3)`. They are not colour images.

| Class | Images | Share |
|---|---|---|
| happy | 7,164 | 24.9% |
| neutral | 4,982 | 17.3% |
| sad | 4,938 | 17.1% |
| fear | 4,103 | 14.2% |
| angry | 3,993 | 13.9% |
| surprise | 3,205 | 11.1% |
| disgust | 436 | 1.5% |

**Data-quality findings (content-hash check)**
- 1,261 exact duplicate images in `train` (27,560 unique of 28,821).
- 45 identical images appear under more than one label.
- With a random split, duplicates can end up in both train and test and slightly inflate accuracy.
- FER-2013 labels are noisy; human accuracy was estimated at about 65% ± 5% (Goodfellow et al., 2013, arXiv:1307.0414).

---

## Method

1. **Data understanding:** class counts, sample images, size and pixel checks.
2. **Preprocessing:** label + one-hot encoding, stratified 80/20 train/test split, then a 20% validation split of the training part (18,444 / 4,612 / 5,765 images), division by 255.
3. **Custom CNN:** 4 x (Conv2D + BatchNorm + MaxPool + Dropout), Dense 256 and 128, softmax. Trained with augmentation (rotation, shifts, zoom, horizontal flip), Adam, early stopping.
4. **MobileNetV2:** images resized 48 → 96 px, frozen ImageNet base, GlobalAveragePooling, Dense 256, Dropout 0.5, softmax. Trained for 10 epochs **without** augmentation.
5. **Selection:** MobileNetV2 was chosen on test accuracy.

---

## Streamlit app

`app.py` loads `deepfer_mobilenetv2.keras` and `emotion_labels.pkl`, then:

1. converts the upload to grayscale (fixing phone-photo rotation),
2. crops the largest detected face (images of 96 px or less are assumed to be face crops already),
3. resizes to 48 px and then 96 px, scales to [0, 1] (the same pipeline as training),
4. shows the prediction, the face the model actually saw, and the top-3 probabilities,
5. flags predictions below 40% confidence as unreliable.

Prediction is **on demand per uploaded image**. It is not real-time video.

The preprocessing was checked against the notebook's training pipeline on 200 real dataset images (maximum difference 0.0025 on a 0-1 scale).

### Run it

```bash
git clone https://github.com/YOUR-USERNAME/DeepFER.git
cd DeepFER
pip install -r requirements.txt
streamlit run app.py
```

Tested with TensorFlow 2.21, Streamlit 1.66 and OpenCV 4.13. Place `deepfer_mobilenetv2.keras` and `emotion_labels.pkl` (created by the last section of the notebook) next to `app.py`.

---

## Known limitations

- Wrong more often than right (47.63% on 7 classes).
- Trained on small grayscale front-facing crops; colour photos, side views, masks and poor lighting degrade results.
- It predicts the expression label, not what a person feels. Facial expression is an unreliable signal of inner emotion.
- Test accuracy is mildly optimistic because of duplicate images across the split.
- No class weights, no fine-tuning, MobileNetV2 inputs scaled to [0, 1] instead of the [-1, 1] range it was pre-trained with.

## Responsible use

Do not use this to make decisions about people. Under the EU AI Act, inferring emotions of people in **workplaces and education institutions** has been prohibited since 2 February 2025, with narrow medical and safety exceptions. Other uses may carry their own obligations. This is not legal advice; seek proper review before any real deployment.

---

## Next steps (prioritised)

1. De-duplicate before splitting; use the official `validation` folder as an extra check.
2. Use `mobilenet_v2.preprocess_input`; add class weights; report macro F1 and per-class recall.
3. Train to convergence, then fine-tune the top MobileNetV2 layers at a low learning rate, with augmentation.
4. Add a majority-class baseline and a stronger from-scratch CNN for a fair comparison.
5. Evaluate on a second dataset to measure real-world generalization.
6. Only then consider webcam input, multi-face detection and cloud deployment.

---

## Project structure

```text
DeepFER/
├── DeepFER_Facial_Emotion_Recognition.ipynb
├── deepfer_mobilenetv2.keras
├── emotion_labels.pkl
├── app.py
├── requirements.txt
├── README.md
└── dataset/
```

## Technologies

Python, NumPy, Pandas, Matplotlib, Seaborn, TensorFlow/Keras, OpenCV, Streamlit.

## References

- Goodfellow et al., *Challenges in Representation Learning: A report on three machine learning contests* (2013), arXiv:1307.0414
- TensorFlow / Keras, MobileNetV2, OpenCV and Streamlit documentation
