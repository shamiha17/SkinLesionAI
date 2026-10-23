# 🔬 SkinLesionAI — AI-Powered Skin Image Classification

An educational machine learning project that uses **MobileNetV2 and Streamlit** to classify skin images into eight output categories through a simple, interactive web application.

> **Disclaimer:** SkinLesionAI is a research and educational prototype, not a medical device. It must not be used to diagnose, rule out, or treat any medical condition.

## ✨ Features

- 🧠 **MobileNetV2:** Transfer-learning-based image classification.
- 🖼️ **Image Upload:** Upload JPG, JPEG, and PNG images.
- 📊 **Prediction Dashboard:** View the top predicted class and output scores for all classes.
- 📈 **Score Visualization:** Explore model outputs with a chart.
- 📥 **CSV Export:** Download prediction scores.
- 🎨 **Modern UI:** Dark navy and teal interface.
- ⚡ **Cached Model Loading:** Avoid reloading the model on every Streamlit rerun.

## 🧪 Model Information

- **Architecture:** MobileNetV2
- **Task:** Eight-class image classification
- **Input size:** 160 × 160 pixels
- **Framework:** TensorFlow / Keras
- **Interface:** Streamlit

### Output Categories

| Code | Category |
|---|---|
| `akiec` | Actinic keratoses and intraepithelial carcinoma |
| `bcc` | Basal cell carcinoma |
| `bkl` | Benign keratosis-like lesions |
| `df` | Dermatofibroma |
| `mel` | Melanoma |
| `normal` | Normal category in the project's training data |
| `nv` | Melanocytic nevi |
| `vasc` | Vascular lesions |

*These are model output labels, not confirmed diagnoses.*

## 📊 Model Evaluation

On the project's eight-class held-out test split, the model achieved:

| Metric | Result |
|---|---:|
| Test accuracy | 69.35% |
| Macro F1-score | 0.5308 |
| Weighted F1-score | 0.7214 |
| Test samples | 1,674 |

These results describe performance on this particular test split only. They do not establish clinical accuracy or performance on other populations, devices, or image sources. The `normal` category may also be affected by differences between data sources.

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- MobileNetV2
- Streamlit
- NumPy
- Pillow
- Pandas-compatible CSV output

## 📁 Project Structure

```text
SkinLesionAI/
├── app.py
├── requirements.txt
├── README.md
├── best_mobilenetv2_8class.keras  # Model file, hosted separately
└── skinlesion_8class_labels.json
```

The trained model file is not included in the Git repository. Download or place the compatible model file in the project root before running the app.

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/shamiha17/SkinLesionAI.git
cd SkinLesionAI
```

### 2. Create a virtual environment

**Windows:**

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Use a Python version supported by your installed TensorFlow release. Python 3.11 or 3.12 may be suitable, depending on the TensorFlow version.

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Add the trained model

Place these files in the same directory as `app.py`:

- `best_mobilenetv2_8class.keras`
- `skinlesion_8class_labels.json`

The model must match the architecture, preprocessing, input dimensions, and label order expected by the application.

### 5. Run the application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal, usually `http://localhost:8501`.

## 🖥️ How to Use

1. Start the Streamlit application.
2. Upload a supported skin image.
3. Select **Analyze image**.
4. Review the top predicted category and model output score.
5. Explore the scores for all eight categories.
6. Optionally download the scores as a CSV file.

## ⚠️ Limitations and Responsible Use

- This project is for educational and research purposes only.
- It has not been validated for clinical use.
- Model predictions can be incorrect, including predictions with high output scores.
- Performance may vary with skin tone, lighting, camera quality, image source, and other factors.
- Rare categories have limited test samples, so their evaluation metrics may be unstable.
- The `normal` category and images from different sources may introduce dataset bias.
- The evaluation does not establish external validity or clinical reliability.
- Always consult a qualified healthcare professional about skin concerns.

**Do not use this application to make medical decisions or delay professional care.**

## 👩‍💻 Author

**Shamiha Jahan**

GitHub: [@shamiha17](https://github.com/shamiha17)

## 📄 License

No license has been specified yet. Unless a license is added, others do not automatically receive permission to reuse, modify, or distribute this project's code. Check the terms of the datasets and pretrained assets separately before redistribution.
