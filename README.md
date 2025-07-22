# Email/SMS Spam Classifier

A lightweight, demo-friendly **Streamlit web app** that classifies short text messages (emails or SMS) as **Spam** or **Ham (Not Spam)** using a **machine learning ensemble**. Ideal for learning text preprocessing, TF‑IDF vectorization, and model stacking in scikit-learn — with an interactive UI you can launch locally in minutes.

---

## ✨ Key Features

* **Real-time Classification:** Type/paste a message and instantly see the prediction.
* **Text Preprocessing Pipeline:** Lowercasing, tokenization, alphanumeric filtering, stopword removal, and Porter stemming.
* **TF‑IDF Vectorization:** Converts cleaned text into numerical feature vectors.
* **Ensemble Model (StackingClassifier):** Combines strengths of multiple base learners (SVC, MultinomialNB, ExtraTrees) with a meta RandomForest for robust performance.
* **Interactive Streamlit UI:** Text area input, Predict & Clear buttons, and quick-load example Spam/Ham test messages.
* **Confidence Scores:** Displays model probability for each class.
* **Robustness:** Graceful handling of missing model/vectorizer files and automatic NLTK data download (stopwords, punkt) when needed.

---

## 🖼️ App Preview

> *Add screenshots or a short GIF of the running app here.*

---

## 📚 Table of Contents

1. [Project Overview](#-project-overview)
2. [How It Works](#️-how-it-works)

   * [Preprocessing Steps](#preprocessing-steps)
   * [Vectorization](#vectorization)
   * [Sparse→Dense Conversion](#sparse→dense-conversion)
   * [Model Architecture](#model-architecture)
3. [Quick Start](#-quick-start)

   * [Prerequisites](#prerequisites)
   * [Installation](#installation)
   * [Download NLTK Data](#download-nltk-data)
   * [Add Model Artifacts](#place-your-trained-models)
   * [Run the App](#-run-the-application)
4. [Usage Guide](#-usage)
5. [Project Structure](#-project-structure)
6. [Training Your Own Model](#-training-your-own-model-optional)
7. [Troubleshooting](#-troubleshooting)
8. [Extending the App](#-extending-the-app)
9. [Contributing](#-contributing)
10. [License](#-license)
11. [Acknowledgements](#-acknowledgements)

---

## 📘 Project Overview

This project demonstrates a complete **text classification workflow** wrapped in a friendly **Streamlit interface**. Users can paste in email/SMS text and immediately see whether the message is likely spam. The project is designed for students, data science beginners, and educators who want:

* A clear example of an **NLP preprocessing + ML pipeline**.
* A demonstration of **StackingClassifier ensembles** in scikit-learn.
* A ready-to-run **interactive demo** for workshops, portfolio projects, or classroom use.

---

## 🛠️ How It Works

At a high level, the app transforms raw text into numeric features and feeds them to a pre-trained ensemble classifier.

### Preprocessing Steps

Each input message is processed using the following steps (order matters):

1. **Lowercase** the full string.
2. **Tokenize** into word-like units.
3. **Filter non-alphanumeric** tokens (keep words/numbers; drop punctuation & symbols).
4. **Remove English stopwords** (e.g., "the", "and", "is").
5. **Stemming** via NLTK's **PorterStemmer** (e.g., "running" → "run").
6. **Rejoin** tokens or directly pass the cleaned token list to the vectorizer (depending on your implementation).

> *Note:* If you change preprocessing at training time, the **same function** must be applied at inference time for consistent results.

### Vectorization

We use **scikit-learn's `TfidfVectorizer`** trained on the project dataset. It converts preprocessed text into a sparse matrix where each column corresponds to a vocabulary term and values represent TF‑IDF weights.

* Save the fitted vectorizer (e.g., `vectorizer.pkl`).
* Load it in the app to transform new input text.

### Sparse→Dense Conversion

Some base estimators (notably `sklearn.svm.SVC` with certain kernels) expect **dense NumPy arrays** rather than sparse matrices. Because the StackingClassifier forwards data to its base learners, we convert the TF‑IDF sparse matrix to dense using `.toarray()` or `.A` before prediction.

> *Performance Tip:* Dense conversion can increase memory usage for large vocabularies. For production systems, consider models that accept sparse input (e.g., linear models, MultinomialNB) or use dimensionality reduction.

### Model Architecture

A **StackingClassifier** combines multiple base models and learns how to weigh their predictions using a meta-learner. A typical configuration for this project:

**Base Estimators**

* `('svc', SVC(probability=True))`
* `('mnb', MultinomialNB())`
* `('ext', ExtraTreesClassifier(n_estimators=200, random_state=42))`

**Final Estimator (Meta-model)**

* `RandomForestClassifier(n_estimators=200, random_state=42)`

**Workflow**

1. Each base model predicts on the training data (often via cross-validated out-of-fold predictions in stacking to reduce leakage).
2. These predictions become meta-features.
3. The meta RandomForest learns to combine them into the final decision.

---

## 🚀 Quick Start

Get up and running locally.

### Prerequisites

* Python **3.7+** (3.10+ recommended)
* `pip`
* Recommended: Virtual environment (`venv`, `conda`, `pipenv`, etc.)

### Installation

```bash
# Clone the repository
git clone https://github.com/Anish62027/Email-SMS-Spam-Classifier.git
cd Email-SMS-Spam-Classifier

# Create & activate virtual environment (choose one)
# --- Windows PowerShell ---
python -m venv venv
.\venv\Scripts\activate

# --- macOS / Linux ---
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt  # if provided
# or install core libs directly:
pip install streamlit scikit-learn numpy nltk
```

If you don't yet have a `requirements.txt`, create one like this:

```txt
streamlit
scikit-learn
numpy
nltk
```

### Download NLTK Data

The app attempts to download missing corpora at runtime, but you can pre-download:

```bash
python -m nltk.downloader stopwords punkt
```

### Place Your Trained Models

You **must** have two pickled artifacts in the project root (same folder as `app.py`):

* `vectorizer.pkl`  → Trained `TfidfVectorizer`.
* `model.pkl`       → Trained `StackingClassifier` (or compatible model supporting `.predict()` and `.predict_proba()`).

> *Tip:* See [Training Your Own Model](#-training-your-own-model-optional) if you need help creating these.

### ▶ Run the Application

```bash
streamlit run app.py
```

Streamlit will print a local URL (typically `http://localhost:8501`) — open it in your browser.

---

## 💡 Usage

1. **Enter Message:** Paste or type any email/SMS in the text box.
2. **Predict:** Click **Predict** to classify.
3. **Clear:** Reset the input with **Clear Message**.
4. **Example Buttons:** Quickly test with built-in **Spam Example** or **Ham Example** messages.
5. **View Results:** The app displays:

   * **Prediction:** `SPAM!` or `NOT SPAM (HAM)`.
   * **Confidence:** Probability scores from the model (e.g., `Spam: 0.87`, `Ham: 0.13`).

---

## 📂 Project Structure

A suggested layout (yours may vary):

```
Email-SMS-Spam-Classifier/
├─ app.py                     # Streamlit app entry point
├─ preprocess.py              # (Optional) Text cleaning utilities
├─ train_model.ipynb          # (Optional) Notebook to train vectorizer + model
├─ data/
│  ├─ raw/                    # Original dataset(s)
│  ├─ processed/              # Cleaned data / splits
├─ models/
│  ├─ vectorizer.pkl          # TF-IDF vectorizer (copy to root if app expects root)
│  ├─ model.pkl               # StackingClassifier (copy to root if app expects root)
├─ examples/
│  ├─ spam_examples.txt
│  ├─ ham_examples.txt
├─ requirements.txt
├─ LICENSE
└─ README.md
```


## 🧩 Troubleshooting

### 1. **`NotFittedError` in MultinomialNB or other models**

This typically means your model (or a base learner inside the StackingClassifier) was **not fitted** before being pickled/used. Confirm that you called `.fit()` on the full pipeline and that you're loading the correct artifact.

**Checklist:**

* Did training complete successfully without errors?
* Are you loading the same versions of scikit-learn used during training?
* Did you save and load the **trained** estimator (not just an uninitialized instance)?

### 2. **PowerShell `ExecutionPolicy` prevents venv activation**

If you see an error like `running scripts is disabled on this system`, set an execution policy for the current user:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Then re-run:

```powershell
.\venv\Scripts\Activate
```

(You may need to start a new PowerShell session.)

### 3. **Missing NLTK Data**

If the app errors on stopwords or punkt:

```bash
python -m nltk.downloader stopwords punkt
```

Or add a runtime downloader in code (see `app.py` example).

### 4. **Missing `vectorizer.pkl` / `model.pkl`**

The app will stop and show an error. Make sure the files exist in the expected path.

### 5. **Memory Issues with Dense Conversion**

Large vocabularies + dense conversion → big arrays. Consider:

* Limiting `max_features` in `TfidfVectorizer`.
* Using `LinearSVC` via `CalibratedClassifierCV` (dense but fewer features after chi2 select).
* Dropping dense-unfriendly base learners.

---

## 🌱 Extending the App

Ideas to grow the project:

* Add **model training mode** in-app (upload CSV, train, download artifacts).
* Track **prediction history** and allow user feedback (correct/incorrect) for active learning.
* Add **explainability**: show top weighted words for Spam/Ham decisions.
* Support **multilingual spam detection** (switch language stopwords/models).
* Add **REST API** layer (FastAPI) behind the Streamlit UI.
* Containerize with **Docker** for reproducibility.
* Deploy to **Streamlit Community Cloud**, **Railway**, or **Azure App Service**.

---

## 🤝 Contributing

Contributions are welcome — bug reports, feature requests, docs improvements, refactors, or new training scripts.

**To contribute:**

1. Fork the repo.
2. Create a feature branch: `git checkout -b feature/my-improvement`.
3. Make your changes & add tests if applicable.
4. Lint/format (`ruff`, `black`, `isort` suggested but optional).
5. Commit & push.
6. Open a Pull Request describing your change.

Please open an Issue first for large changes or model refactors.

---

## 📄 License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

*Example MIT boilerplate:*

```txt
MIT License

Copyright (c) 2025 Anish Kumar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 🙏 Acknowledgements

* Built with **[Streamlit](https://streamlit.io/)**.
* Machine learning powered by **[scikit-learn](https://scikit-learn.org/)**.
* Text processing via **[NLTK](https://www.nltk.org/)**.
* Inspired by open datasets and community spam/ham tutorials.

---

## 📬 Contact

**Author:** Anish kumar
**GitHub:** **
**Email:** 

If you use this project in teaching, please let me know — I'd love to see how you extend it!

---

### Badge Ideas (Optional)

Add these at the top of the README once you wire up CI/tools:

```markdown
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-app-brightgreen)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
```

---

**Happy classifying!** 🚀
