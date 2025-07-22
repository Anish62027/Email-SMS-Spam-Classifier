# Email-SMS-Spam-Classifier

📧 Email/SMS Spam Classifier
This project provides a web-based application built with Streamlit that classifies text messages (emails or SMS) as either "spam" or "not spam" (ham) using a machine learning model. It's designed to be an interactive tool for demonstration and understanding of text classification.

✨ Features
Real-time Classification: Instantly classify messages entered by the user.

Text Preprocessing: Includes robust text cleaning steps (lowercasing, tokenization, alphanumeric filtering, stopword removal, stemming).

TF-IDF Vectorization: Utilizes Term Frequency-Inverse Document Frequency for numerical feature extraction from text.

Ensemble Learning Model: Employs a powerful StackingClassifier for high accuracy.

Interactive GUI: User-friendly interface built with Streamlit, featuring:

Text input area.

Predict and Clear buttons.

Pre-defined example spam and ham messages for quick testing.

Clear display of prediction (Spam/Not Spam) and confidence scores.

Robustness: Includes error handling for missing model files and NLTK data downloads.

⚙️ How It Works
The project follows a standard machine learning pipeline for text classification:

Data Preprocessing: Raw text input undergoes a series of cleaning steps:

Conversion to lowercase.

Tokenization (splitting text into words).

Removal of non-alphanumeric characters.

Elimination of common English stopwords.

Stemming (reducing words to their root form using Porter Stemmer).

Feature Extraction (Vectorization): The preprocessed text is converted into a numerical format using a pre-trained TfidfVectorizer. This process transforms text into a sparse matrix representation, capturing word importance.

Sparse to Dense Conversion: A crucial step where the sparse matrix output from TF-IDF is converted into a dense NumPy array. This is necessary because some internal components of the StackingClassifier (like SVC) expect dense input.

Machine Learning Prediction: The numerical vector is fed into a pre-trained StackingClassifier model. This ensemble model combines predictions from multiple base classifiers (e.g., Support Vector Classifier, Multinomial Naive Bayes, Extra Trees Classifier) and uses a final estimator (e.g., RandomForestClassifier) to make the ultimate prediction.

Result Display: The model outputs a binary prediction (0 for ham, 1 for spam) and associated probabilities. The Streamlit app then displays this result clearly to the user.

🚀 Getting Started
Follow these instructions to get a copy of the project up and running on your local machine.

Prerequisites
Python 3.7+

pip (Python package installer)

Installation
Clone the repository:

git clone https://github.com/Anish62027/Email-SMS-Spam-Classifier.git
cd Email-SMS-Spam-Classifier

(Replace https://github.com/Anish62027/Email-SMS-Spam-Classifier.git with your actual repository URL if it's different in the future)

Create a virtual environment (recommended):

python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

Install the required Python packages:

pip install streamlit scikit-learn numpy nltk

Download NLTK data:
The application will attempt to download stopwords and punkt automatically. If it fails or you prefer to do it manually, run:

python -m nltk.downloader stopwords punkt

Place your trained models:
Ensure you have your trained vectorizer.pkl (TfidfVectorizer) and model.pkl (StackingClassifier) files in the root directory of the project. These files are essential for the application to run.

🏃‍♀️ Running the Application
Once the prerequisites are met and models are in place:

streamlit run app.py

This command will open the Streamlit application in your default web browser.

💡 Usage
Enter Message: Type or paste any email or SMS content into the "Enter the message here:" text area.

Predict: Click the "Predict" button to classify the message.

Clear Message: Click the "Clear Message" button to clear the text area.

Load Examples: Use the "Load Spam Example" or "Load Ham Example" buttons to quickly populate the text area with pre-defined messages for testing.

View Results: The classification ("SPAM!" or "NOT SPAM (HAM)") and confidence scores will be displayed below the input area.

📊 Model Details
The core of the classifier is a StackingClassifier from scikit-learn. This ensemble model typically consists of:

Base Estimators:

sklearn.svm.SVC (Support Vector Classifier)

sklearn.naive_bayes.MultinomialNB (Multinomial Naive Bayes)

sklearn.ensemble.ExtraTreesClassifier (Extra Trees Classifier)

Final Estimator (Meta-model):

sklearn.ensemble.RandomForestClassifier (Random Forest Classifier)

The text features are generated using TfidfVectorizer.

🤝 Contributing
Contributions are welcome! If you have suggestions for improvements, bug fixes, or new features, please open an issue or submit a pull request.

📄 License
This project is licensed under the MIT License - see the LICENSE file for details. (You might want to create a LICENSE file in your repo if you haven't already)

🙏 Acknowledgements
Built with Streamlit.

Leverages scikit-learn for machine learning models.

Utilizes NLTK for natural language processing tasks.

Inspired by various online tutorials and resources on spam classification.
