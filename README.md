# ReviewGuard AI – Review Verification Using AI

An AI-powered review analysis system that classifies product reviews as **Computer-Generated (CG)** or **Original (OR)** using Natural Language Processing and Machine Learning.

ReviewGuard AI helps users analyze review text, identify potentially suspicious patterns, examine sentiment, and generate analysis reports through an interactive Streamlit dashboard.

---

## Project Overview

Online product reviews influence purchasing decisions, but not every review is necessarily genuine. ReviewGuard AI explores how machine learning and natural language processing can help identify potentially computer-generated reviews.

The application uses a trained Logistic Regression model with TF-IDF text vectorization to analyze review content and predict its classification.

It also provides an interactive dashboard for individual reviews, batch analysis, URL-based extraction, sentiment analysis, historical results, and report generation.

## Key Features

- **AI-Powered Classification:** Classifies reviews as Computer-Generated (CG) or Original (OR).
- **Single Review Analysis:** Analyze individual product reviews.
- **Batch Analysis:** Upload review datasets and analyze multiple reviews.
- **URL Analyzer:** Extract and analyze review content from supported product or sample pages.
- **Sentiment Analysis:** Examine the sentiment and polarity of review text.
- **Suspicious Pattern Detection:** Explore potentially unusual review patterns.
- **Analytics Dashboard:** View analysis statistics and visual insights.
- **Review History:** Access previously processed reviews within the application.
- **Export Reports:** Export analysis results in CSV and JSON formats.
- **Model Information:** View details about the underlying machine learning approach.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Frontend / Dashboard | Streamlit |
| Machine Learning | Scikit-learn |
| ML Algorithm | Logistic Regression |
| Text Processing | TF-IDF Vectorization |
| Natural Language Processing | TextBlob |
| Data Processing | Pandas, NumPy |
| Web Scraping | BeautifulSoup4, curl_cffi |
| Data Visualization | Matplotlib |
| Model Storage | Pickle |

## How It Works

1. **Input:** The user enters a product review, uploads a dataset, or provides a supported URL.
2. **Text Processing:** The review text is cleaned and prepared for analysis.
3. **Feature Extraction:** TF-IDF converts the text into numerical features.
4. **Machine Learning Prediction:** The trained Logistic Regression model analyzes the extracted features.
5. **Classification:** The application predicts one of two classes:
   - **CG:** Computer-Generated
   - **OR:** Original
6. **Results:** The dashboard displays the prediction and relevant analysis.

## Project Structure

```text
ReviewGuard_Project/
│
├── app.py
├── engine.py
├── main.py
├── train_model.py
│
├── model/
│   ├── fake_review_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── styles/
│   └── style.css
│
├── tests/
│   ├── test_app_mock.py
│   └── test_engine.py
│
├── legacy/
│   └── app_original.py
│
├── .streamlit/
│   └── config.toml
│
├── fake_reviews_dataset.csv
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/suryavr6/Review-Verification-Using-AI.git
```

### 2. Navigate to the Project Directory

```bash
cd Review-Verification-Using-AI
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

**Windows PowerShell:**

```powershell
.\venv\Scripts\Activate.ps1
```

### 5. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 6. Run the Application

```bash
streamlit run app.py
```

The application will open in your default browser. If it does not, open the local URL displayed in the terminal.

## Machine Learning Approach

ReviewGuard AI uses a supervised machine learning approach.

**Algorithm:** Logistic Regression

**Feature Extraction:** TF-IDF (Term Frequency–Inverse Document Frequency)

**Classification Labels:**

| Label | Meaning |
|---|---|
| CG | Computer-Generated |
| OR | Original |

The trained model and vectorizer are stored in the `model/` directory and loaded by the application for predictions.

## Testing

The project includes automated tests for important application and processing components.

Run the tests using:

```bash
python -m pytest tests/
```

## Important Note

ReviewGuard AI provides machine-learning-based predictions using review text. Its results are estimates, not definitive proof that a review is genuine, fake, or written by a particular person.

The application does not independently verify a reviewer's identity, purchase history, or actual product experience. Prediction quality depends on the training data and the patterns learned by the model.

URL extraction may also depend on the structure and accessibility of the target website.

## Future Enhancements

- Improve model performance with larger and more diverse datasets.
- Explore advanced NLP and deep learning models.
- Improve support for different product-review websites.
- Develop a mobile-friendly version.
- Explore multilingual review analysis.
- Add more advanced review analytics and visualizations.

## Author

**Surya VR**

B.Sc. Artificial Intelligence and Machine Learning  
Dr. N.G.P. Arts and Science College

- **GitHub:** [suryavr6](https://github.com/suryavr6)
- **LinkedIn:** [Surya V.R.](https://www.linkedin.com/in/surya-v-r-27028841b/)

---

*Built as an Artificial Intelligence and Machine Learning project to explore practical applications of NLP, text classification, and machine learning in online review analysis.*
