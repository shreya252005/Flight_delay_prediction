# Flight_delay_prediction

# Flight Delay Prediction Pipeline (>60 Minutes)

A machine learning project built in Google Colab to predict severe U.S. domestic flight delays using tabular flight performance records, an optimized XGBoost classification pipeline, and an interactive prediction interface.

---

## 🚀 Project Overview
Predicting aviation delays is critical for minimizing passenger disruption and understanding operational bottlenecks. This project processes a large-scale U.S. Department of Transportation (DOT) flight dataset to classify whether a specific flight will experience a severe delay exceeding **60 minutes**.

### Key Pipeline Stages:
1. **Data Ingestion & Filtering:** Loaded raw flight datasets and scoped down to core operational features (`DayOfWeek`, `Date`, `DepTime`, `Airline`, `Origin`, `Dest`, `CarrierDelay`).
2. **Datetime Engineering:** Parsed dates using Pandas and extracted numerical temporal features (`month`, `day`) for model ingestion.
3. **One-Hot Encoding:** Transformed high-cardinality categorical variables (`Airline`, `Origin`, `Dest`) into binary dummy matrices (`pd.get_dummies`).
4. **Target Definition:** Engineered a binary classification target variable (`is_delayed_60+`) based on a 60-minute delay threshold.
5. **Exploratory Data Analysis (EDA):** Re-engineered one-hot encoded variables back into human-readable strings to analyze delay distributions, class imbalance (~7.5% severe delay rate), and regional airport risk factors.
6. **Machine Learning Model:** Trained a supervised **XGBoost Classifier** configured with `random_state=0` and `eval_metric='logloss'` for reproducible gradient-boosted tree performance.
7. **Interactive Inference CLI:** Features a user input function that dynamically structures raw queries into the exact training feature space, calculates delay probabilities (`predict_proba`), and outputs a real-world delay verdict.

---

## 🛠️ Tech Stack & Dependencies
* **Environment:** Google Colab (Cloud Linux VM)
* **Language:** Python 3.x
* **Core Libraries:**
  * `pandas`, `numpy` (Data manipulation & numerical arrays)
  * `scikit-learn` (Data splitting & evaluation metrics)
  * `xgboost` (Binary classification model)
  * `matplotlib`, `seaborn` (Data visualization & distribution histograms)

---

