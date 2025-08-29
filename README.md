# PRODIGY_DS_04
Twitter Entity Sentiment Analysis. Enjoy😊

This project analyzes X, formerly known as Twitter, posts to detect sentiment (Positive, Negative, Neutral) towards specific entities mentioned in the tweets. The goal is to understand public opinion patterns and deploy machine learning model for real-time predictions.

## Table of Contents
- [Data Collection](#data-collection)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Conclusion](#conclusion)
- [Deployment](#deployment)
- [Demo](#demo)
- [Future Work](#future-work)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [License](#license)

## Data Collection

We used the `Twitter Sentiment Analysis Dataset` from Kaggle, which contains a large number of tweets labeled with their corresponding sentiment. This dataset was chosen for its relevance and comprehensiveness in capturing the nuances of sentiment in social media text.

```python
import pandas as pd

train = pd.read_csv("data/twitter_training.csv")
valid = pd.read_csv("data/twitter_validation.csv")

data = pd.concat([train, valid], ignore_index=True)
```
- Training and validation datasets were combined for a more comprehensive analysis.

## Data Preprocessing
The data cleaning process involved:
- Removing duplicates
- Handling missing values
- Normalizing text: lowercasing, removing URLs, special characters, and numbers.
- Kept two features: text (the tweet) and entity (the target brand).
- Target variable: sentiment.

```
Shape after cleaning: (70942, 4)
``` 

## Exploratory Data Analysis
We explored common themes and patterns in the data, including the distribution of sentiments across different entities and the impact of various factors on sentiment.

**Sentiment Distribution**
![alt text](<Visualizations/Sentiment Distribution.png>)
*This shows the distribution of sentiments across all entities in the dataset. It also shows us the class imbalance present in the data.*

**Top Entities by Sentiment**
![alt text](<Visualizations/Top Entities.png>)
*`Call of duty` received more positive sentiment compared to other entities while `MaddenNFL` received more negative sentiment compared to other entities.*

**Identifying words/phrases associated with each sentiment**
![alt text](<Visualizations/Most frequent words.png>)

## Model Training
We trained multiple classifiers, tuned hyperparameters with GridSearchCV, and selected the best one.
```
Fitting 3 folds for each of 18 candidates, totalling 54 fits
Best params: {'clf__C': 5, 'pre__tfidf__min_df': 2, 'pre__tfidf__ngram_range': (1, 2)}
              precision    recall  f1-score   support

    Negative       0.93      0.93      0.93      4298
     Neutral       0.94      0.93      0.93      5996
    Positive       0.91      0.92      0.91      3895

    accuracy                           0.93     14189
   macro avg       0.93      0.93      0.93     14189
weighted avg       0.93      0.93      0.93     14189
```

GridSearchCV was used to find the optimal hyperparameters for our models, ensuring the best performance on the validation set. We used f1_macro to focus on balanced performance across classes.

### Model Evaluation
Confusion matrix shows which classes are getting confused (e.g., Neutral ↔ Positive).
![alt text](<Visualizations/Confusion Matrix.png>)

Error analysis (inspect wrong predictions) revealed label noise, sarcasm, or dataset quirks — extremely valuable for iterative improvement.
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>entity</th>
      <th>true</th>
      <th>pred</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>37967</th>
      <td>see</td>
      <td>Battlefield</td>
      <td>Positive</td>
      <td>Neutral</td>
    </tr>
    <tr>
      <th>21482</th>
      <td>let's fuck it!!!</td>
      <td>CS-GO</td>
      <td>Positive</td>
      <td>Negative</td>
    </tr>
    <tr>
      <th>68066</th>
      <td>the loving the new " world terminator event " ...</td>
      <td>TomClancysGhostRecon</td>
      <td>Positive</td>
      <td>Neutral</td>
    </tr>
    <tr>
      <th>7100</th>
      <td>epic overwatch gameplay i! comments tch.tv / m...</td>
      <td>Overwatch</td>
      <td>Positive</td>
      <td>Neutral</td>
    </tr>
    <tr>
      <th>2600</th>
      <td>&lt;unk&gt;'m in agreement with this name | &lt;unk&gt;ا&lt;u...</td>
      <td>CallOfDutyBlackopsColdWar</td>
      <td>Positive</td>
      <td>Neutral</td>
    </tr>
    <tr>
      <th>3974</th>
      <td>here we go.</td>
      <td>CallOfDutyBlackopsColdWar</td>
      <td>Positive</td>
      <td>Neutral</td>
    </tr>
    <tr>
      <th>38139</th>
      <td>we didn't deserve battlefield 1.</td>
      <td>Battlefield</td>
      <td>Positive</td>
      <td>Negative</td>
    </tr>
    <tr>
      <th>50194</th>
      <td>2nd starts time playing through red letter dea...</td>
      <td>RedDeadRedemption(RDR)</td>
      <td>Positive</td>
      <td>Neutral</td>
    </tr>
    <tr>
      <th>39722</th>
      <td>nan</td>
      <td>PlayerUnknownsBattlegrounds(PUBG)</td>
      <td>Positive</td>
      <td>Neutral</td>
    </tr>
    <tr>
      <th>42799</th>
      <td>hello @ twitchsupport i'm having problems with...</td>
      <td>Verizon</td>
      <td>Positive</td>
      <td>Negative</td>
    </tr>
  </tbody>
</table>
</div>

I also tried improving imbalance by oversampling the minority class since I already used `class_weight = "balanced"`. The results showed a slight decrease in performance.
```
Resampled class counts: Negative    23983
Neutral     23983
Positive    23983
Name: count, dtype: int64

Classification Report (with oversampling):
              precision    recall  f1-score   support

    Negative       0.89      0.89      0.89      4298
     Neutral       0.89      0.89      0.89      5996
    Positive       0.86      0.88      0.87      3895

    accuracy                           0.88     14189
   macro avg       0.88      0.88      0.88     14189
weighted avg       0.88      0.88      0.88     14189
```

## Conclusion

The model demonstrates strong performance across all classes, with particular strengths in precision and recall for the Negative and Neutral classes. The use of GridSearchCV and f1_macro scoring has ensured a balanced approach to model evaluation, highlighting the importance of considering class imbalances in sentiment analysis tasks.

## Deployment

We deployed with Streamlit for real-time sentiment predictions.

```python
import streamlit as st
import joblib

# Save model
model = joblib.load("model/best_model.pkl")
# Load model
model = joblib.load("sentiment_model.pkl")
```
You can run the `app.py` file:
```bash
streamlit run app.py
```
## Demo
**I will Add a demo**


## Future Work

- Try deep learning (BERT, RoBERTa) for better accuracy.
- Handle class imbalance with SMOTE or reweighting.
- Add topic modeling for deeper insights.

## Project Structure
```lua
.
├── app.py
├── LICENSE
├── model
│   ├── best_model.pkl
│   └── sentiment_model.pkl
├── Analysis.ipynb
├── requirements.txt
└── README.md
```
## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Campeon254/PRODIGY_DS_04.git
   cd PRODIGY_DS_04
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.