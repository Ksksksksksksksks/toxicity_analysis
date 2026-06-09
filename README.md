# Temporal and Behavioral Patterns of Toxicity and Sentiment in Twitter

**Author:** Kseniia Khudiakova  
**Course:** CENG 790 – Big Data Analytics, METU  
**Objective:** Academic project to practice scalable data processing with PySpark, train a toxicity classifier, and test two hypotheses:  
1. Toxicity varies by weekday and hour.  
2. Toxicity is equivalent to negative sentiment.

## Repository Structure

- **`bigdata-eda.ipynb`**  
  Exploratory data analysis and preprocessing of Sentiment140 (1.6M tweets) and Twitter Toxic Tweets (32K tweets). Includes text cleaning, deduplication, date parsing, and class balance visualisation. Outputs cleaned Parquet files.

- **`toxicity-classifier.ipynb`**  
  Trains and evaluates toxicity classifiers (Logistic Regression, Random Forest, Naive Bayes) using TF‑IDF features (HashingTF/CountVectorizer + IDF). Handles class imbalance with instance weighting. Saves the best model (Naive Bayes + CountVectorizer) and metrics.

- **`markup-dataset.ipynb`**  
  Applies the trained model to the entire Sentiment140 dataset. Adds `toxicity_score` and `is_toxic` columns. Aggregates results by user and by (weekday, hour) to create feature sets for hypothesis testing.

  
- **`analysis.ipynb`**  
  Performs statistical hypothesis testing:  
  - ANOVA and Tukey HSD for weekday and hour effects on toxicity ratio.  
  - Pearson and Spearman correlations between user‑level toxicity and positive sentiment ratio.  
  - Kruskal–Wallis test for sentiment groups.  
  Generates all final figures (bar plots, heatmaps, scatter plots, boxplots).

- **`report.pdf`**  
  Conference‑style paper (IEEEtran) describing the full project.

## How to Reproduce

1. Install dependencies:
   ```bash
   pip install pyspark findspark numpy scipy pandas matplotlib seaborn statsmodels scikit-posthocs
   ```
2. Download the datasets:
   - Sentiment140: [Kaggle – kazanova/sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140)
   - Twitter Toxic Tweets: [Kaggle – umitka/twitter-toxic-tweets](https://www.kaggle.com/datasets/umitka/twitter-toxic-tweets)
3. Update file paths in the notebooks to point to your local copies.
4. Run notebooks in order:
   - `bigdata-eda.ipynb`
   - `toxicity-classifier.ipynb`
   - `markup-dataset.ipynb`
   - `analysis.ipynb`

## Results Summary

- **H1 confirmed:** Toxicity peaks on Tuesdays and at 21:00; lowest on Saturdays and early morning (ANOVA p < 0.001).  
- **H2 rejected:** Correlation between user‑level toxicity and positive sentiment ratio < 0.04.  
- Best classifier: Naive Bayes + CountVectorizer + IDF, recall = 0.695, F1 = 0.502.
