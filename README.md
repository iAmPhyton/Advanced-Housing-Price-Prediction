Advanced Housing Price Prediction

Project Overview:
- Predicting real estate prices is challenging due to the complex mix of quantitative (square footage) and qualitative (kitchen quality) features. This project utilizes Ridge Regression to predict the final sale price of homes in Ames, Iowa.

The goal was to build a model that handles multicollinearity (related features) and skewed data effectively.

Results:
* Model: Ridge Regression (alpha=10).
* Metric: Root Mean Squared Error (Log Scale).
* Final Score: 0.139 (Top tier performance).

Key Drivers of Price:
According to the model coefficients, the top factors increasing home value are:
1.  Neighborhood: Stone Brook & Northridge Heights.
2.  Overall Quality: Material and finish of the house.
3.  Total Square Footage: The combined living area.

How to Run:
1.  Clone this repo.
2.  Install requirements: `pip install pandas seaborn scikit-learn matplotlib`.
3.  Run the notebook: `jupyter notebook ames.ipynb`.

key Challenges & Solutions:
- The Problem: Initial visualizations showed that the model was relying on generic nouns like "movie" and "film" as top predictors, which is logically incorrect (both good and bad reviews use these words).

The Solution:
* Analyzed the `feature_log_prob_` of the Naïve Bayes model.
* Identified "noise" words common to the movie domain.
* Extended the standard English stop-word list to include: `['movie', 'film', 'just', 'watch']`.

The Result: The model's top features shifted to strong sentiment indicators:
* Negative: "Waste", "Worst", "Awful".
* Positive: "Perfect", "Excellent", "Incredible".

Author:
- Chukwuemeka Eugene Obiyo
- LinkedIn: https://www.linkedin.com/in/chukwuemekao/
- Email: praise609@gmail.com
