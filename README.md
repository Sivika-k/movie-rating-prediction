# Movie Rating Prediction using Machine Learning

## Project Overview

The Movie Rating Prediction project is a real-world machine learning regression problem that aims to predict the rating of a movie based on its features such as genre, year of release, duration, votes, and cast details.

In the entertainment industry, movie ratings play a crucial role in determining a film’s success. This project helps in understanding the key factors that influence movie ratings and builds a predictive model using machine learning techniques.

The entire workflow includes:
- Data collection and preprocessing  
- Feature engineering  
- Model building  
- Evaluation  
- Prediction  

## Objective

The main objective of this project is:

- To analyze IMDb movie data  
- To identify patterns affecting movie ratings  
- To build a regression model to predict ratings  
- To evaluate model performance using statistical metrics  
- To gain hands-on experience in real-world ML workflow  

## Dataset Description

The dataset used in this project is **IMDb Movies India dataset** containing over 15,000 records.

### Features:

| Feature | Description |
|--------|-------------|
| Name | Movie title |
| Year | Release year of the movie |
| Duration | Length of the movie in minutes |
| Genre | Category of the movie (Action, Drama, Comedy, etc.) |
| Votes | Number of user ratings |
| Director | Director of the movie |
| Actor 1 | Lead actor |
| Actor 2 | Supporting actor |
| Actor 3 | Supporting actor |
| Rating | Target variable (movie rating) |

## Data Preprocessing

Data preprocessing is a critical step to improve model performance.

### 1. Handling Missing Values
- Removed rows with missing ratings  
- Filled missing categorical values with "Unknown"  
- Replaced null values in numeric columns with 0  

### 2. Data Cleaning
- Extracted numeric year from string format  
- Removed "min" from duration column  
- Converted votes into numeric format  
- Handled incorrect or missing data entries  

### 3. Feature Engineering
- Applied One-Hot Encoding on Genre  
- Converted categorical variables into numerical format  
- Selected important numerical features  

### 4. Final Features Used
- Year  
- Duration  
- Votes  
- Genre encoded columns  

## Machine Learning Model

### Algorithm Used:
- Random Forest Regressor

### Why Random Forest?

Random Forest is used because:

- It works well for both regression and classification  
- Handles large datasets efficiently  
- Reduces overfitting using multiple decision trees  
- Provides better accuracy compared to simple models  

### Model Parameters:
- n_estimators = 200  
- max_depth = 10  
- random_state = 42  

## Model Evaluation

The model is evaluated using standard regression metrics:

| Metric | Value |
|--------|-------|
| Mean Squared Error (MSE) | ~1.25 |
| R² Score | ~0.32 |

## Prediction Output

prediction:

- Predicted Rating → **4.55**  
- Actual Rating → **3.30**

## Project Workflow

1. Import libraries  
2. Load dataset  
3. Data cleaning  
4. Handling missing values  
5. Feature engineering  
6. Encoding categorical variables  
7. Train-test split  
8. Model training  
9. Prediction  
10. Evaluation  

## Author

## Sivika K
