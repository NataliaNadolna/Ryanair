# Aircraft Take-Off Weight Prediction (TOW)

## Project Overview

The goal of this project is to predict the **actual take-off weight (ActualTOW)** of aircraft based on flight data from **Ryanair flights in 2016**.


## Exploratory Data Analysis (EDA)

The dataset includes both categorical and numerical features. It consists of **29 731 rows**, of which **26 800** are complete. The remaining **2 931 rows** have missing values in various columns.

### Missing Data Overview

| Column              | Missing Values |
|---------------------|----------------|
| `ActualTOW`         | 433            |
| `FlownPassengers`   | 95             |
| `BagsCount`         | 2 284          |
| `FlightBagsWeight`  | 2 478          |

**Decision:**
- Rows with missing values represent less than 10% of all rows.
- `ActualTOW` (target variable) - The model cannot be trained on samples where the target value is missing. These rows must be excluded.
- `FlownPassengers`, `BagsCount`, `FlightBagsWeight` - These features are essential for predicting `ActualTOW`.

Due to the importance of these observations, I decided to remove rows with missing values to maintain the quality and consistency of the dataset.

## Features

### Original Features

- `DepartureDate`
- `DepartureYear`, `DepartureMonth`, `DepartureDay`
- `FlightNumber`
- `DepartureAirport`, `ArrivalAirport`, `Route`
- `ActualFlightTime`
- `ActualTotalFuel`
- `FlownPassengers`
- `BagsCount`
- `FlightBagsWeight`

### Engineered Features

- `DayOfWeek`: day of the week (0 = Monday, ... , 6 = Sunday)
- `FuelPerMinute`: fuel burned per minute of flight
- `FuelPerPassenger`: fuel burned per passenger
- `PassengersPerBag`: ratio of passengers to bags
- `AvgBagWeight`: average weight per bag
- `IsWeekend`: 1 for Saturday/Sunday flights, 0 for others


## Data Processing

- Object-type columns were converted to `category` to allow native use with XGBoost's `enable_categorical=True`.
- The dataset was cleaned and filtered for missing values.
- Name of `FLownPassengers` was changed to `FlownPassengers`.
- Columns `DepartureDate` and `FlightNumber` were removed.

## Modeling

### Model

- `XGBRegressor` from the **XGBoost** library

### Objective

- Minimize **mean squared error (MSE)**

### Validation

- **Stratified K-Fold Cross-Validation** (n=5)
- Stratification was based on the target variable (`ActualTOW`) using quantile bins (`qcut` with 10 bins)

### Hyperparameter Tuning

- Used **Optuna** for automatic optimization
- 400 trials (`n_trials=400`)
- Each trial involved:
  - Random selection of a subset of features
  - Random search over model hyperparameters


## Best Model

The best model was saved to `best_xgb_model.pkl` and uses the optimal set of features and hyperparameters found during the tuning process.

Optimal features: `DepartureYear`, `DepartureMonth`, `DepartureDay`, `DepartureAirport`, `ArrivalAirport`, `ActualFlightTime`, `ActualTotalFuel`, `FlownPassengers`, `FlightBagsWeight`, `DayOfWeek`, `FuelPerMinute`, `AvgBagWeight`, `IsWeekend`

Optimal hyperparameters:
- `n_estimators`: 470, 
- `max_depth`: 4, 
- `learning_rate`: 0.09694081261961722, 
- `subsample`: 0.9214753371833588, 
- `colsample_bytree`: 0.6674115091571178, 
- `gamma`: 0.8418003889366719, 
- `reg_alpha`: 1.5778679003672926, 
- `reg_lambda`: 2.1406100860160375


## Prediction

The validation dataset was prepared by:

- Dropping unused columns
- Applying the same preprocessing as the training set
- Using the trained model to generate predictions
- Saving the results to `predicted.csv`



