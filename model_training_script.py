
# train_traffic_model.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def generate_training_data(samples=20000):
    """Generate synthetic data for training the model"""
    print("Generating synthetic training data...")
    # Create random vehicle counts
    data = {
        'cars': np.random.randint(0, 15, samples),
        'bikes': np.random.randint(0, 10, samples),
        'buses': np.random.randint(0, 5, samples),
        'trucks': np.random.randint(0, 5, samples),
        'rickshaws': np.random.randint(0, 8, samples),
        'hour': np.random.randint(0, 24, samples),
        'day_of_week': np.random.randint(0, 7, samples)
    }
    
    # Calculate a synthetic green time based on vehicle counts and weights
    df = pd.DataFrame(data)
    
    # Add rush hour factor (7-9am and 5-7pm get higher weights)
    rush_hour_morning = ((df['hour'] >= 7) & (df['hour'] <= 9))
    rush_hour_evening = ((df['hour'] >= 17) & (df['hour'] <= 19))
    df['rush_hour'] = rush_hour_morning | rush_hour_evening
    
    # Weekend factor (weekends get different weights)
    df['weekend'] = df['day_of_week'] >= 5  # 5=Sat, 6=Sun
    
    # Calculate green time using a formula that considers vehicle types and time factors
    df['green_time'] = (
        1.2 * df['cars'] +      # Cars have moderate impact
        0.8 * df['bikes'] +     # Bikes have less impact
        2.5 * df['buses'] +     # Buses have high impact
        2.5 * df['trucks'] +    # Trucks have high impact
        1.5 * df['rickshaws']   # Rickshaws have moderate-high impact
    )
    
    # Add time-based adjustments
    df.loc[df['rush_hour'], 'green_time'] *= 1.2  # 20% longer during rush hour
    df.loc[df['weekend'], 'green_time'] *= 0.9    # 10% shorter during weekends
    
    # Add some noise to make it more realistic
    noise = np.random.normal(0, 2, samples)
    df['green_time'] += noise
    
    # Ensure minimum and maximum values (10-60 seconds)
    df['green_time'] = np.clip(df['green_time'], 10, 60)
    df['green_time'] = df['green_time'].round()
    
    # Remove temporary columns
    df = df.drop(['rush_hour', 'weekend'], axis=1)
    
    # Save dataset
    if not os.path.exists('data'):
        os.makedirs('data')
    df.to_csv('data/traffic_signal_data.csv', index=False)
    
    print(f"Generated {samples} samples of synthetic data")
    return df

def train_model(data=None):
    """Train the traffic signal timing model"""
    if data is None:
        # Check if dataset exists
        if os.path.exists('data/traffic_signal_data.csv'):
            print("Loading existing dataset...")
            data = pd.read_csv('data/traffic_signal_data.csv')
        else:
            # Generate synthetic data
            data = generate_training_data()
    
    print("Training model with dataset shape:", data.shape)
    
    # Split features and target
    X = data.drop('green_time', axis=1)
    y = data['green_time']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train with hyperparameter tuning
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [6, 8, 10, None],
        'min_samples_split': [2, 5, 10]
    }
    
    grid_search = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    
    print("Starting hyperparameter tuning...")
    grid_search.fit(X_train_scaled, y_train)
    print(f"Best parameters: {grid_search.best_params_}")
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test_scaled)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Test set metrics:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"R² Score: {r2:.2f}")
    
    # Create model directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Save model and scaler
    with open('models/timing_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    
    with open('models/timing_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("Model and scaler saved successfully")
    
    # Plot feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('models/feature_importance.png')
    
    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Green Time')
    plt.ylabel('Predicted Green Time')
    plt.title('Actual vs Predicted Green Time')
    plt.tight_layout()
    plt.savefig('models/prediction_performance.png')
    
    # Test the model with some sample inputs
    print("\nTesting model with sample inputs:")
    
    # Test cases
    test_cases = [
        # Rush hour with high traffic
        {'cars': 12, 'bikes': 8, 'buses': 3, 'trucks': 2, 'rickshaws': 5, 'hour': 8, 'day_of_week': 2},
        # Weekend with moderate traffic
        {'cars': 7, 'bikes': 4, 'buses': 1, 'trucks': 1, 'rickshaws': 3, 'hour': 13, 'day_of_week': 6},
        # Late night with low traffic
        {'cars': 2, 'bikes': 1, 'buses': 0, 'trucks': 0, 'rickshaws': 1, 'hour': 23, 'day_of_week': 3}
    ]
    
    for i, case in enumerate(test_cases):
        case_df = pd.DataFrame([case])
        case_scaled = scaler.transform(case_df)
        prediction = best_model.predict(case_scaled)[0]
        
        print(f"Case {i+1}: {case}")
        print(f"Predicted green time: {prediction:.1f} seconds\n")
    
    return best_model, scaler

if __name__ == "__main__":
    print("Traffic Signal Timing Model Trainer")
    print("----------------------------------")
    
    train_model()
