# traffic_management_system.py
import numpy as np
import pandas as pd
import pickle
import time
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

class TimingModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def train(self, training_data=None):
        """Train the model using historical data or generate synthetic data if none provided"""
        if training_data is None:
            # Generate synthetic training data if none provided
            print("Generating synthetic training data...")
            X, y = self._generate_synthetic_data()
        else:
            # Use provided training data
            X = training_data.drop('green_time', axis=1)
            y = training_data['green_time']
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train a RandomForest model
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            min_samples_split=5,
            random_state=42
        )
        
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        # Save the model and scaler
        self._save_model()
        
        print("Model trained successfully!")
        return self
    
    def _generate_synthetic_data(self, samples=20000):
        """Generate synthetic data for training the model"""
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
        # This represents our domain knowledge about how signal timing should work
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
        
        return df.drop('green_time', axis=1), df['green_time']
    
    def predict_timing(self, vehicle_counts, hour=None, day_of_week=None):
        """Predict the optimal green time based on vehicle counts and time factors"""
        if not self.is_trained:
            self.load_model()
            
        if hour is None:
            hour = int(time.strftime('%H'))
        if day_of_week is None:
            day_of_week = int(time.strftime('%w'))  # 0=Sunday, 6=Saturday
        
        # Prepare input data
        X = pd.DataFrame({
            'cars': [vehicle_counts['cars']],
            'bikes': [vehicle_counts['bikes']],
            'buses': [vehicle_counts['buses']],
            'trucks': [vehicle_counts['trucks']],
            'rickshaws': [vehicle_counts['rickshaws']],
            'hour': [hour],
            'day_of_week': [day_of_week]
        })
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make prediction
        green_time = self.model.predict(X_scaled)[0]
        
        # Ensure prediction is within bounds
        green_time = max(10, min(60, green_time))
        return int(round(green_time))
    
    def _save_model(self):
        """Save the trained model and scaler to files"""
        if not os.path.exists('models'):
            os.makedirs('models')
        
        with open('models/timing_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        
        with open('models/timing_scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
    
    def load_model(self):
        """Load the trained model and scaler from files"""
        try:
            with open('models/timing_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            
            with open('models/timing_scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            
            self.is_trained = True
            print("Model loaded successfully!")
        except FileNotFoundError:
            print("Model files not found. Training a new model...")
            self.train()
        
        return self


class TrafficManagementSystem:
    def __init__(self):
        self.timing_model = TimingModel()
        self.lane_states = {0: {}, 1: {}, 2: {}, 3: {}}
        self.logs = []
    
    def setup(self, train_new_model=False):
        """Set up the traffic management system"""
        if train_new_model:
            print("Training new traffic timing model...")
            self.timing_model.train()
        else:
            print("Loading existing traffic timing model...")
            self.timing_model.load_model()
    
    def process_lane(self, lane, simulate=True, vehicle_counts=None, hour=None, day_of_week=None):
        """Process a lane to determine optimal signal timing"""
        # If vehicle_counts provided, use them instead of detection
        if vehicle_counts is not None:
            # Calculate total
            total_count = sum(vehicle_counts.values())
            
            # Predict timing
            allocated_time = self.timing_model.predict_timing(vehicle_counts, hour, day_of_week)
            
            # Apply business rules (optional)
            # Example: If total_count is very low, reduce time to minimum
            if total_count <= 2:
                allocated_time = 10  # Minimum time
            
            # Update lane state
            self.lane_states[lane] = {
                'vehicle_count': total_count,
                'allocated_time': allocated_time,
                'counts_by_type': vehicle_counts,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Log the event
            log_entry = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'lane': lane,
                'vehicle_count': total_count,
                'allocated_time': allocated_time,
                'vehicle_counts': vehicle_counts
            }
            self.logs.append(log_entry)
            
            return self.lane_states[lane]
        
        elif simulate:
            # Simulate vehicle detection (only for testing)
            print(f"Simulating detection for lane {lane}...")
            
            # Generate random vehicle counts based on lane and time of day
            hour = int(time.strftime('%H'))
            day_of_week = int(time.strftime('%w'))  # 0=Sunday, 6=Saturday
            
            # Simulate different traffic patterns
            is_rush_hour = (7 <= hour <= 9) or (17 <= hour <= 19)
            is_weekend = day_of_week >= 5
            
            # Base counts
            if is_rush_hour and not is_weekend:
                # Rush hour on weekday
                base = {
                    'cars': np.random.randint(5, 12),
                    'bikes': np.random.randint(3, 7),
                    'buses': np.random.randint(1, 4),
                    'trucks': np.random.randint(1, 3),
                    'rickshaws': np.random.randint(2, 6)
                }
            elif is_weekend:
                # Weekend
                base = {
                    'cars': np.random.randint(4, 8),
                    'bikes': np.random.randint(2, 5),
                    'buses': np.random.randint(0, 2),
                    'trucks': np.random.randint(0, 2),
                    'rickshaws': np.random.randint(1, 4)
                }
            else:
                # Normal weekday
                base = {
                    'cars': np.random.randint(2, 7),
                    'bikes': np.random.randint(1, 4),
                    'buses': np.random.randint(0, 3),
                    'trucks': np.random.randint(0, 3),
                    'rickshaws': np.random.randint(1, 4)
                }
            
            # Get timing prediction
            allocated_time = self.timing_model.predict_timing(base, hour, day_of_week)
            
            # Calculate total vehicles
            total_count = sum(base.values())
            
            # Update lane state
            self.lane_states[lane] = {
                'vehicle_count': total_count,
                'allocated_time': allocated_time,
                'counts_by_type': base,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Log the event
            log_entry = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'lane': lane,
                'vehicle_count': total_count,
                'allocated_time': allocated_time,
                'vehicle_counts': base
            }
            self.logs.append(log_entry)
            
            return self.lane_states[lane]
        
        else:
            # This would use computer vision in a real-world implementation
            print("Error: Real vehicle detection not implemented. Use simulate=True or provide vehicle_counts.")
            return None
    
    def get_lane_status(self, lane):
        """Get the current status of a lane"""
        if lane in self.lane_states:
            return self.lane_states[lane]
        return None
    
    def export_logs(self, filename='traffic_logs.csv'):
        """Export logs to CSV file"""
        if not self.logs:
            print("No logs to export.")
            return
        
        # Convert logs to DataFrame
        df = pd.DataFrame(self.logs)
        
        # Expand the vehicle_counts dictionary
        for vehicle_type in ['cars', 'bikes', 'buses', 'trucks', 'rickshaws']:
            df[vehicle_type] = df['vehicle_counts'].apply(lambda x: x.get(vehicle_type, 0))
        
        # Drop the dictionary column
        df = df.drop('vehicle_counts', axis=1)
        
        # Save to CSV
        df.to_csv(filename, index=False)
        print(f"Logs exported to {filename}")
