import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

class CPUSchedulingPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.features = None
        self.accuracy = None
        
    def load_data(self, file_path):
        """Load and prepare the dataset"""
        print(f"Loading data from {file_path}")
        try:
            df = pd.read_csv(file_path)
            print(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def train_model(self, df, test_size=0.2, random_state=42):
        """Train the model with the provided dataset"""
        # Select the most important features for prediction
        self.features = ['NumProcesses', 'AvgPriority', 'AvgBurstTime', 'AvgArrivalTime']
        
        # Additional derived features that might help in prediction
        if set(['FCFS_AWT', 'SJF_AWT', 'SRTF_AWT', 'RR_AWT']).issubset(df.columns):
            df['FCFS_SJF_Diff'] = df['FCFS_AWT'] - df['SJF_AWT']
            df['FCFS_SRTF_Diff'] = df['FCFS_AWT'] - df['SRTF_AWT']
            df['FCFS_RR_Diff'] = df['FCFS_AWT'] - df['RR_AWT']
            df['SJF_SRTF_Diff'] = df['SJF_AWT'] - df['SRTF_AWT']
            self.features.extend(['FCFS_SJF_Diff', 'FCFS_SRTF_Diff', 'FCFS_RR_Diff', 'SJF_SRTF_Diff'])
            
        X = df[self.features]
        y = df['BestAlgo_by_Score']
        
        print("Selected features:", self.features)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Create a pipeline with scaling and model
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(random_state=random_state))
        ])
        
        # Hyperparameter tuning
        param_grid = {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [None, 10, 20, 30],
            'classifier__min_samples_split': [2, 5, 10]
        }
        
        print("Starting hyperparameter tuning...")
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        # Get the best model
        self.model = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
        
        # Evaluate the model
        y_pred = self.model.predict(X_test)
        self.accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model accuracy: {self.accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Feature importances
        importances = self.model.named_steps['classifier'].feature_importances_
        feature_importance = pd.DataFrame({
            'Feature': self.features,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        print("\nFeature Importances:")
        print(feature_importance)
        
        return self.accuracy
    
    def save_model(self, file_path="cpu_scheduling_model.pkl"):
        """Save the trained model to disk"""
        if self.model is None:
            print("No model to save. Please train the model first.")
            return False
            
        try:
            with open(file_path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'features': self.features,
                    'accuracy': self.accuracy
                }, f)
            print(f"Model saved successfully to {file_path}")
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def load_model(self, file_path="cpu_scheduling_model.pkl"):
        """Load a trained model from disk"""
        try:
            with open(file_path, 'rb') as f:
                model_data = pickle.load(f)
                
            self.model = model_data['model']
            self.features = model_data['features']
            self.accuracy = model_data['accuracy']
            
            print(f"Model loaded successfully from {file_path}")
            print(f"Model accuracy: {self.accuracy:.4f}")
            print(f"Features used: {self.features}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict_best_algorithm(self, input_data):
        """Predict the best CPU scheduling algorithm based on input parameters"""
        if self.model is None:
            print("No model available. Please train or load a model first.")
            return None
        
        # Prepare input data
        input_df = pd.DataFrame([input_data])
        
        # If we have the comparison metrics in our feature set but not in input_data,
        # we need to estimate them
        derived_features = ['FCFS_SJF_Diff', 'FCFS_SRTF_Diff', 'FCFS_RR_Diff', 'SJF_SRTF_Diff']
        for feature in derived_features:
            if feature in self.features and feature not in input_data:
                # Simple heuristic estimates - these could be improved with a secondary model
                if feature == 'FCFS_SJF_Diff':
                    input_df[feature] = input_data.get('AvgBurstTime', 0) * 0.2
                elif feature == 'FCFS_SRTF_Diff':
                    input_df[feature] = input_data.get('AvgBurstTime', 0) * 0.3
                elif feature == 'FCFS_RR_Diff':
                    input_df[feature] = input_data.get('AvgBurstTime', 0) * 0.1
                elif feature == 'SJF_SRTF_Diff':
                    input_df[feature] = input_data.get('AvgBurstTime', 0) * 0.1
        
        # Make sure we have all required features
        missing_features = [f for f in self.features if f not in input_df.columns]
        if missing_features:
            print(f"Missing required features: {missing_features}")
            return None
            
        # Select only the features the model was trained on
        input_df = input_df[self.features]
        
        # Make prediction
        prediction = self.model.predict(input_df)[0]
        probabilities = self.model.predict_proba(input_df)[0]
        
        # Get all algorithms and their probabilities
        algo_probs = {
            algo: prob for algo, prob in zip(self.model.classes_, probabilities)
        }
        
        sorted_algos = sorted(algo_probs.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'best_algorithm': prediction,
            'confidence': algo_probs[prediction],
            'all_probabilities': sorted_algos
        }


def train_and_save_model(data_file="scheduling_data.csv", model_file="cpu_scheduling_model.pkl"):
    """Train and save a model based on the provided dataset"""
    predictor = CPUSchedulingPredictor()
    df = predictor.load_data(data_file)
    
    if df is not None:
        predictor.train_model(df)
        predictor.save_model(model_file)
        return predictor
    return None


def interactive_prediction():
    """Interactive CLI for making predictions"""
    predictor = CPUSchedulingPredictor()
    
    # Try to load model, or train if it doesn't exist
    if not predictor.load_model():
        print("Model not found. Training a new model...")
        predictor = train_and_save_model()
        if predictor is None:
            print("Failed to train model. Exiting.")
            return
    
    print("\n" + "="*50)
    print("CPU Scheduling Algorithm Predictor")
    print("="*50)
    
    while True:
        print("\nEnter the process details (or 'q' to quit):")
        
        try:
            num_processes = input("Number of processes: ")
            if num_processes.lower() == 'q':
                break
            num_processes = int(num_processes)
            
            avg_priority = float(input("Average priority (1-10): "))
            avg_burst_time = float(input("Average burst time (ms): "))
            avg_arrival_time = float(input("Average arrival time (ms): "))
            
            # Optional parameters - can be extended based on model requirements
            additional_params = {}
            
            print("\nWould you like to enter additional parameters? (y/n)")
            if input().lower() == 'y':
                print("Enter FCFS_AWT (Average Waiting Time): ")
                additional_params['FCFS_AWT'] = float(input())
                
                print("Enter SJF_AWT (Average Waiting Time): ")
                additional_params['SJF_AWT'] = float(input())
                
                print("Enter SRTF_AWT (Average Waiting Time): ")
                additional_params['SRTF_AWT'] = float(input())
                
                print("Enter RR_AWT (Average Waiting Time): ")
                additional_params['RR_AWT'] = float(input())
                
                # Calculate derived features
                additional_params['FCFS_SJF_Diff'] = additional_params['FCFS_AWT'] - additional_params['SJF_AWT']
                additional_params['FCFS_SRTF_Diff'] = additional_params['FCFS_AWT'] - additional_params['SRTF_AWT']
                additional_params['FCFS_RR_Diff'] = additional_params['FCFS_AWT'] - additional_params['RR_AWT']
                additional_params['SJF_SRTF_Diff'] = additional_params['SJF_AWT'] - additional_params['SRTF_AWT']
            
            # Prepare input data
            input_data = {
                'NumProcesses': num_processes,
                'AvgPriority': avg_priority,
                'AvgBurstTime': avg_burst_time,
                'AvgArrivalTime': avg_arrival_time,
                **additional_params
            }
            
            # Make prediction
            result = predictor.predict_best_algorithm(input_data)
            
            if result:
                print("\nPrediction Results:")
                print(f"Best algorithm: {result['best_algorithm']}")
                print(f"Confidence: {result['confidence']:.4f}")
                
                print("\nAll algorithm probabilities:")
                for algo, prob in result['all_probabilities']:
                    print(f"{algo}: {prob:.4f}")
            else:
                print("Failed to make prediction.")
                
        except ValueError as e:
            print(f"Invalid input: {e}")
        except Exception as e:
            print(f"Error: {e}")
    
    print("Thank you for using the CPU Scheduling Algorithm Predictor!")


# Web interface using Flask
def create_web_app():
    from flask import Flask, request, jsonify, render_template
    
    app = Flask(__name__)
    predictor = CPUSchedulingPredictor()
    
    # Try to load model, or train if it doesn't exist
    if not predictor.load_model():
        print("Model not found. Training a new model...")
        predictor = train_and_save_model()
        if predictor is None:
            print("Failed to train model. Web app may not work correctly.")
    
    @app.route('/')
    def home():
        return render_template('index.html')
    
    @app.route('/predict', methods=['POST'])
    def predict():
        try:
            data = request.json
            result = predictor.predict_best_algorithm(data)
            
            if result:
                return jsonify(result)
            else:
                return jsonify({"error": "Failed to make prediction"}), 400
                
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    return app


# Example HTML template for the web interface
def create_html_template():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>CPU Scheduling Algorithm Predictor</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
            }
            .form-group {
                margin-bottom: 15px;
            }
            label {
                display: block;
                margin-bottom: 5px;
            }
            input {
                width: 100%;
                padding: 8px;
                box-sizing: border-box;
            }
            button {
                background-color: #4CAF50;
                color: white;
                padding: 10px 15px;
                border: none;
                cursor: pointer;
            }
            .result {
                margin-top: 20px;
                padding: 15px;
                border: 1px solid #ddd;
                display: none;
            }
            .advanced-toggle {
                color: blue;
                text-decoration: underline;
                cursor: pointer;
            }
            .advanced-section {
                display: none;
                border: 1px solid #eee;
                padding: 10px;
                margin-top: 10px;
            }
        </style>
    </head>
    <body>
        <h1>CPU Scheduling Algorithm Predictor</h1>
        <p>Enter the details of your CPU scheduling scenario to get a recommendation.</p>
        
        <div class="form-container">
            <div class="form-group">
                <label for="numProcesses">Number of Processes:</label>
                <input type="number" id="numProcesses" min="1" required>
            </div>
            
            <div class="form-group">
                <label for="avgPriority">Average Priority (1-10):</label>
                <input type="number" id="avgPriority" min="1" max="10" step="0.1" required>
            </div>
            
            <div class="form-group">
                <label for="avgBurstTime">Average Burst Time (ms):</label>
                <input type="number" id="avgBurstTime" min="0" step="0.1" required>
            </div>
            
            <div class="form-group">
                <label for="avgArrivalTime">Average Arrival Time (ms):</label>
                <input type="number" id="avgArrivalTime" min="0" step="0.1" required>
            </div>
            
            <div>
                <span class="advanced-toggle" onclick="toggleAdvanced()">+ Advanced Options</span>
                <div class="advanced-section" id="advancedSection">
                    <div class="form-group">
                        <label for="fcfsAwt">FCFS Average Waiting Time (if known):</label>
                        <input type="number" id="fcfsAwt" min="0" step="0.1">
                    </div>
                    
                    <div class="form-group">
                        <label for="sjfAwt">SJF Average Waiting Time (if known):</label>
                        <input type="number" id="sjfAwt" min="0" step="0.1">
                    </div>
                    
                    <div class="form-group">
                        <label for="srtfAwt">SRTF Average Waiting Time (if known):</label>
                        <input type="number" id="srtfAwt" min="0" step="0.1">
                    </div>
                    
                    <div class="form-group">
                        <label for="rrAwt">RR Average Waiting Time (if known):</label>
                        <input type="number" id="rrAwt" min="0" step="0.1">
                    </div>
                </div>
            </div>
            
            <button onclick="predict()">Predict Best Algorithm</button>
        </div>
        
        <div class="result" id="result">
            <h2>Prediction Results</h2>
            <p><strong>Best Algorithm:</strong> <span id="bestAlgo"></span></p>
            <p><strong>Confidence:</strong> <span id="confidence"></span></p>
            
            <h3>All Algorithm Probabilities:</h3>
            <div id="allProbs"></div>
        </div>
        
        <script>
            function toggleAdvanced() {
                const advSection = document.getElementById('advancedSection');
                const toggle = document.querySelector('.advanced-toggle');
                
                if (advSection.style.display === 'block') {
                    advSection.style.display = 'none';
                    toggle.textContent = '+ Advanced Options';
                } else {
                    advSection.style.display = 'block';
                    toggle.textContent = '- Advanced Options';
                }
            }
            
            function predict() {
                // Collect basic input data
                const data = {
                    NumProcesses: parseInt(document.getElementById('numProcesses').value),
                    AvgPriority: parseFloat(document.getElementById('avgPriority').value),
                    AvgBurstTime: parseFloat(document.getElementById('avgBurstTime').value),
                    AvgArrivalTime: parseFloat(document.getElementById('avgArrivalTime').value)
                };
                
                // Check if advanced fields have values
                const fcfsAwt = document.getElementById('fcfsAwt').value;
                const sjfAwt = document.getElementById('sjfAwt').value;
                const srtfAwt = document.getElementById('srtfAwt').value;
                const rrAwt = document.getElementById('rrAwt').value;
                
                if (fcfsAwt && sjfAwt && srtfAwt && rrAwt) {
                    data.FCFS_AWT = parseFloat(fcfsAwt);
                    data.SJF_AWT = parseFloat(sjfAwt);
                    data.SRTF_AWT = parseFloat(srtfAwt);
                    data.RR_AWT = parseFloat(rrAwt);
                    
                    // Calculate derived features
                    data.FCFS_SJF_Diff = data.FCFS_AWT - data.SJF_AWT;
                    data.FCFS_SRTF_Diff = data.FCFS_AWT - data.SRTF_AWT;
                    data.FCFS_RR_Diff = data.FCFS_AWT - data.RR_AWT;
                    data.SJF_SRTF_Diff = data.SJF_AWT - data.SRTF_AWT;
                }
                
                // Send prediction request
                fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(data)
                })
                .then(response => response.json())
                .then(result => {
                    if (result.error) {
                        alert(`Error: ${result.error}`);
                        return;
                    }
                    
                    // Display results
                    document.getElementById('bestAlgo').textContent = result.best_algorithm;
                    document.getElementById('confidence').textContent = (result.confidence * 100).toFixed(2) + '%';
                    
                    // Display all probabilities
                    const probsDiv = document.getElementById('allProbs');
                    probsDiv.innerHTML = '';
                    
                    result.all_probabilities.forEach(([algo, prob]) => {
                        const p = document.createElement('p');
                        p.innerHTML = `<strong>${algo}:</strong> ${(prob * 100).toFixed(2)}%`;
                        probsDiv.appendChild(p);
                    });
                    
                    // Show result section
                    document.getElementById('result').style.display = 'block';
                })
                .catch(error => {
                    alert(`Error: ${error.message}`);
                });
            }
        </script>
    </body>
    </html>
    """
    
    # Save the template to disk
    import os
    if not os.path.exists('templates'):
        os.makedirs('templates')
        
    with open('templates/index.html', 'w') as f:
        f.write(html)
    
    print("HTML template created at templates/index.html")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='CPU Scheduling Algorithm Predictor')
    parser.add_argument('--mode', type=str, default='cli', choices=['cli', 'web', 'train'],
                       help='Mode to run the application in (cli, web, or train)')
    parser.add_argument('--data', type=str, default='scheduling_data.csv',
                       help='Path to the dataset')
    parser.add_argument('--model', type=str, default='cpu_scheduling_model.pkl',
                       help='Path to save/load the model')
    parser.add_argument('--port', type=int, default=5000,
                       help='Port for the web app')
    
    args = parser.parse_args()
    
    if args.mode == 'cli':
        interactive_prediction()
    elif args.mode == 'web':
        create_html_template()
        app = create_web_app()
        app.run(debug=True, port=args.port)
    elif args.mode == 'train':
        train_and_save_model(args.data, args.model)