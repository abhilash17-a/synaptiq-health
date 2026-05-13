import pandas as pd
import numpy as np
import os
import torch
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pytorch_tabnet.tab_model import TabNetClassifier

class MentalHealthPredictor:
    def __init__(self, dataset_path=None, model_dir="saved_model"):
        # Use absolute path for robustness in production
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_dir = os.path.join(base_dir, model_dir)
        self.dataset_path = dataset_path
        self.model_path = os.path.join(self.model_dir, "tabnet_model")
        self.preprocessors_path = os.path.join(self.model_dir, "preprocessors.pkl")
        
        self.class_mapping = {
            'Poor': 'Worst',
            'Excellent': 'Poor',
            'Good': 'Better',
            'Fair': 'Good'
        }
        
        self.categorical_cols = [
            'Gender', 'Stress_Level', 'Support_Systems_Access', 
            'Work_Environment_Impact', 'Online_Support_Usage'
        ]
        
        self.tabnet_model = None
        self.encoders = {}
        self.target_encoder = None
        self.scaler = None
        self.features = None
        
        self.initialize()

    def initialize(self):
        if os.path.exists(self.preprocessors_path) and os.path.exists(self.model_path + ".zip"):
            self.load_saved_model()
        else:
            self.train_and_save()

    def load_saved_model(self):
        print(f"\nAttempting to load model from: {self.model_path}")
        self.tabnet_model = TabNetClassifier()
        
        # Explicitly check for the zip file to avoid extension confusion
        zip_path = self.model_path if self.model_path.endswith('.zip') else self.model_path + '.zip'
        
        if os.path.exists(zip_path):
            # TabNet's load_model usually expects the path WITHOUT .zip
            # but if it fails, we try with it.
            try:
                self.tabnet_model.load_model(self.model_path.replace('.zip', ''))
            except:
                self.tabnet_model.load_model(zip_path)
        else:
            raise FileNotFoundError(f"Model file not found at {zip_path}")
        
        with open(self.preprocessors_path, 'rb') as f:
            preprocessors = pickle.load(f)
            self.encoders = preprocessors['encoders']
            self.target_encoder = preprocessors['target_encoder']
            self.scaler = preprocessors['scaler']
            # Fallback if 'features' key is missing from old saves
            self.features = preprocessors.get('features')
            if not self.features:
                if self.dataset_path and os.path.exists(self.dataset_path):
                    print("Warning: 'features' key missing. Deriving from dataset...")
                    df = pd.read_csv(self.dataset_path)
                    self.features = [col for col in df.columns if col not in ['User_ID', 'Mental_Health_Status']]
                else:
                    # Hardcoded fallback for production if dataset is missing
                    print("Using hardcoded feature list fallback...")
                    self.features = [
                        'Age', 'Gender', 'Technology_Usage_Hours', 'Social_Media_Usage_Hours',
                        'Gaming_Hours', 'Screen_Time_Hours', 'Stress_Level', 'Sleep_Hours',
                        'Physical_Activity_Hours', 'Support_Systems_Access', 
                        'Work_Environment_Impact', 'Online_Support_Usage'
                    ]
        print("Model loaded successfully!")

    def train_and_save(self):
        print("\nNo saved model found. Training Model...")
        dataset = pd.read_csv(self.dataset_path)
        dataset['Mental_Health_Status'] = dataset['Mental_Health_Status'].map(self.class_mapping)
        dataset = dataset.drop(columns=['User_ID']).dropna()

        for col in self.categorical_cols:
            le = LabelEncoder()
            dataset[col] = le.fit_transform(dataset[col])
            self.encoders[col] = le

        self.target_encoder = LabelEncoder()
        dataset['Mental_Health_Status'] = self.target_encoder.fit_transform(dataset['Mental_Health_Status'])
        
        self.features = [col for col in dataset.columns if col != 'Mental_Health_Status']
        X = dataset[self.features]
        y = dataset['Mental_Health_Status']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.tabnet_model = TabNetClassifier(
            n_d=16, n_a=16,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=1e-2),
            scheduler_params={"step_size": 10, "gamma": 0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            mask_type='entmax'
        )

        self.tabnet_model.fit(
            X_train=X_train_scaled,
            y_train=y_train.values,
            eval_set=[(X_test_scaled, y_test.values)],
            eval_name=['validation'],
            eval_metric=['accuracy'],
            max_epochs=50,
            patience=10,
            batch_size=512,
            virtual_batch_size=64,
            num_workers=0,
            drop_last=False
        )

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        
        self.tabnet_model.save_model(self.model_path)
        preprocessors = {
            'encoders': self.encoders,
            'target_encoder': self.target_encoder,
            'scaler': self.scaler,
            'features': self.features
        }
        with open(self.preprocessors_path, 'wb') as f:
            pickle.dump(preprocessors, f)
        print(f"Model saved to '{self.model_dir}'")

    def predict(self, user_data):
        input_df = pd.DataFrame([user_data])
        for col in self.categorical_cols:
            if input_df[col].iloc[0] not in list(self.encoders[col].classes_):
                input_df[col] = self.encoders[col].classes_[0]
            input_df[col] = self.encoders[col].transform(input_df[col])
            
        input_df = input_df[self.features]
        input_scaled = self.scaler.transform(input_df)
        pred = self.tabnet_model.predict(input_scaled)
        return self.target_encoder.inverse_transform(pred)[0]

    def get_recommendations(self, user_data):
        recs = []
        if user_data['Sleep_Hours'] < 7:
            recs.append(f"Increase sleep to at least 7-8 hours (Currently: {user_data['Sleep_Hours']}h)")
        if user_data['Physical_Activity_Hours'] < 1:
            recs.append(f"Try to get at least 1 hour of physical activity daily (Currently: {user_data['Physical_Activity_Hours']}h)")
        if user_data['Screen_Time_Hours'] > 8:
            recs.append(f"Reduce total screen time below 8 hours (Currently: {user_data['Screen_Time_Hours']}h)")
        if user_data['Social_Media_Usage_Hours'] > 3:
            recs.append(f"Limit social media usage to under 2-3 hours (Currently: {user_data['Social_Media_Usage_Hours']}h)")
        if user_data['Support_Systems_Access'] == 'No':
            recs.append("Consider seeking support systems (counseling, family, or groups)")
        return recs

    def get_what_if(self, user_data):
        better_data = user_data.copy()
        better_data['Sleep_Hours'] = max(better_data['Sleep_Hours'], 8.0)
        better_data['Physical_Activity_Hours'] = max(better_data['Physical_Activity_Hours'], 1.5)
        better_data['Screen_Time_Hours'] = min(better_data['Screen_Time_Hours'], 5.0)
        better_data['Social_Media_Usage_Hours'] = min(better_data['Social_Media_Usage_Hours'], 1.5)
        better_data['Support_Systems_Access'] = 'Yes'
        better_data['Stress_Level'] = 'Low'
        return self.predict(better_data)

if __name__ == "__main__":
    # CLI Mode
    DATASET_PATH = r"C:\Users\abhil\.cache\kagglehub\datasets\waqi786\mental-health-and-technology-usage-dataset\versions\1\mental_health_and_technology_usage_2024.csv"
    predictor = MentalHealthPredictor(DATASET_PATH)
    
    while True:
        print("\n" + "="*60)
        print("ENTER YOUR VALUES TO TEST THE MODEL")
        print("="*60)
        try:
            user_data = {
                'Age': float(input("Age: ")),
                'Gender': input(f"Gender {list(predictor.encoders['Gender'].classes_)}: "),
                'Technology_Usage_Hours': float(input("Tech Usage Hours: ")),
                'Social_Media_Usage_Hours': float(input("Social Media Hours: ")),
                'Gaming_Hours': float(input("Gaming Hours: ")),
                'Screen_Time_Hours': float(input("Total Screen Time: ")),
                'Stress_Level': input(f"Stress Level {list(predictor.encoders['Stress_Level'].classes_)}: "),
                'Sleep_Hours': float(input("Sleep Hours: ")),
                'Physical_Activity_Hours': float(input("Physical Activity Hours: ")),
                'Support_Systems_Access': input(f"Support Systems {list(predictor.encoders['Support_Systems_Access'].classes_)}: "),
                'Work_Environment_Impact': input(f"Work Impact {list(predictor.encoders['Work_Environment_Impact'].classes_)}: "),
                'Online_Support_Usage': input(f"Online Support {list(predictor.encoders['Online_Support_Usage'].classes_)}: ")
            }
            
            result = predictor.predict(user_data)
            print(f"\nPREDICTED RESULT: {result}")
            
            recs = predictor.get_recommendations(user_data)
            print("\nRECOMMENDATIONS:")
            for r in recs: print(f"- {r}")
            
            what_if = predictor.get_what_if(user_data)
            print(f"\nWHAT-IF IMPROVEMENT: If habits improved, status could be: {what_if}")
            
        except Exception as e:
            print(f"Error: {e}")
            
        if input("\nTry again? (y/n): ").lower() != 'y': break