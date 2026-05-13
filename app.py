from flask import Flask, render_template, request, jsonify
from dl import MentalHealthPredictor
import os

app = Flask(__name__)

# Initialize Predictor
LOCAL_DATASET = r"C:\Users\abhil\.cache\kagglehub\datasets\waqi786\mental-health-and-technology-usage-dataset\versions\1\mental_health_and_technology_usage_2024.csv"
DATASET_PATH = LOCAL_DATASET if os.path.exists(LOCAL_DATASET) else "mental_health_and_technology_usage_2024.csv"

predictor = MentalHealthPredictor(DATASET_PATH if os.path.exists(DATASET_PATH) else None)

@app.route('/')
def index():
    return render_template('index.html', options={
        'Gender': list(predictor.encoders['Gender'].classes_),
        'Stress_Level': list(predictor.encoders['Stress_Level'].classes_),
        'Support_Systems_Access': list(predictor.encoders['Support_Systems_Access'].classes_),
        'Work_Environment_Impact': list(predictor.encoders['Work_Environment_Impact'].classes_),
        'Online_Support_Usage': list(predictor.encoders['Online_Support_Usage'].classes_)
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        # Convert numeric fields
        numeric_fields = [
            'Age', 'Technology_Usage_Hours', 'Social_Media_Usage_Hours', 
            'Gaming_Hours', 'Screen_Time_Hours', 'Sleep_Hours', 'Physical_Activity_Hours'
        ]
        for field in numeric_fields:
            data[field] = float(data[field])
            
        prediction = predictor.predict(data)
        recommendations = predictor.get_recommendations(data)
        what_if = predictor.get_what_if(data)
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'recommendations': recommendations,
            'what_if': what_if
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
