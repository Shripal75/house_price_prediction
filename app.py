from flask import Flask, render_template, request
import numpy as np
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load your trained model
model = joblib.load('first_house_model.pkl')

# Keep history of predictions for graphs
PREDICTION_HISTORY = []      # each item: {"area": ..., "price": ...}


@app.route('/')
def home():
    # Render the Home Page (no prediction logic here)
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None
    error = None
    advice_text = None

    if request.method == 'POST':
        try:
            overall_qual = int(request.form['overall_qual'])
            gr_liv_area = float(request.form['gr_liv_area'])   # we'll use this as X (input)
            garage_cars = int(request.form['garage_cars'])
            total_bsmt_sf = float(request.form['total_bsmt_sf'])
            full_bath = int(request.form['full_bath'])
            year_built = int(request.form['year_built'])

            features = np.array([[overall_qual, gr_liv_area, garage_cars,
                                  total_bsmt_sf, full_bath, year_built]])
            prediction = model.predict(features)
            price = float(prediction[0])   # Y (output)

            # ---------- SIMPLE "AI" ADVICE BASED ON PRICE ----------
            if price < 3000000:
                advice_text = "Budget-friendly property. Good for first-time buyers. 🙂"
            elif price < 8000000:
                advice_text = "Mid-range house with balanced features. 👍"
            else:
                advice_text = "High-end property. Check locality & amenities carefully. 💼"

            # ---------- SAVE TO HISTORY FOR GRAPHS ----------
            PREDICTION_HISTORY.append({
                "area": gr_liv_area,
                "price": price
            })
            # keep only last 30 predictions
            if len(PREDICTION_HISTORY) > 30:
                PREDICTION_HISTORY.pop(0)

        except Exception as e:
            error = f"Error: {str(e)}"

    return render_template(
        'predict.html',
        prediction=prediction[0] if prediction is not None else None,
        advice_text=advice_text,
        error=error
    )


@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/create-account')
def create_account():
    return render_template('create-account.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/analytics')
def analytics():
    # Prepare scatter data
    areas = [item["area"] for item in PREDICTION_HISTORY]
    prices = [item["price"] for item in PREDICTION_HISTORY]

    # Build data for Chart.js scatter: [{x: area, y: price}, ...]
    scatter_points = [{"x": a, "y": p} for a, p in zip(areas, prices)]

    return render_template(
        'analytics.html',
        scatter_points=scatter_points,
        total_points=len(scatter_points)
    )




if __name__ == '__main__':
    app.run(debug=True)
