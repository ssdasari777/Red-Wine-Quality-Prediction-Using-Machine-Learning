# Red Wine Quality Prediction Using Machine Learning

This project was developed as part of my Bachelor’s thesis. It uses machine learning algorithms to predict the quality of red wine based on physicochemical properties.

The final model is deployed using Flask to provide a simple web interface where users can input wine characteristics and get a quality prediction.

## Project Highlights

- Dataset: Red Wine Quality dataset from the UCI Machine Learning Repository
- Target: Predict wine quality scores (0–10)
- Algorithms tested: SVM, Naive Bayes, Random Forest
- Best Result: Random Forest with ~80.2% accuracy
- Deployment: Flask-based web application for user-friendly interaction

## Dataset Features

The dataset includes 11 physicochemical attributes:
- Fixed Acidity
- Volatile Acidity
- Citric Acid
- Residual Sugar
- Chlorides
- Free Sulfur Dioxide
- Total Sulfur Dioxide
- Density
- pH
- Sulphates
- Alcohol

## Folder Structure
- app.py # Flask app
- model_training.py # ML model training script
- model.pkl # Trained model file
- templates/ # HTML files
- static/ # CSS and image files
- requirements.txt # Python dependencies
- README.md


## Future Improvements

- Improve model accuracy with hyperparameter tuning
- Add support for white wine dataset
- Containerize with Docker
- Deploy to cloud (Heroku / Render)

## License

This project is licensed under the MIT License.
