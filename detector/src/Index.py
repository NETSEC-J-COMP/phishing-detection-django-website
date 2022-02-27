# Importing libraries
import joblib
import detector.src.Inputscript as Inputscript

def main(url):
    # Load the pickle file
    classifier = joblib.load("detector/src/final_model/rf_final.pkl")
    
    # Checking and Predicting
    checkprediction = Inputscript.main(url)
    prediction = classifier.predict(checkprediction)

    return(prediction[0])