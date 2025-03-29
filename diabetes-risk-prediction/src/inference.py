def predict_sample(model, sample):
    """
    Predict the class for a single sample using the provided model.
    
    Returns:
        The predicted class.
    """
    prediction = model.predict(sample)
    return prediction
