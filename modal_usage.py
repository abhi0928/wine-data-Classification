from joblib import load
import numpy as np \

model = load('modal_usage.joblib')

features = np.array([[-0.1970216, -0.82452235, -1.10503332, -0.53383847, -0.2064146, -0.36417712,
  0.09169564, -0.27213203,  0.3828889, -0.83501275, -0.50350745]])

print(model.predict(features))
