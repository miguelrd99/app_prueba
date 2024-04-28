import joblib 
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris 

data = load_iris()
X, y = data.data, data.target

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, "/Users/martin_ramiro/Desktop/app_prueba/backend/model.joblib")