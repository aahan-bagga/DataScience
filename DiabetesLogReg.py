import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, GridSearchCV
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class diabetesLogReg:
    df = pd.read_csv("/Users/aahan_bagga/Desktop/diabetes_data.csv")
    X=df.drop(["Outcome"], axis=1)
    Y=df["Outcome"]
    preg = 0
    glucose = 0
    BP = 0
    skinThickness = 0
    insulin = 0
    bmi = 0
    diabetesPedigreeFunction = 0
    age = 0
    def __init__(self, p, g, BP, ST, I, BMI, DPF, age):
        self.preg = p
        self.glucose = g
        self.BP = BP
        self.skinThickness = ST
        self.insulin = I
        self.bmi = BMI
        self.diabetesPedigreeFunction = DPF
        self.age = age


    def preprocessing(self):
        #K-fold cross validation
        kf = KFold(n_splits = 9, shuffle = True, random_state = 19)

        global X_train, X_test, Y_train, Y_test
        for training_index, testing_index in kf.split(self.X):
            X_train, X_test = self.X.iloc[training_index], self.X.iloc[testing_index]
            Y_train, Y_test = self.Y.iloc[training_index], self.Y.iloc[testing_index]


        #Noramlization marginally better than Standardization
        scaler = MinMaxScaler()
        global x_train_s, x_test_s
        x_train_s = scaler.fit_transform(X_train)
        x_test_s = scaler.transform(X_test)

    def train(self):
        #HYPERPARAM TUNING

        global model
        model = LogisticRegression(max_iter = 50, class_weight="balanced")

        param_grid = {
            "C": np.logspace(-3,3,7),
            "penalty": ["l1", "l2"],
            "solver": ["newton-cg", "lbfgs", "sag", "saga", "liblinear"],
            "max_iter": [10,20,30,40,50]
        }

        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')

        grid_search.fit(x_train_s, Y_train)
        print(f"Best hyperparameters: {grid_search.best_params_}")

        global tuned_model
        tuned_model = grid_search.best_estimator_
        
        
        model.fit(x_train_s,Y_train)
        y_pred = model.predict(x_test_s)
        t_y_pred = tuned_model.predict(x_test_s)
        return (accuracy_score(Y_test, y_pred) * 100), (accuracy_score(Y_test, t_y_pred) * 100)




    
    def diabetes_pred(self):
        prob = tuned_model.predict_proba([[self.preg, self.glucose, self.BP, self.skinThickness, self.insulin, self.bmi, self.diabetesPedigreeFunction, self.age]])
        print(prob)
        if prob[0,1] > 0.5:
            return "Diabetes"
        else:
            return "No Diabetes"
    
    #def decision_boundary_graph():
    


d = diabetesLogReg(2,126,45,23,340,30,0.12,29)

d.preprocessing()
print(d.train())
print(d.diabetes_pred())