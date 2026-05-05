from flask import Flask, render_template
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("students.csv")

X = df[["Hours", "Sleep", "Attendance"]]
y = df["Marks"]

model = LinearRegression()
model.fit(X, y)

prediction = model.predict([[5, 7, 90]])[0]

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html", prediction=round(prediction, 2))

app.run(debug=True)
