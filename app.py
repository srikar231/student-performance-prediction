from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("students.csv")

X = df[["Hours", "Sleep", "Attendance"]]
y = df["Marks"]

model = LinearRegression()
model.fit(X, y)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        hours = float(request.form["hours"])
        sleep = float(request.form["sleep"])
        attendance = float(request.form["attendance"])

        prediction = model.predict([[hours, sleep, attendance]])[0]

    return render_template("index.html", prediction=prediction)

app.run(debug=True)
