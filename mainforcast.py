from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Sample data
data = {
    "timestamp": ["1/1/2023", "2/1/2023", "3/1/2023", "4/1/2023", "5/1/2023", "6/1/2023", "7/1/2023", "8/1/2023",
                  "9/1/2023", "10/1/2023", "11/1/2023", "12/1/2023", "1/1/2024", "2/1/2024", "3/1/2024", "4/1/2024",
                  "5/1/2024"],
    "SLA_Passed_Incident_%": [81, 77, 73, 74, 81, 72, 77, 81, 93, 78, 84, 88, 90, 91, 92, 89, 88],
    "TeamSize": [5] * 17,
    "Leavedays": [6, 10, 11, 9, 3, 12, 7, 6, 4, 9, 7, 11, 8, 6, 10, 6, 10],
    "Incident": [270, 270, 280, 290, 300, 310, 310, 320, 290, 330, 340, 340, 350, 380, 380, 410, 420],
    "KB_Coverage": [5, 5, 5, 10, 20, 30, 30, 35, 35, 35, 40, 45, 50, 60, 70, 75, 85],
    "KB_Automatino": [0, 20, 30, 50, 50, 50, 50, 60, 60, 70, 80, 80, 80, 80, 80, 80, 80]
}

df = pd.DataFrame(data)

def forecast_metrics(df, features, target):
    X = df[features]
    y = df[target]
    model = LinearRegression()
    model.fit(X, y)
    next_month_data = np.array([df[feature].iloc[-1] for feature in features]).reshape(1, -1)
    forecast_value = model.predict(next_month_data)[0]
    return forecast_value

@app.route('/')
def index():
    sla_forecast = forecast_metrics(df, ["Incident", "KB_Coverage", "KB_Automatino"], "SLA_Passed_Incident_%")
    leave_forecast = forecast_metrics(df, ["SLA_Passed_Incident_%", "Incident", "KB_Coverage", "KB_Automatino"], "Leavedays")
    incident_forecast = forecast_metrics(df, ["SLA_Passed_Incident_%", "Leavedays", "KB_Coverage", "KB_Automatino"], "Incident")

    # Plotting the SLA_Passed_Incident_% with forecast
    plt.figure(figsize=(10, 5))
    plt.plot(df['timestamp'], df['SLA_Passed_Incident_%'], marker='o', label='Historical Data')
    plt.plot(['New Month'], [sla_forecast], color='r', linestyle='--', marker='o', label='Forecast')
    plt.xticks(rotation=45)
    plt.xlabel('Timestamp')
    plt.ylabel('SLA Passed Incident %')
    plt.legend()
    plt.tight_layout()
    img1 = io.BytesIO()
    plt.savefig(img1, format='png')
    img1.seek(0)
    plot_url1 = base64.b64encode(img1.getvalue()).decode('utf8')

    # Plotting the forecast for leave
    plt.figure(figsize=(10, 5))
    plt.plot(df['timestamp'], df['Leavedays'], marker='o', label='Historical Data')
    plt.plot(['New Month'], [leave_forecast], color='r', linestyle='--', marker='o', label=f'Forecast: {leave_forecast:.2f}')
    plt.xticks(rotation=45)
    plt.xlabel('Timestamp')
    plt.ylabel('Leave Days')
    plt.legend()
    plt.tight_layout()
    img2 = io.BytesIO()
    plt.savefig(img2, format='png')
    img2.seek(0)
    plot_url2 = base64.b64encode(img2.getvalue()).decode('utf8')

    # Plotting the forecast for incident
    plt.figure(figsize=(10, 5))
    plt.plot(df['timestamp'], df['Incident'], marker='o', label='Historical Data')
    plt.plot(['New Month'], [incident_forecast], color='r', linestyle='--', marker='o', label=f'Forecast: {incident_forecast:.2f}')
    plt.xticks(rotation=45)
    plt.xlabel('Timestamp')
    plt.ylabel('Incident Count')
    plt.legend()
    plt.tight_layout()
    img3 = io.BytesIO()
    plt.savefig(img3, format='png')
    img3.seek(0)
    plot_url3 = base64.b64encode(img3.getvalue()).decode('utf8')

    return render_template('sla_forecasting.html', plot_url1=plot_url1, plot_url2=plot_url2, plot_url3=plot_url3, tables=[df.to_html(classes='data', header="true")])

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
