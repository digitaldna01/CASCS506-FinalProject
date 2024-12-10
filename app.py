# add imports below
from flask import Flask, render_template, request, send_file
import pandas as pd
import matplotlib.pyplot as plt
import io

app = Flask(__name__)


df = pd.read_csv("./data/breast_cancer.csv")  # Replace with your dataset path data/breast_cancer.csv
features = list(df.columns[:1]) + list(df.columns[2:])
print(features)

# Define the main route
@app.route('/')
def index():
    return render_template('index.html', features=features)

# add other routes below

# letting user plto different features
@app.route('/plot', methods=['POST'])
def plot():
    feature1 = request.form.get('feature1')
    feature2 = request.form.get('feature2')
    label_column = 'diagnosis'  # Adjust based on your dataset

    plt.switch_backend('agg')

    # Create a new figure without the GUI
    fig, ax = plt.subplots(figsize=(6, 4)) 
    # for diagnosis, color in zip(['benign', 'malignant'], ['blue', 'red']):
    #     subset = df[df[label_column] == diagnosis]
    #     ax.scatter(subset[feature1], subset[feature2], label=diagnosis, color=color)
    color_map = {'M': 'red', 'B': 'blue'}
    unique_diagnoses = df[label_column].unique()
    # for _, row in df.iterrows():
    #     color = 'blue' if row[label_column] == 'M' else 'red'
    #     ax.scatter(row[feature1], row[feature2], color=color)
    for diagnosis in unique_diagnoses:
        subset = df[df[label_column] == diagnosis]
        ax.scatter(subset[feature1], subset[feature2], label=diagnosis, color=color_map[diagnosis])
    ax.set_xlabel(feature1)
    ax.set_ylabel(feature2)

    ax.legend([color_map[diagnosis] for diagnosis in unique_diagnoses], labels=['malignant', 'benign'])
    plt.tight_layout()

    # Save plot to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    print("plotted")

    return send_file(img, mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True)