📈 Microsoft Stock Price Prediction using LSTM

A Deep Learning project built using TensorFlow LSTM to predict Microsoft (MSFT) stock closing prices based on historical data. The model applies data preprocessing, scaling, sliding-window sequence creation, and multi-layer LSTM architecture for time-series forecasting.

🚀 Project Overview

This project demonstrates how to use Long Short-Term Memory (LSTM) neural networks for stock price prediction.
It includes:

Data preprocessing

Exploratory data analysis (EDA)

Feature correlation heatmap

Sliding window dataset creation

Training an LSTM model

Predicting closing prices

Visualizing predictions vs actual values

📂 Dataset

The dataset used is:

MicrosoftStock.csv


Required Columns:

date

open

high

low

close

volume

Make sure the file is placed in the root of the project or update the file path accordingly.

🛠️ Technologies Used

Python

TensorFlow / Keras

Pandas

NumPy

Scikit-learn

Matplotlib

Seaborn

📦 Installation
1. Clone the repository
git clone https://github.com/vijaymangore/Microsoft-Stock-Price-Prediction-using-LSTM.git
cd stock-lstm

2. Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate        # For Linux/Mac
venv\Scripts\activate           # For Windows

3. Install dependencies
pip install -r requirements.txt

📘 Usage

Place your CSV file:

MicrosoftStock.csv


Run the script:

python main.py


The script performs:

Data reading

Conversion of dates

Heatmap visualization

LSTM model training

Plotting actual vs predicted stock prices

📊 Visualizations Included
✔️ Open vs Close price over time
✔️ Trading Volume over time
✔️ Correlation Heatmap
✔️ Predicted vs Actual stock closing price

These plots help understand market behavior and evaluate model accuracy.

🤖 Model Architecture

The LSTM model includes:

LSTM Layer (64 units, return sequences)

LSTM Layer (64 units)

Dense Layer (128 neurons, ReLU)

Dropout Layer (0.50)

Output Dense Layer (1 neuron)

Optimizer: Adam
Loss: MAE
Metric: RootMeanSquaredError

📈 Sample Prediction Output

A final graph is generated:

Blue → Training Data

Orange → Actual Testing Data

Red → Predicted Closing Prices

This helps visualize how well the model forecasts unseen values.

🗂️ Project Structure
├── MicrosoftStock.csv
├── main.py
├── requirements.txt
└── README.md

🔮 Future Improvements

Add hyperparameter tuning (Keras Tuner)

Add GRU/BiLSTM models

Use multivariate features

Deploy using Streamlit

Add rolling validation performance metrics

🙌 Contributing

Pull requests are welcome. For major changes, open an issue first to discuss your idea.

📜 License

This project is open-source and available under the MIT License.
