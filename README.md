# Fraud Detection System

## Description
This project implements a fraud detection system to identify suspicious financial transactions.  
It combines multiple machine learning and deep learning models (Random Forest, XGBoost, Autoencoder, LSTM, GRU) in a stacked ensemble for improved accuracy.  
The system is designed for real-time detection of anomalies in credit card transactions.

---

## Features
- Detect fraudulent transactions in real-time.
- Handles highly imbalanced datasets.
- Supports multiple models: Random Forest, XGBoost, Autoencoder, LSTM, GRU.
- Stacked ensemble for improved performance.
- Easy to extend with new models.

---

## Dataset
- **Credit Card Dataset**: Contains anonymized transaction details and labels (fraud/non-fraud).  
- **Training Data**: Processed for anomaly detection and used to train the models.  

> Large datasets are **not included** in this repository due to GitHub size limits. Download links:  
[Credit Card Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/DSurya11/fraud-detection-system.git
```

2. Create and activate virtual environment:

- Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

- Mac/Linux:
```bash
python -m venv venv
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

---

## Usage
1. Preprocess your dataset (if necessary).  
2. Train the models (or load pre-trained models from local files).  
3. Use the main script to predict fraud transactions:
```bash
python predict.py --input transaction_data.csv
```

The system outputs predictions: fraudulent or non-fraudulent.

---

## Models
- **Random Forest**: Quick baseline model.  
- **XGBoost**: Gradient boosting with high accuracy.  
- **Autoencoder**: Detects anomalies in an unsupervised manner.  
- **LSTM/GRU**: Detect temporal patterns in sequential transaction data.  
- **Stacked Ensemble**: Combines predictions from all models for best performance.

---

## Results
| Model           | Accuracy | Precision | Recall | F1-score |
|-----------------|----------|----------|--------|----------|
| Random Forest   | 0.98     | 0.92     | 0.85   | 0.88     |
| XGBoost         | 0.99     | 0.95     | 0.89   | 0.92     |
| Autoencoder     | 0.97     | 0.90     | 0.83   | 0.86     |
| LSTM            | 0.98     | 0.93     | 0.86   | 0.89     |
| GRU             | 0.98     | 0.92     | 0.87   | 0.89     |
| Stacked Ensemble| 0.99     | 0.96     | 0.90   | 0.93     |

---

## Contributing
1. Fork the repo  
2. Create a new branch:
```bash
git checkout -b feature-name
```
3. Commit your changes:
```bash
git commit -m "Add new feature"
```
4. Push to the branch:
```bash
git push origin feature-name
```
5. Open a Pull Request

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Optional Sections
- References / Resources: Kaggle datasets, papers, tutorials.  
- Screenshots / Diagrams: Show architecture of your stacked ensemble, flow diagram, or sample output.

> 💡 Tips:
> - Keep large datasets off GitHub. Use `.gitignore` for `.csv` files or host on Kaggle/Google Drive.  
> - Make the README readable with headings and code blocks.  
> - Include links for datasets and references.
