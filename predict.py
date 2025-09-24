import joblib
import numpy as np
from tensorflow.keras.models import load_model
from fastapi import FastAPI
from pydantic import BaseModel

rf = joblib.load("rf_fraud_model.pkl")
xgb = joblib.load("xgb_fraud_model.pkl")
meta = joblib.load("meta_model.pkl")
autoencoder = load_model("autoencoder_model.h5", compile=False)
lstm_model = load_model("lstm_model.h5", compile=False)
gru_model = load_model("gru_model.h5", compile=False)
scaler = joblib.load("scaler.pkl")

app = FastAPI()

class Transaction(BaseModel):
    data: list

@app.post("/predict")
def predict(transaction: Transaction):
    x = np.array(transaction.data).reshape(1, -1)
    x_scaled = scaler.transform(x)
    x_seq = x_scaled.reshape((1, 1, x_scaled.shape[1]))

    y_rf = rf.predict(x_scaled)[0]
    y_xgb = xgb.predict(x_scaled)[0]

    x_ae_pred = autoencoder.predict(x_scaled)
    ae_error = np.mean(np.square(x_scaled - x_ae_pred), axis=1)
    y_ae = int(ae_error > np.percentile(ae_error, 95))

    y_lstm = int((lstm_model.predict(x_seq) > 0.5)[0][0])
    y_gru = int((gru_model.predict(x_seq) > 0.5)[0][0])

    stacked = np.array([[y_rf, y_xgb, y_ae, y_lstm, y_gru]])
    y_final = int(meta.predict(stacked)[0])

    return {"fraud": bool(y_final)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("predict:app", host="127.0.0.1", port=8000, reload=True)
