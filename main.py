from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
import json
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional
import joblib
import os

app = FastAPI()

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especifica los orígenes permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelos Pydantic para validación de datos
class PatientData(BaseModel):
    edad: int
    sexo: str
    departamento: str
    grupo_edad: str
    # Agregar más campos según sea necesario

class PredictionResponse(BaseModel):
    probabilidad_supervivencia: float
    factores_riesgo: List[str]
    recomendaciones: List[str]

# Variables globales para modelos entrenados
prophet_model = None
survival_model = None
scaler = None
DATASET_PATH = "datos.csv"

def load_models():
    """Carga o entrena los modelos necesarios"""
    global prophet_model, survival_model, scaler
    
    # Cargar datos
    df = pd.read_csv(DATASET_PATH)
    
    # Entrenar Prophet
    df_prophet = df.groupby('ano')['tasa_mortalidad'].mean().reset_index()
    df_prophet.columns = ['ds', 'y']
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'], format='%Y')
    
    prophet_model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode='multiplicative'
    )
    prophet_model.fit(df_prophet)
    
    # Guardar modelo Prophet
    prophet_model.save('prophet_model.json')
    
    # Aquí entrenarías tu modelo de supervivencia
    # Este es un ejemplo simplificado
    survival_features = ['grupo_edad_encoded', 'sexo_encoded', 'dpto_encoded']
    # survival_model = RandomForestClassifier()
    # survival_model.fit(X_train, y_train)
    # joblib.dump(survival_model, 'survival_model.joblib')

def plot_to_json(fig):
    """Convierte una figura de matplotlib a JSON"""
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode()
    buf.close()
    plt.close(fig)
    return img_str

@app.on_event("startup")
async def startup_event():
    """Carga los modelos al iniciar la aplicación"""
    load_models()

@app.post("/predict/survival")
async def predict_survival(patient: PatientData):
    """Endpoint para predicción de supervivencia individual"""
    try:
        # Procesar datos del paciente
        # Hacer predicción
        # Este es un ejemplo simplificado
        prediction = {
            "probabilidad_supervivencia": 0.85,  # Ejemplo
            "factores_riesgo": [
                "Edad avanzada",
                "Ubicación geográfica"
            ],
            "recomendaciones": [
                "Seguimiento regular",
                "Control de factores de riesgo"
            ]
        }
        return JSONResponse(content=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analysis/complete")
async def get_complete_analysis():
    """Endpoint para obtener el análisis completo"""
    try:
        df = pd.read_csv(DATASET_PATH)
        
        # Generar todas las visualizaciones
        visualizations = {}
        
        # 1. Pronóstico general
        future = prophet_model.make_future_dataframe(periods=60, freq='M')
        forecast = prophet_model.predict(future)
        
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat'],
            name='Pronóstico'
        ))
        visualizations['forecast'] = fig_forecast.to_json()
        
        # 2. Análisis demográfico
        fig_demo, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        df['sexo'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax1)
        df.groupby('gru_edad')['n'].sum().plot(kind='bar', ax=ax2)
        visualizations['demografia'] = plot_to_json(fig_demo)
        
        # Agregar más visualizaciones según sea necesario
        
        return JSONResponse(content={
            "visualizations": visualizations,
            "summary_stats": {
                "total_cases": len(df),
                "mortality_rate": df['tasa_mortalidad'].mean(),
                "trend": "increasing" if forecast['trend'].iloc[-1] > forecast['trend'].iloc[0] else "decreasing"
            }
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update/dataset")
async def update_dataset(file: UploadFile = File(...)):
    """Endpoint para actualizar el dataset y reentrenar los modelos"""
    try:
        content = await file.read()
        with open(DATASET_PATH, "wb") as f:
            f.write(content)
        load_models()
        return JSONResponse(content={"message": "Dataset actualizado y modelos reentrenados"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)