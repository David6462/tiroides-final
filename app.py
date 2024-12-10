import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Configuración de la página
st.set_page_config(
    page_title="Análisis de Mortalidad por Tiroides",
    page_icon="📊",
    layout="wide"
)

# Título principal
st.title("📊 Análisis de Mortalidad por Tiroides")
st.markdown("---")

# Función para cargar datos
@st.cache_data
def cargar_datos(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df['ano'] = pd.to_numeric(df['ano'], errors='coerce')
        df['n'] = pd.to_numeric(df['n'], errors='coerce')
        df['total'] = pd.to_numeric(df['total'], errors='coerce')
        df['dpto'] = df['dpto'].str.upper().str.strip()
        df['tasa_mortalidad'] = (df['n'] / df['total']) * 100000
        return df
    return None

# Sidebar para carga de datos
st.sidebar.header("Configuración")
uploaded_file = st.sidebar.file_uploader("Cargar archivo CSV", type=['csv'])

if uploaded_file is not None:
    df = cargar_datos(uploaded_file)
    
    # Tabs para diferentes análisis
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Pronósticos", "👥 Demografía", "🗺️ Análisis Regional", "📊 Visualizaciones", "🔮 Predicción Individual"])
    
    with tab1:
        st.header("Análisis de Pronósticos")
        
        # Selector de años para pronóstico
        anos_pronostico = st.slider("Años a pronosticar", 1, 10, 5)
        
        # Análisis con Prophet
        df_prophet = df.groupby('ano')['tasa_mortalidad'].mean().reset_index()
        df_prophet.columns = ['ds', 'y']
        df_prophet['ds'] = pd.to_datetime(df_prophet['ds'], format='%Y')
        
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='multiplicative',
            interval_width=0.95
        )
        
        with st.spinner('Calculando pronósticos...'):
            model.fit(df_prophet)
            future_dates = model.make_future_dataframe(periods=anos_pronostico, freq='Y')
            forecast = model.predict(future_dates)
            
            # Gráfico de pronóstico
            fig_forecast = plt.figure(figsize=(12, 6))
            plt.plot(df_prophet['ds'], df_prophet['y'], 'ko-', label='Datos históricos')
            plt.plot(forecast['ds'], forecast['yhat'], 'b-', label='Pronóstico')
            plt.fill_between(forecast['ds'], 
                           forecast['yhat_lower'], 
                           forecast['yhat_upper'],
                           color='blue', 
                           alpha=0.2, 
                           label='Intervalo de confianza 95%')
            plt.title('Pronóstico de Mortalidad por Tiroides')
            plt.xlabel('Año')
            plt.ylabel('Tasa de Mortalidad (por 100,000 habitantes)')
            plt.legend()
            st.pyplot(fig_forecast)
            
            # Tabla de pronósticos
            st.subheader("Valores pronosticados")
            future_forecast = forecast[forecast['ds'] > df_prophet['ds'].max()]
            future_forecast_display = pd.DataFrame({
                'Año': future_forecast['ds'].dt.year,
                'Tasa Esperada': future_forecast['yhat'].round(2),
                'Límite Inferior': future_forecast['yhat_lower'].round(2),
                'Límite Superior': future_forecast['yhat_upper'].round(2)
            })
            st.dataframe(future_forecast_display)
    
    with tab2:
        st.header("Análisis Demográfico")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribución por sexo
            fig_sex = plt.figure(figsize=(8, 8))
            df['sexo'].value_counts().plot(kind='pie', autopct='%1.1f%%')
            plt.title('Distribución por Sexo')
            st.pyplot(fig_sex)
        
        with col2:
            # Distribución por edad
            fig_age = plt.figure(figsize=(10, 6))
            df_edad = df.groupby('gru_edad')['n'].sum().sort_index()
            df_edad.plot(kind='bar')
            plt.title('Distribución por Grupo de Edad')
            plt.xlabel('Grupo de Edad')
            plt.ylabel('Número de Casos')
            plt.xticks(rotation=45)
            st.pyplot(fig_age)
    
    with tab3:
        st.header("Análisis Regional")
        
        # Selector de departamentos
        departamentos = sorted(df['dpto'].unique())
        dptos_seleccionados = st.multiselect(
            "Seleccionar departamentos para análisis",
            options=departamentos,
            default=departamentos[:5]
        )
        
        if dptos_seleccionados:
            # Análisis por departamento
            df_filtered = df[df['dpto'].isin(dptos_seleccionados)]
            
            # Tasa de mortalidad por departamento
            fig_regional = plt.figure(figsize=(12, 6))
            for dpto in dptos_seleccionados:
                dpto_data = df_filtered[df_filtered['dpto'] == dpto]
                plt.plot(dpto_data['ano'], dpto_data['tasa_mortalidad'], 'o-', label=dpto)
            
            plt.title('Tasa de Mortalidad por Departamento')
            plt.xlabel('Año')
            plt.ylabel('Tasa de Mortalidad')
            plt.legend(bbox_to_anchor=(1.05, 1))
            plt.grid(True, alpha=0.3)
            st.pyplot(fig_regional)
    
    with tab4:
        st.header("Visualizaciones Detalladas")
        
        # Selector de tipo de visualización
        viz_type = st.selectbox(
            "Seleccionar tipo de visualización",
            ["Heatmap de Mortalidad", "Correlaciones", "Tendencias Temporales"]
        )
        
        if viz_type == "Heatmap de Mortalidad":
            # Heatmap de últimos 10 años
            anos_max = df['ano'].max()
            df_reciente = df[df['ano'] > anos_max - 10]
            
            pivot_table = df_reciente.pivot_table(
                values='tasa_mortalidad',
                index='dpto',
                columns='ano',
                aggfunc='mean'
            )
            
            fig_heatmap = plt.figure(figsize=(12, 8))
            sns.heatmap(pivot_table, 
                       cmap='Reds',
                       annot=True,
                       fmt='.0f')
            plt.title('Tasas de Mortalidad por Departamento\nÚltimos 10 años')
            st.pyplot(fig_heatmap)
            
        elif viz_type == "Correlaciones":
            # Matriz de correlación
            columnas_numericas = ['ano', 'n', 'total', 'tasa_mortalidad']
            df_num = df[columnas_numericas]
            corr_matrix = df_num.corr()
            
            fig_corr = plt.figure(figsize=(8, 6))
            sns.heatmap(corr_matrix, 
                       annot=True,
                       cmap='coolwarm',
                       center=0,
                       fmt='.2f')
            plt.title('Matriz de Correlación')
            st.pyplot(fig_corr)
            
        else:
            # Tendencias temporales
            fig_trend = plt.figure(figsize=(12, 6))
            df.groupby('ano')['tasa_mortalidad'].mean().plot()
            plt.title('Tendencia Temporal de la Tasa de Mortalidad')
            plt.xlabel('Año')
            plt.ylabel('Tasa de Mortalidad')
            plt.grid(True, alpha=0.3)
            st.pyplot(fig_trend)
            
    with tab5:
        st.header("Predicción de Riesgo Individual")
        st.markdown("""
        Este módulo utiliza machine learning para estimar el riesgo individual basado en las características del paciente.
        Por favor, complete la siguiente información:
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            edad = st.selectbox(
                "Grupo de Edad",
                options=[
                    "0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34",
                    "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65+"
                ]
            )
            
            sexo = st.radio("Sexo", ["F", "M"])
            
            dpto_residencia = st.selectbox(
                "Departamento de Residencia",
                options=sorted(df['dpto'].unique()) if uploaded_file is not None else []
            )
            
        with col2:
            antecedentes_familiares = st.radio(
                "¿Antecedentes familiares de enfermedad tiroidea?",
                ["Sí", "No"]
            )
            
            imc = st.number_input(
                "Índice de Masa Corporal (IMC)",
                min_value=10.0,
                max_value=50.0,
                value=25.0,
                step=0.1
            )
            
            enfermedades_previas = st.multiselect(
                "Enfermedades previas",
                [
                    "Diabetes",
                    "Hipertensión",
                    "Enfermedad cardiovascular",
                    "Ninguna"
                ]
            )
        
        # Botón para calcular predicción
        if st.button("Calcular Riesgo"):
            # Aquí iría la lógica del modelo de predicción
            # Por ahora usaremos un cálculo simple basado en reglas
            
            # Factor base por edad
            edad_factor = {
                "0-4": 0.1, "5-9": 0.1, "10-14": 0.2, "15-19": 0.3,
                "20-24": 0.4, "25-29": 0.5, "30-34": 0.6, "35-39": 0.7,
                "40-44": 0.8, "45-49": 0.9, "50-54": 1.0, "55-59": 1.1,
                "60-64": 1.2, "65+": 1.3
            }
            
            # Calcular score base
            risk_score = edad_factor[edad]
            
            # Ajustar por otros factores
            if sexo == "F":
                risk_score *= 1.2  # Las mujeres tienen mayor riesgo
            if antecedentes_familiares == "Sí":
                risk_score *= 1.5
            if imc > 30:
                risk_score *= 1.2
            if len(enfermedades_previas) > 0 and "Ninguna" not in enfermedades_previas:
                risk_score *= (1 + (len(enfermedades_previas) * 0.1))
                
            # Normalizar score a probabilidad
            prob = min(risk_score / 3, 0.99)
            
            # Mostrar resultados
            st.markdown("---")
            st.subheader("Resultados del Análisis")
            
            # Mostrar gauge chart para el riesgo
            
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Nivel de Riesgo"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps' : [
                        {'range': [0, 33], 'color': "lightgreen"},
                        {'range': [33, 66], 'color': "yellow"},
                        {'range': [66, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': prob * 100
                    }
                }
            ))
            
            st.plotly_chart(fig)
            
            # Mostrar recomendaciones basadas en el riesgo
            st.subheader("Recomendaciones")
            if prob < 0.33:
                st.success("""
                📌 Riesgo Bajo
                - Mantener chequeos regulares anuales
                - Continuar con hábitos saludables
                - Monitorear cualquier cambio en síntomas
                """)
            elif prob < 0.66:
                st.warning("""
                📌 Riesgo Moderado
                - Programar revisión con endocrinólogo
                - Realizar pruebas de función tiroidea cada 6 meses
                - Evaluar factores de riesgo modificables
                - Considerar cambios en el estilo de vida
                """)
            else:
                st.error("""
                📌 Riesgo Alto
                - Consultar especialista de inmediato
                - Realizar evaluación completa de la tiroides
                - Considerar pruebas adicionales (ultrasonido, etc.)
                - Seguimiento cercano y regular
                - Implementar cambios en el estilo de vida
                """)
                
            # Mostrar disclaimer
            st.markdown("---")
            st.caption("""
            ⚠️ IMPORTANTE: Esta predicción es solo una estimación basada en factores de riesgo conocidos.
            No constituye un diagnóstico médico. Siempre consulte con un profesional de la salud para
            una evaluación adecuada.
            """)
    
    # Descarga de datos procesados
    st.sidebar.markdown("---")
    if st.sidebar.button("Descargar Datos Procesados"):
        csv = df.to_csv(index=False)
        st.sidebar.download_button(
            label="Confirmar Descarga",
            data=csv,
            file_name="datos_procesados.csv",
            mime="text/csv"
        )
else:
    st.info("👆 Por favor, carga un archivo CSV para comenzar el análisis.")
    st.markdown("""
    El archivo debe contener las siguientes columnas:
    - ano: Año del registro
    - dpto: Departamento
    - sexo: Género
    - gru_edad: Grupo de edad
    - n: Número de casos
    - total: Población total
    """)