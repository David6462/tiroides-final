import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Pronósticos", "👥 Demografía", "🗺️ Análisis Regional", "📊 Visualizaciones"])
    
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