import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import plotly.express as px

# Configuración de la página
st.set_page_config(
    page_title="Análisis de Mortalidad por Tiroides",
    page_icon="📊",
    layout="wide"
)

# Configuración de estilo
plt.style.use('classic')
sns.set_palette("husl")

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
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Pronósticos",
        "👥 Demografía",
        "🗺️ Análisis Regional",
        "📊 Visualizaciones",
        "🔮 Predicción Individual"
    ])
    
    with tab1:
        st.header("Análisis de Pronósticos")
        
        # Análisis con Prophet
        df_prophet = df.groupby('ano')['tasa_mortalidad'].mean().reset_index()
        df_prophet.columns = ['ds', 'y']
        df_prophet['ds'] = pd.to_datetime(df_prophet['ds'], format='%Y')
        
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False
        )
        
        with st.spinner('Calculando pronósticos...'):
            model.fit(df_prophet)
            future_dates = model.make_future_dataframe(periods=5, freq='Y')
            forecast = model.predict(future_dates)
            
            # Gráfico de pronóstico con Plotly
            fig_forecast = go.Figure()
            
            # Datos históricos
            fig_forecast.add_trace(go.Scatter(
                x=df_prophet['ds'],
                y=df_prophet['y'],
                name='Datos históricos',
                mode='markers+lines',
                line=dict(color='blue')
            ))
            
            # Línea de pronóstico
            fig_forecast.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat'],
                name='Pronóstico',
                mode='lines',
                line=dict(color='red', dash='dash')
            ))
            
            # Intervalo de confianza
            fig_forecast.add_trace(go.Scatter(
                x=forecast['ds'].tolist() + forecast['ds'].tolist()[::-1],
                y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(0,100,255,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Intervalo de confianza 95%'
            ))
            
            fig_forecast.update_layout(
                title='Pronóstico de Mortalidad por Tiroides',
                xaxis_title='Año',
                yaxis_title='Tasa de Mortalidad (por 100,000 habitantes)',
                showlegend=True
            )
            
            st.plotly_chart(fig_forecast)
            
            # Componentes de Prophet
            fig_components = model.plot_components(forecast)
            st.pyplot(fig_components)
    
    with tab2:
        st.header("Análisis Demográfico")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Gráfico de distribución por sexo con Plotly
            sex_counts = df['sexo'].value_counts()
            fig_sex = go.Figure(data=[go.Pie(
                labels=sex_counts.index,
                values=sex_counts.values,
                hole=0.3
            )])
            fig_sex.update_layout(title='Distribución por Sexo')
            st.plotly_chart(fig_sex)
        
        with col2:
            # Distribución por edad con Plotly
            age_counts = df.groupby('gru_edad')['n'].sum()
            fig_age = px.bar(
                x=age_counts.index,
                y=age_counts.values,
                title='Distribución por Grupo de Edad'
            )
            fig_age.update_layout(
                xaxis_title='Grupo de Edad',
                yaxis_title='Número de Casos'
            )
            st.plotly_chart(fig_age)
        
        # Distribución por edad y sexo
        pivot_edad_sexo = df.pivot_table(
            values='n',
            index='gru_edad',
            columns='sexo',
            aggfunc='sum'
        ).fillna(0)
        
        fig_edad_sexo = go.Figure(data=[
            go.Bar(name='Hombres', x=pivot_edad_sexo.index, y=pivot_edad_sexo['M']),
            go.Bar(name='Mujeres', x=pivot_edad_sexo.index, y=pivot_edad_sexo['F'])
        ])
        fig_edad_sexo.update_layout(
            barmode='stack',
            title='Distribución de Mortalidad por Grupo de Edad y Sexo',
            xaxis_title='Grupo de Edad',
            yaxis_title='Número de Casos'
        )
        st.plotly_chart(fig_edad_sexo)
    
    with tab3:
        st.header("Análisis Regional")
        
        # Heatmap de tasas de mortalidad
        anos_max = df['ano'].max()
        df_reciente = df[df['ano'] > anos_max - 10]
        
        pivot_table = df_reciente.pivot_table(
            values='tasa_mortalidad',
            index='dpto',
            columns='ano',
            aggfunc='mean'
        )
        
        fig_heatmap = px.imshow(
            pivot_table,
            aspect='auto',
            color_continuous_scale='Reds',
            title='Tasas de Mortalidad por Departamento (últimos 10 años)'
        )
        fig_heatmap.update_layout(
            xaxis_title='Año',
            yaxis_title='Departamento'
        )
        st.plotly_chart(fig_heatmap)
        
        # Análisis por departamento
        dept_stats = df.groupby('dpto').agg({
            'tasa_mortalidad': 'mean',
            'total': 'mean',
            'n': 'sum'
        }).reset_index()
        
        fig_bubble = px.scatter(
            dept_stats,
            x='tasa_mortalidad',
            y='total',
            size='n',
            color='tasa_mortalidad',
            hover_name='dpto',
            title='Análisis de Mortalidad y Población por Departamento',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_bubble)
    
    with tab4:
        st.header("Visualizaciones Detalladas")
        
        # Matriz de correlación
        columnas_numericas = ['ano', 'n', 'total', 'tasa_mortalidad']
        corr_matrix = df[columnas_numericas].corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            color_continuous_scale='RdBu',
            aspect='auto',
            title='Matriz de Correlación'
        )
        st.plotly_chart(fig_corr)
        
        # Evolución temporal
        fig_evolution = px.line(
            df.groupby('ano')['tasa_mortalidad'].mean().reset_index(),
            x='ano',
            y='tasa_mortalidad',
            title='Evolución Temporal de la Tasa de Mortalidad'
        )
        st.plotly_chart(fig_evolution)
            
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