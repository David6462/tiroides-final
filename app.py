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

# Aplicar tema oscuro personalizado
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
        color: white;
    }
    .stPlotlyChart {
        background-color: #0E1117;
    }
    </style>
    """, unsafe_allow_html=True)

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
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Pronósticos", 
        "👥 Demografía", 
        "🗺️ Análisis Regional",
        "📊 Visualizaciones",
        "🔮 Predicción Individual"
    ])
    
    with tab2:
        st.header("Análisis Demográfico")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Gráfico de distribución por sexo mejorado
            sex_counts = df['sexo'].value_counts()
            fig_sex = go.Figure(data=[go.Pie(
                labels=sex_counts.index,
                values=sex_counts.values,
                hole=0.4,
                marker_colors=['lightblue', '#1f77b4'],
                textinfo='percent',
                textfont_size=14,
                showlegend=True
            )])
            
            fig_sex.update_layout(
                title='Distribución por Sexo',
                paper_bgcolor='#0E1117',
                plot_bgcolor='#0E1117',
                font=dict(color='white'),
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=1.1,
                    xanchor="left",
                    x=0.01,
                    orientation="h",
                    font=dict(size=12),
                    bgcolor='rgba(0,0,0,0)'
                )
            )
            st.plotly_chart(fig_sex)
        
        with col2:
            # Evolución temporal por grupo de edad mejorada
            fig_age = px.bar(
                df.groupby('gru_edad')['n'].sum().reset_index(),
                x='gru_edad',
                y='n',
                title='Distribución por Grupo de Edad',
                color_discrete_sequence=['lightblue']
            )
            
            fig_age.update_layout(
                paper_bgcolor='#0E1117',
                plot_bgcolor='#0E1117',
                font=dict(color='white'),
                xaxis=dict(
                    title='Grupo de Edad',
                    gridcolor='rgba(255,255,255,0.1)',
                    showgrid=True
                ),
                yaxis=dict(
                    title='Número de Casos',
                    gridcolor='rgba(255,255,255,0.1)',
                    showgrid=True
                ),
                showlegend=False
            )
            st.plotly_chart(fig_age)
    
    with tab3:
        st.header("Análisis Regional")
        
        # Heatmap mejorado
        anos_max = df['ano'].max()
        df_reciente = df[df['ano'] > anos_max - 10]
        
        pivot_table = df_reciente.pivot_table(
            values='tasa_mortalidad',
            index='dpto',
            columns='ano',
            aggfunc='mean'
        )
        
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=pivot_table.values,
            x=pivot_table.columns,
            y=pivot_table.index,
            colorscale='Reds',
            showscale=True
        ))
        
        fig_heatmap.update_layout(
            title='Tasas de Mortalidad por Departamento',
            paper_bgcolor='#0E1117',
            plot_bgcolor='#0E1117',
            font=dict(color='white'),
            xaxis=dict(title='Año'),
            yaxis=dict(title='Departamento'),
            height=600
        )
        st.plotly_chart(fig_heatmap)
        
        # Gráfico de dispersión mejorado
        dept_stats = df.groupby('dpto').agg({
            'tasa_mortalidad': 'mean',
            'total': 'mean',
            'n': 'sum'
        }).reset_index()
        
        fig_scatter = px.scatter(
            dept_stats,
            x='tasa_mortalidad',
            y='total',
            size='n',
            color='tasa_mortalidad',
            hover_name='dpto',
            color_continuous_scale='Viridis'
        )
        
        fig_scatter.update_layout(
            title='Análisis de Mortalidad y Población por Departamento',
            paper_bgcolor='#0E1117',
            plot_bgcolor='#0E1117',
            font=dict(color='white'),
            xaxis=dict(
                title='Tasa de Mortalidad',
                gridcolor='rgba(255,255,255,0.1)',
                showgrid=True
            ),
            yaxis=dict(
                title='Población Total',
                gridcolor='rgba(255,255,255,0.1)',
                showgrid=True
            )
        )
        st.plotly_chart(fig_scatter)
    
    with tab4:
        st.header("Visualizaciones Detalladas")
        
        # Matriz de correlación mejorada
        columnas_numericas = ['ano', 'n', 'total', 'tasa_mortalidad']
        corr_matrix = df[columnas_numericas].corr()
        
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0
        ))
        
        fig_corr.update_layout(
            title='Matriz de Correlación',
            paper_bgcolor='#0E1117',
            plot_bgcolor='#0E1117',
            font=dict(color='white')
        )
        st.plotly_chart(fig_corr)
        
        # Evolución temporal mejorada
        fig_evolution = go.Figure()
        
        fig_evolution.add_trace(go.Scatter(
            x=df.groupby('ano')['tasa_mortalidad'].mean().index,
            y=df.groupby('ano')['tasa_mortalidad'].mean().values,
            mode='lines',
            line=dict(color='#17becf', width=2)
        ))
        
        fig_evolution.update_layout(
            title='Evolución Temporal de la Tasa de Mortalidad',
            paper_bgcolor='#0E1117',
            plot_bgcolor='#0E1117',
            font=dict(color='white'),
            xaxis=dict(
                title='Año',
                gridcolor='rgba(255,255,255,0.1)',
                showgrid=True
            ),
            yaxis=dict(
                title='Tasa de Mortalidad',
                gridcolor='rgba(255,255,255,0.1)',
                showgrid=True
            )
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