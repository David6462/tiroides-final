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

# Configuración de visualización
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
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Pronósticos", 
        "👥 Demografía", 
        "🗺️ Análisis Regional", 
        "📊 Visualizaciones Detalladas",
        "🔮 Predicción Individual",
        "📑 Análisis Adicional"
    ])
    
    with tab1:
        st.header("Análisis de Pronósticos")
        
        # Selector de años para pronóstico
        anos_pronostico = st.slider("Años a pronosticar", 1, 10, 5)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Análisis general con Prophet
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
                
                fig_forecast = plt.figure(figsize=(12, 6))
                plt.plot(df_prophet['ds'], df_prophet['y'], 'ko-', label='Datos históricos', alpha=0.6)
                plt.plot(forecast['ds'], forecast['yhat'], 'b-', label='Pronóstico', linewidth=2)
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
                plt.grid(True, alpha=0.3)
                st.pyplot(fig_forecast)
        
        with col2:
            # Componentes del modelo
            fig_components = model.plot_components(forecast)
            st.pyplot(fig_components)
        
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
        
        # Análisis regional con Prophet
        st.subheader("Análisis Regional con Prophet")
        top_dptos = df.groupby('dpto')['tasa_mortalidad'].mean().nlargest(5).index
        
        fig_regional = plt.figure(figsize=(15, 10))
        
        for dpto in top_dptos:
            df_dpto = df[df['dpto'] == dpto].copy()
            df_prophet = df_dpto.groupby('ano')['tasa_mortalidad'].mean().reset_index()
            
            if len(df_prophet) >= 2:
                df_prophet.columns = ['ds', 'y']
                df_prophet['ds'] = pd.to_datetime(df_prophet['ds'], format='%Y')
                df_prophet = df_prophet.dropna()
                
                try:
                    model = Prophet(
                        yearly_seasonality=True,
                        weekly_seasonality=False,
                        daily_seasonality=False
                    )
                    model.fit(df_prophet)
                    
                    future = model.make_future_dataframe(periods=5, freq='Y')
                    forecast = model.predict(future)
                    
                    plt.plot(df_prophet['ds'], df_prophet['y'], 'o-', label=f'{dpto} (histórico)', alpha=0.6)
                    plt.plot(forecast['ds'], forecast['yhat'], '--', label=f'{dpto} (pronóstico)')
                except Exception as e:
                    st.warning(f"Error procesando {dpto}: {str(e)}")
        
        plt.title('Pronóstico Regional de Mortalidad por Tiroides')
        plt.xlabel('Año')
        plt.ylabel('Tasa de Mortalidad (por 100,000 habitantes)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_regional)
    
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
        
        # Distribución de Mortalidad por Grupo de Edad y Sexo
        st.subheader("Distribución de Mortalidad por Grupo de Edad y Sexo")
        fig_edad_sexo = plt.figure(figsize=(12, 6))
        # Asegurarse de que los datos estén ordenados
        df_edad_sexo = df.groupby(['gru_edad', 'sexo'])['n'].sum().unstack()
        df_edad_sexo = df_edad_sexo.fillna(0)  # Manejar valores faltantes
        df_edad_sexo.plot(kind='bar', stacked=True)
        plt.title('Distribución de Mortalidad por Grupo de Edad y Sexo')
        plt.xlabel('Grupo de Edad')
        plt.ylabel('Número de Casos')
        plt.legend(title='Sexo')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig_edad_sexo)
        
        # Tasas de mortalidad por grupo de edad
        st.subheader("Tasas de Mortalidad por Grupo de Edad")
        col3, col4 = st.columns(2)
        
        with col3:
            fig_violin1 = plt.figure(figsize=(12, 6))
            sns.violinplot(data=df, x='gru_edad', y='tasa_mortalidad', color='lightblue')
            plt.title('Distribución de Tasas de Mortalidad por Grupo de Edad')
            plt.xticks(rotation=45)
            st.pyplot(fig_violin1)
        
        with col4:
            # Vista detallada (sin outliers)
            threshold = df['tasa_mortalidad'].quantile(0.95)
            df_filtered = df[df['tasa_mortalidad'] <= threshold].copy()
            fig_violin2 = plt.figure(figsize=(12, 6))
            sns.violinplot(data=df_filtered, x='gru_edad', y='tasa_mortalidad', color='lightblue')
            plt.title('Vista Detallada (sin valores extremos)')
            plt.xticks(rotation=45)
            st.pyplot(fig_violin2)
    
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
            
            # Segmentación de departamentos
            st.subheader("Segmentación de Departamentos")
            
            try:
                # Preparar datos para clustering
                dept_stats = df.groupby('dpto').agg({
                    'tasa_mortalidad': ['mean', 'std', 'max']
                }).reset_index()
                
                dept_stats.columns = ['dpto_'] + ['_'.join(col).strip() for col in dept_stats.columns[1:]]
                
                # Manejar valores faltantes
                dept_stats = dept_stats.fillna({
                    'tasa_mortalidad_mean': dept_stats['tasa_mortalidad_mean'].median(),
                    'tasa_mortalidad_std': 0,
                    'tasa_mortalidad_max': dept_stats['tasa_mortalidad_max'].median()
                })
                
                # Normalizar datos y aplicar clustering
                scaler = StandardScaler()
                X = scaler.fit_transform(dept_stats[['tasa_mortalidad_mean', 'tasa_mortalidad_std', 'tasa_mortalidad_max']])
                
                kmeans = KMeans(n_clusters=3, random_state=42)
                dept_stats['cluster'] = kmeans.fit_predict(X)
                
                # Crear gráfico
                fig_cluster = plt.figure(figsize=(15, 10))
                colors = ['#FF9999', '#66B2FF', '#99FF99']
                markers = ['o', 's', '^']
                
                for i in range(3):
                    cluster_data = dept_stats[dept_stats['cluster'] == i]
                    plt.scatter(cluster_data['tasa_mortalidad_mean'],
                              cluster_data['tasa_mortalidad_std'],
                              c=colors[i],
                              marker=markers[i],
                              s=200,
                              label=f'Cluster {i+1}',
                              alpha=0.6)
                    
                    # Añadir etiquetas
                    for _, row in cluster_data.iterrows():
                        plt.annotate(row['dpto_'],
                                   (row['tasa_mortalidad_mean'], row['tasa_mortalidad_std']),
                                   xytext=(5, 5),
                                   textcoords='offset points',
                                   fontsize=9,
                                   bbox=dict(facecolor='white', 
                                           edgecolor='gray',
                                           alpha=0.7,
                                           boxstyle='round,pad=0.3'))
                
                plt.title('Clusters de Departamentos por Mortalidad', size=14, pad=20)
                plt.xlabel('Tasa de Mortalidad Media', size=12)
                plt.ylabel('Desviación Estándar', size=12)
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.grid(True, alpha=0.3, linestyle='--')
                plt.tight_layout()
                st.pyplot(fig_cluster)
                
                # Mostrar resumen de clusters
                st.subheader("Resumen de Clusters")
                for i in range(3):
                    cluster_data = dept_stats[dept_stats['cluster'] == i]
                    st.write(f"**Cluster {i+1}:**")
                    st.write(f"- Número de departamentos: {len(cluster_data)}")
                    st.write(f"- Tasa media: {cluster_data['tasa_mortalidad_mean'].mean():.2f}")
                    st.write(f"- Departamentos: {', '.join(cluster_data['dpto_'].tolist())}")
                    
            except Exception as e:
                st.error(f"Error en la segmentación: {str(e)}")
    
    with tab4:
        st.header("Visualizaciones Detalladas")
        
        # Selector de tipo de visualización
        viz_type = st.selectbox(
            "Seleccionar tipo de visualización",
            ["Heatmap de Mortalidad", "Distribución por Edad y Sexo", "Análisis de Población", "Evolución Temporal", "Correlaciones"]
        )
        
        if viz_type == "Heatmap de Mortalidad":
            # Heatmap de últimos 10 años
            anos_max = df['ano'].max()
            df_reciente = df[df['ano'] > anos_max - 10]
            
            # Crear tabla pivote
            pivot_table = df_reciente.pivot_table(
                values='tasa_mortalidad',
                index='dpto',
                columns='ano',
                aggfunc='mean'
            )
            
            # Ordenar departamentos por promedio
            promedios = pivot_table.mean(axis=1)
            pivot_table = pivot_table.reindex(promedios.sort_values(ascending=False).index)
            
            fig = plt.figure(figsize=(15, 12))
            sns.heatmap(
                pivot_table,
                cmap='Reds',
                annot=True,
                fmt='.0f',
                cbar_kws={
                    'label': 'Tasa de Mortalidad por 100,000 habitantes',
                    'orientation': 'horizontal'
                }
            )
            plt.title('Tasas de Mortalidad por Departamento\nÚltimos 10 años', pad=20)
            plt.xlabel('Año')
            plt.ylabel('Departamento')
            plt.tight_layout()
            st.pyplot(fig)
            
        elif viz_type == "Distribución por Edad y Sexo":
            col1, col2 = st.columns(2)
            
            with col1:
                # Distribución por grupo de edad y sexo
                fig1 = plt.figure(figsize=(12, 6))
                df_edad_sexo = df.groupby(['gru_edad', 'sexo'])['n'].sum().unstack()
                df_edad_sexo = df_edad_sexo.fillna(0)  # Manejar valores faltantes
                df_edad_sexo.plot(kind='bar', stacked=True)
                plt.title('Distribución de Mortalidad por Grupo de Edad y Sexo')
                plt.xlabel('Grupo de Edad')
                plt.ylabel('Número de Casos')
                plt.legend(title='Sexo')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig1)
            
            with col2:
                # Vista detallada de tasas por grupo de edad
                fig2 = plt.figure(figsize=(12, 6))
                threshold = df['tasa_mortalidad'].quantile(0.95)
                df_filtered = df[df['tasa_mortalidad'] <= threshold].copy()
                sns.violinplot(data=df_filtered, x='gru_edad', y='tasa_mortalidad', color='lightblue')
                plt.title('Distribución de Tasas de Mortalidad por Grupo de Edad\n(sin valores extremos)')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig2)
                
        elif viz_type == "Análisis de Población":
            # Análisis de mortalidad y población por departamento
            dept_stats = df.groupby('dpto').agg({
                'tasa_mortalidad': 'mean',
                'total': 'mean',
                'n': 'sum'
            }).reset_index()
            
            # Configuración de la figura
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 16))
            fig.patch.set_facecolor('#f0f0f0')
            
            # Gráfico para población > 20,000
            mask_high = dept_stats['total'] >= 20000
            high_data = dept_stats[mask_high]
            
            scatter1 = ax1.scatter(high_data['tasa_mortalidad'],
                                 high_data['total'],
                                 s=high_data['n'] * 2,
                                 c=high_data['tasa_mortalidad'],
                                 cmap='RdYlBu_r',
                                 alpha=0.7,
                                 edgecolor='white',
                                 linewidth=1)
            
            # Etiquetas para población alta
            for _, row in high_data.iterrows():
                ax1.annotate(row['dpto'],
                           (row['tasa_mortalidad'], row['total']),
                           xytext=(10, 0),
                           textcoords='offset points',
                           bbox=dict(facecolor='white',
                                   edgecolor='gray',
                                   alpha=0.8,
                                   boxstyle='round,pad=0.3'),
                           horizontalalignment='left',
                           verticalalignment='center')
            
            # Gráfico para población < 20,000
            mask_low = dept_stats['total'] < 20000
            low_data = dept_stats[mask_low].sort_values('total', ascending=False)
            
            scatter2 = ax2.scatter(low_data['tasa_mortalidad'],
                                 low_data['total'],
                                 s=low_data['n'] * 3,
                                 c=low_data['tasa_mortalidad'],
                                 cmap='RdYlBu_r',
                                 alpha=0.7,
                                 edgecolor='white',
                                 linewidth=1)
            
            # Etiquetas para población baja
            for idx, row in low_data.iterrows():
                offset = 10 if idx % 2 == 0 else -10
                ha = 'left' if idx % 2 == 0 else 'right'
                ax2.annotate(row['dpto'],
                           (row['tasa_mortalidad'], row['total']),
                           xytext=(offset, 0),
                           textcoords='offset points',
                           bbox=dict(facecolor='white',
                                   edgecolor='gray',
                                   alpha=0.8,
                                   boxstyle='round,pad=0.3'),
                           horizontalalignment=ha,
                           verticalalignment='center')
            
            # Configuración común para ambos ejes
            for ax in [ax1, ax2]:
                ax.set_facecolor('white')
                ax.grid(True, alpha=0.3, linestyle='--', color='gray')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.tick_params(axis='both', which='major', labelsize=10)
            
            # Títulos y etiquetas
            ax1.set_title('Análisis de Mortalidad y Población por Departamento\n(Población > 20,000)',
                         pad=20, size=12, weight='bold')
            ax2.set_title('Detalle de Departamentos con Población < 20,000',
                         pad=20, size=12, weight='bold')
            
            for ax in [ax1, ax2]:
                ax.set_xlabel('Tasa de Mortalidad (por 100,000 habitantes)', size=10)
                ax.set_ylabel('Población Total', size=10)
            
            # Barra de color
            plt.colorbar(scatter1, ax=[ax1, ax2],
                        label='Tasa de Mortalidad por 100k hab.',
                        orientation='vertical')
            
            plt.tight_layout()
            st.pyplot(fig)
            
        elif viz_type == "Evolución Temporal":
            # Evolución de la mortalidad por grupo de edad
            pivot_trend = df.groupby(['ano', 'gru_edad'])['n'].sum().unstack()
            pivot_trend = pivot_trend.fillna(0)  # Manejar valores faltantes
            
            fig = plt.figure(figsize=(12, 6))
            pivot_trend.plot(kind='area', stacked=True)
            plt.title('Evolución de la Mortalidad por Grupo de Edad')
            plt.xlabel('Año')
            plt.ylabel('Número de Casos')
            plt.legend(title='Grupo de Edad', bbox_to_anchor=(1.05, 1))
            plt.tight_layout()
            st.pyplot(fig)
            
        else:  # Correlaciones
            # Matriz de correlación
            columnas_numericas = ['ano', 'n', 'total', 'tasa_mortalidad']
            df_num = df[columnas_numericas]
            corr_matrix = df_num.corr()
            
            fig = plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix,
                       annot=True,
                       cmap='coolwarm',
                       center=0,
                       fmt='.2f',
                       square=True)
            plt.title('Matriz de Correlación - Variables Numéricas')
            plt.tight_layout()
            st.pyplot(fig)
            
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