# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import anderson, shapiro
from statsmodels.stats.diagnostic import lilliefors
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import io
import openai
import os

# Configuración de la página
st.set_page_config(
    page_title="Analytics Statistics Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🤖 Analytics Statistics Assistant")
st.markdown("""
Esta aplicación te ayuda a realizar análisis estadísticos descriptivos e inferenciales para análisis de datos general.
Carga tus datos y consulta a OpenAI qué análisis realizar, luego ejecuta las funciones disponibles.
""")

# Sidebar para configuración
st.sidebar.header("🔧 Configuración")

# Configuración de OpenAI API
st.sidebar.subheader("Configuración de OpenAI")
openai_api_key = st.sidebar.text_input("Ingresa tu API Key de OpenAI:", type="password")
openai_client = None

if openai_api_key:
    try:
        openai_client = openai.OpenAI(api_key=openai_api_key)
        st.sidebar.success("✅ OpenAI configurado correctamente")
    except Exception as e:
        st.sidebar.error(f"Error configurando OpenAI: {e}")
else:
    st.sidebar.warning("⚠️ Ingresa tu API Key de OpenAI para usar las recomendaciones")

# Función para consultar OpenAI
def consultar_openai(prompt, max_tokens=2000, temperature=0.7, model="gpt-4"):
    """Consulta a OpenAI GPT para obtener recomendaciones y explicaciones"""
    try:
        if not openai_client:
            return "Error: Cliente OpenAI no configurado. Por favor, ingresa tu API Key en la barra lateral."
        
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Eres un experto en estadística aplicada y análisis de datos. Proporciona explicaciones claras, precisas y prácticas."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error al consultar OpenAI: {str(e)}"

# ASISTENTE TEÓRICO EN ESTADÍSTICA (se muestra siempre, sin necesidad de datos)
st.subheader("📚 Asistente Teórico en Estadística")
st.markdown("""
Consulta conceptos teóricos sobre métodos estadísticos, interpretación de resultados y mejores prácticas.
*No requiere que tengas datos cargados.*
""")

theory_question = st.text_area(
    "Haz tu pregunta sobre conceptos estadísticos:",
    placeholder="Ej: ¿Cuándo debo usar una prueba T en lugar de ANOVA? ¿Qué diferencia hay entre correlación y causalidad? ¿Cómo interpreto un p-valor? ¿Qué son las pruebas no paramétricas y cuándo usarlas?",
    height=120,
    key="theory_question_main"
)

if st.button("Consultar teoría estadística", key="theory_consultation_main") and theory_question:
    if openai_api_key:
        with st.spinner("El experto en estadística está analizando tu consulta..."):
            try:
                # Preparar contexto para asesoría teórica
                theory_context = f"""
                Eres un experto en estadística aplicada y metodología de investigación. 
                Responde la siguiente pregunta teórica sobre conceptos estadísticos:
                
                Pregunta del usuario: {theory_question}
                
                Por favor, proporciona una explicación clara y completa que incluya:
                
                1. **Definición conceptual**: Explica el concepto estadístico de manera accesible
                2. **Cuándo aplicarlo**: En qué situaciones o tipos de datos se utiliza
                3. **Supuestos requeridos**: Qué condiciones deben cumplirse
                4. **Interpretación**: Cómo interpretar los resultados correctamente
                5. **Limitaciones y consideraciones**: Precauciones y casos donde no aplica
                6. **Ejemplos prácticos**: Ejemplos ilustrativos del concepto
                7. **Relación con otros conceptos**: Cómo se relaciona con otros métodos estadísticos
                
                Si la pregunta involucra comparar métodos (ej: T-test vs ANOVA), incluye:
                - Diferencias clave entre los métodos
                - Ventajas y desventajas de cada uno
                - Criterios para elegir entre ellos
                - Ejemplos específicos de aplicación
                
                Mantén un tono pedagógico pero preciso, adecuado para profesionales que necesitan aplicar estos conceptos en análisis de datos.
                """
                
                theory_response = consultar_openai(theory_context)
                st.success("📚 Respuesta del Experto en Estadística:")
                
                # Mejorar la presentación de la respuesta
                st.markdown("---")
                st.markdown(theory_response)
                st.markdown("---")
                
            except Exception as e:
                st.error(f"Error en la consulta teórica: {e}")
    else:
        st.error("🔑 Necesitas configurar tu API Key de OpenAI en la barra lateral para usar el asistente teórico")

# Información sobre el asistente teórico
with st.expander("💡 ¿Qué puedo preguntar al asistente teórico?"):
    st.markdown("""
    **Ejemplos de preguntas que puedes hacer:**
    
    - **Conceptos básicos**: "¿Qué es un p-valor y cómo lo interpreto?"
    - **Comparación de métodos**: "¿Cuándo usar ANOVA en lugar de pruebas T?"
    - **Supuestos**: "¿Qué supuestos debe cumplir una regresión lineal?"
    - **Interpretación**: "¿Cómo interpreto un intervalo de confianza del 95%?"
    - **Selección de tests**: "¿Cuándo debo usar pruebas paramétricas vs no paramétricas?"
    - **Diseño de estudios**: "¿Qué consideraciones debo tener para un estudio A/B?"
    - **Errores comunes**: "¿Cuáles son los errores más comunes en la interpretación estadística?"
    - **Tamaño de muestra**: "¿Cómo determino el tamaño de muestra adecuado para mi estudio?"
    
    **Este asistente es puramente teórico y no analiza tus datos específicos.**
    """)

# Línea separadora
st.markdown("---")

# Carga de datos (esta parte permanece igual)
st.sidebar.subheader("Carga de Datos")
uploaded_file = st.sidebar.file_uploader("Sube tu archivo Excel o CSV", type=['xlsx', 'csv'])

@st.cache_data
def load_data(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)

# Funciones de muestreo del notebook
def generate_sample(df, sample_size, method="simple", stratify_col=None, random_state=None):
    """
    Genera un muestreo a partir de un DataFrame, permitiendo elegir entre:
    - Muestreo aleatorio simple
    - Muestreo estratificado
    """
    # Verificar si sample_size es un porcentaje o una cantidad fija
    if isinstance(sample_size, float):
        if sample_size <= 0 or sample_size > 1:
            raise ValueError("Si 'sample_size' es un porcentaje, debe estar entre 0 y 1.")
        sample_size = int(len(df) * sample_size)

    if sample_size <= 0 or sample_size > len(df):
        raise ValueError("El tamaño de la muestra debe ser mayor que 0 y menor o igual al total de datos.")

    # Muestreo Aleatorio Simple
    if method == "simple":
        sample_df = df.sample(n=sample_size, random_state=random_state)

    # Muestreo Estratificado
    elif method == "stratified":
        if stratify_col is None:
            raise ValueError("Se requiere un 'stratify_col' para realizar el muestreo estratificado.")
        if stratify_col not in df.columns:
            raise ValueError(f"La columna '{stratify_col}' no existe en el DataFrame.")

        sample_df = df.groupby(stratify_col, group_keys=False).apply(
            lambda x: x.sample(n=int(sample_size * len(x) / len(df)), random_state=random_state)
        )

    else:
        raise ValueError("El método de muestreo debe ser 'simple' o 'stratified'.")

    return sample_df.reset_index(drop=True)

def calculate_sample_size(population_size, margin_of_error=0.05, confidence_level=0.95, proportion=0.5):
    """
    Calcula el tamaño de muestra requerido en función del margen de error, nivel de confianza y proporción esperada.
    """
    # Validaciones
    if not (0 < margin_of_error < 1):
        raise ValueError("El margen de error debe estar entre 0 y 1.")
    if not (0 < confidence_level < 1):
        raise ValueError("El nivel de confianza debe estar entre 0 y 1.")
    if not (0 < proportion < 1):
        raise ValueError("La proporción debe estar entre 0 y 1.")
    if population_size <= 0:
        raise ValueError("El tamaño de la población debe ser mayor que 0.")

    # Obtener el valor Z correspondiente al nivel de confianza
    z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)

    # Calcular el tamaño de muestra sin ajuste finito
    sample_size = (z_score**2 * proportion * (1 - proportion)) / (margin_of_error**2)

    # Ajuste por población finita (si la población es pequeña)
    adjusted_sample_size = sample_size / (1 + (sample_size - 1) / population_size)

    # Redondear al entero superior
    final_sample_size = int(np.ceil(adjusted_sample_size))

    return final_sample_size

df = None
if uploaded_file is not None:
    try:
        df = load_data(uploaded_file)
        # Eliminar columna Unnamed: 0 si existe
        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)
        st.sidebar.success(f"✅ Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
    except Exception as e:
        st.sidebar.error(f"Error cargando archivo: {e}")

# Mostrar datos si están cargados
if df is not None:
    st.subheader("📋 Vista previa de los datos")
    st.dataframe(df.head())
    
    # Información básica del dataset
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total de registros", df.shape[0])
    with col2:
        st.metric("Total de variables", df.shape[1])
    with col3:
        st.metric("Valores faltantes", df.isnull().sum().sum())
    
    # Selector de variables
    st.subheader("🔍 Selección de variables")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    col1, col2 = st.columns(2)
    with col1:
        if numeric_cols:
            selected_numeric = st.multiselect("Variables numéricas:", numeric_cols, default=numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols)
        else:
            st.warning("No se encontraron variables numéricas")
    
    with col2:
        if categorical_cols:
            selected_categorical = st.multiselect("Variables categóricas:", categorical_cols, default=categorical_cols[0] if categorical_cols else None)
        else:
            st.warning("No se encontraron variables categóricas")

    # Sección de consulta a OpenAI PARA DATOS ESPECÍFICOS (esta va después de cargar datos)
    st.subheader("🤖 Asistente de Análisis para tus Datos")
    st.markdown("Consulta recomendaciones específicas basadas en los datos que has cargado.")
    
    user_question = st.text_area(
        "Describe tu caso de negocio o pregunta qué análisis realizar con tus datos:",
        placeholder="Ej: Quiero analizar si hay diferencias en la satisfacción laboral entre departamentos, y cómo se relaciona con el rendimiento...",
        height=100,
        key="business_question_main"
    )
    
    if st.button("Obtener recomendaciones de análisis", key="business_recommendations_main") and user_question:
        if openai_api_key:
            with st.spinner("OpenAI está analizando tu caso y datos..."):
                try:
                    # Preparar contexto para OpenAI
                    context = f"""
                    Tengo un dataset de análisis de datos con {df.shape[0]} filas y {df.shape[1]} columnas.
                    Variables numéricas: {numeric_cols}
                    Variables categóricas: {categorical_cols}
                    
                    Pregunta del usuario: {user_question}
                    
                    Recomienda qué análisis estadísticos específicos realizar de esta lista:
                    - Muestreo (tamaño de muestra, muestreo aleatorio, estratificado)
                    - Análisis descriptivo general
                    - Pruebas de normalidad
                    - Correlaciones entre variables
                    - Pruebas t (una muestra, muestras independientes, pareadas)
                    - ANOVA (una vía, dos vías)
                    - Pruebas no paramétricas (Mann-Whitney, Kruskal-Wallis, Wilcoxon)
                    - Pruebas de chi-cuadrado
                    - Análisis de homogeneidad de varianzas
                    
                    Para cada análisis recomendado, indica:
                    1. Qué variables usar
                    2. Qué pregunta de negocio responde
                    3. Interpretación esperada
                    """
                    
                    response = consultar_openai(context)
                    st.success("🎯 Recomendaciones de Análisis para tus Datos:")
                    st.markdown("---")
                    st.write(response)
                    st.markdown("---")
                    
                except Exception as e:
                    st.error(f"Error consultando a OpenAI: {e}")
        else:
            st.error("🔑 Necesitas configurar tu API Key de OpenAI en la barra lateral")

# Sección de análisis estadísticos
if df is not None:
    st.header("📊 Análisis Estadísticos")
    
    # Crear pestañas para diferentes tipos de análisis
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "🎯 Muestreo", 
        "📈 Descriptivos", 
        "🔍 Normalidad", 
        "📉 Correlaciones",
        "⚖️ Homogeniedad de Varianzas",
        "✅ Pruebas T",
        "📊 ANOVA",
        "🔄 No Paramétricas"
    ])
    
    with tab1:  # Muestreo
        st.subheader("🎯 Análisis de Muestreo")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Generar Muestra")
            st.markdown("Genera una muestra representativa de tus datos para análisis.")
            
            sample_method = st.radio(
                "Método de muestreo:",
                ["simple", "stratified"],
                format_func=lambda x: "Aleatorio Simple" if x == "simple" else "Estratificado"
            )
            
            sample_size_type = st.radio(
                "Tipo de tamaño de muestra:",
                ["percentage", "absolute"],
                format_func=lambda x: "Porcentaje" if x == "percentage" else "Número absoluto"
            )
            
            if sample_size_type == "percentage":
                sample_size_input = st.slider(
                    "Porcentaje de muestra:",
                    min_value=1,
                    max_value=50,
                    value=20,
                    help="Porcentaje del total de datos a incluir en la muestra"
                )
                sample_size = sample_size_input / 100.0
            else:
                sample_size_input = st.number_input(
                    "Tamaño de muestra:",
                    min_value=1,
                    max_value=len(df),
                    value=min(100, len(df)),
                    help="Número absoluto de registros para la muestra"
                )
                sample_size = sample_size_input
            
            if sample_method == "stratified" and categorical_cols:
                stratify_column = st.selectbox(
                    "Variable para estratificación:",
                    categorical_cols,
                    help="La muestra mantendrá las proporciones de esta variable categórica"
                )
            else:
                stratify_column = None
            
            if st.button("🎲 Generar Muestra", key="generate_sample"):
                try:
                    with st.spinner("Generando muestra..."):
                        sample_df = generate_sample(
                            df, 
                            sample_size, 
                            method=sample_method, 
                            stratify_col=stratify_column, 
                            random_state=42
                        )
                    
                    st.success(f"✅ Muestra generada: {len(sample_df)} registros")
                    
                    # Mostrar información de la muestra
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Tamaño de muestra", len(sample_df))
                    with col2:
                        st.metric("Porcentaje del total", f"{(len(sample_df)/len(df))*100:.1f}%")
                    
                    # Mostrar preview de la muestra
                    st.subheader("Vista previa de la muestra")
                    st.dataframe(sample_df.head())
                    
                    # Mostrar distribución si es muestreo estratificado
                    if sample_method == "stratified" and stratify_column:
                        st.subheader("📋 Distribución en la muestra")
                        sample_dist = sample_df[stratify_column].value_counts()
                        original_dist = df[stratify_column].value_counts()
                        
                        dist_comparison = pd.DataFrame({
                            'Original': original_dist,
                            'Muestra': sample_dist,
                            '% Original': (original_dist / len(df)) * 100,
                            '% Muestra': (sample_dist / len(sample_df)) * 100
                        })
                        
                        st.dataframe(dist_comparison)
                        
                        # Gráfico de comparación
                        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                        
                        # Gráfico original
                        original_dist.plot(kind='bar', ax=ax[0], color='skyblue', alpha=0.7)
                        ax[0].set_title('Distribución Original')
                        ax[0].set_ylabel('Frecuencia')
                        ax[0].tick_params(axis='x', rotation=45)
                        
                        # Gráfico muestra
                        sample_dist.plot(kind='bar', ax=ax[1], color='lightcoral', alpha=0.7)
                        ax[1].set_title('Distribución en Muestra')
                        ax[1].set_ylabel('Frecuencia')
                        ax[1].tick_params(axis='x', rotation=45)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    # Opción para descargar la muestra en Excel
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        sample_df.to_excel(writer, index=False, sheet_name='Muestra')
                    
                    st.download_button(
                        label="📥 Descargar muestra como Excel",
                        data=output.getvalue(),
                        file_name="muestra_generada.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    
                except Exception as e:
                    st.error(f"❌ Error generando muestra: {e}")
        
        with col2:
            st.subheader("🧮 Calcular Tamaño de Muestra")
            st.markdown("Calcula el tamaño mínimo de muestra necesario para tu estudio.")
            
            population_size = st.number_input(
                "Tamaño de la población:",
                min_value=1,
                value=len(df),
                help="Número total de elementos en la población de estudio"
            )
            
            margin_error = st.slider(
                "Margen de error (%):",
                min_value=1,
                max_value=10,
                value=5,
                help="Precisión deseada en los resultados (±%)"
            ) / 100.0
            
            confidence_level = st.slider(
                "Nivel de confianza (%):",
                min_value=80,
                max_value=99,
                value=95,
                help="Probabilidad de que el resultado sea correcto"
            ) / 100.0
            
            proportion = st.slider(
                "Proporción esperada (%):",
                min_value=1,
                max_value=99,
                value=50,
                help="Proporción esperada de la característica en la población (usar 50% si es desconocida)"
            ) / 100.0
            
            if st.button("📐 Calcular Tamaño de Muestra", key="calculate_sample_size"):
                try:
                    with st.spinner("Calculando tamaño de muestra..."):
                        sample_size = calculate_sample_size(
                            population_size=population_size,
                            margin_of_error=margin_error,
                            confidence_level=confidence_level,
                            proportion=proportion
                        )
                    
                    st.success(f"🎯 Tamaño de muestra recomendado: **{sample_size}**")
                    
                    # Información adicional
                    st.info(f"""
                    **Parámetros utilizados:**
                    - Población: {population_size:,}
                    - Margen de error: ±{margin_error*100:.1f}%
                    - Nivel de confianza: {confidence_level*100:.1f}%
                    - Proporción esperada: {proportion*100:.1f}%
                    """)
                    
                    # Comparación con datos actuales
                    if population_size == len(df):
                        coverage = (sample_size / len(df)) * 100
                        st.metric(
                            "Cobertura de tu dataset",
                            f"{coverage:.1f}%",
                            delta=f"{sample_size - len(df)} registros" if sample_size > len(df) else None,
                            delta_color="inverse" if sample_size > len(df) else "normal"
                        )
                        
                        if sample_size > len(df):
                            st.warning("⚠️ Tu dataset actual es más pequeño que el tamaño de muestra recomendado")
                        else:
                            st.success("✅ Tu dataset actual es suficiente para el análisis")
                    
                except Exception as e:
                    st.error(f"❌ Error calculando tamaño de muestra: {e}")
            
            # Información educativa
            with st.expander("💡 ¿Por qué es importante el muestreo?"):
                st.markdown("""
                **El muestreo adecuado es crucial porque:**
                - Reduce costos y tiempo de análisis
                - Permite trabajar con conjuntos de datos manejables
                - Mantiene la representatividad de la población
                - Facilita la generalización de resultados
                
                **Tipos de muestreo:**
                - **Aleatorio simple:** Cada elemento tiene igual probabilidad de ser seleccionado
                - **Estratificado:** Mantiene las proporciones de subgrupos importantes
                """)
    
    with tab2:  # Análisis descriptivos
        st.subheader("Análisis Descriptivo")
        
        # Estadísticas descriptivas básicas
        if numeric_cols:
            st.subheader("Estadísticas Descriptivas por Variable Numérica")
            selected_var = st.selectbox("Selecciona variable numérica:", numeric_cols, key="desc_var")
            if selected_var:
                desc_stats = df[selected_var].describe()
                st.write(desc_stats)
                
                # Histograma y boxplot
                col1, col2 = st.columns(2)
                with col1:
                    fig, ax = plt.subplots()
                    sns.histplot(df[selected_var], kde=True, ax=ax)
                    ax.set_title(f'Distribución de {selected_var}')
                    st.pyplot(fig)
                
                with col2:
                    fig, ax = plt.subplots()
                    sns.boxplot(y=df[selected_var], ax=ax)
                    ax.set_title(f'Boxplot de {selected_var}')
                    st.pyplot(fig)
        
        # Reporte descriptivo simple en lugar del profiling
        st.subheader("Reporte Descriptivo Resumido")
        st.markdown("Genera un reporte resumido de análisis exploratorio de datos.")
        
        if st.button("📊 Generar Reporte Descriptivo", key="desc_report_button"):
            with st.spinner("Generando reporte..."):
                try:
                    # Análisis descriptivo básico
                    st.subheader("📋 Estadísticas Descriptivas Generales")
                    
                    if numeric_cols:
                        st.write("**Variables Numéricas:**")
                        numeric_desc = df[numeric_cols].describe().T
                        numeric_desc['CV'] = (numeric_desc['std'] / numeric_desc['mean']) * 100
                        numeric_desc['missing'] = df[numeric_cols].isnull().sum()
                        st.dataframe(numeric_desc)
                    
                    if categorical_cols:
                        st.write("**Variables Categóricas:**")
                        for cat_var in categorical_cols:
                            st.write(f"**{cat_var}:**")
                            cat_stats = df[cat_var].value_counts().reset_index()
                            cat_stats.columns = ['Valor', 'Frecuencia']
                            cat_stats['Porcentaje'] = (cat_stats['Frecuencia'] / len(df)) * 100
                            st.dataframe(cat_stats)
                    
                    st.success("✅ Reporte descriptivo generado correctamente.")
                    
                except Exception as e:
                    st.error(f"Error generando reporte: {e}")
    
    with tab3:  # Pruebas de normalidad
        st.subheader("🔍 Pruebas de Normalidad")
        st.markdown("Evalúa si tus datos siguen una distribución normal, requisito para muchas pruebas paramétricas.")
        
        if numeric_cols:
            selected_normal_var = st.selectbox("Selecciona variable para prueba de normalidad:", numeric_cols, key="normal_var")
            alpha_normal = st.slider("Nivel de significancia (α):", 0.01, 0.10, 0.05, key="normal_alpha")
            
            if st.button("📊 Ejecutar Pruebas de Normalidad"):
                try:
                    data = df[selected_normal_var].dropna()
                    n = len(data)
                    
                    if n < 3:
                        st.error("Se necesitan al menos 3 observaciones para las pruebas de normalidad")
                    else:
                        st.subheader("📋 Resultados de las Pruebas de Normalidad")
                        st.info(f"**Tamaño de muestra:** {n:,} observaciones")
                        
                        # ==========================================
                        # 1. SHAPIRO-WILK
                        # ==========================================
                        st.markdown("#### 1. Prueba de Shapiro-Wilk")
                        st.caption("Prueba más potente para detectar desviaciones de la normalidad")
                        
                        if n > 5000:
                            st.warning("""
                            ⚠️ **Limitación de Shapiro-Wilk con muestras grandes**
                            
                            Con n > 5000, esta prueba se vuelve extremadamente sensible y puede rechazar 
                            normalidad por desviaciones triviales. Los resultados deben interpretarse con 
                            precaución y complementarse con análisis visual.
                            """)
                            shapiro_stat, shapiro_p = shapiro(data)
                            shapiro_normal = shapiro_p > alpha_normal
                            shapiro_weight = 1  # Peso reducido para muestras grandes
                        else:
                            shapiro_stat, shapiro_p = shapiro(data)
                            shapiro_normal = shapiro_p > alpha_normal
                            shapiro_weight = 3  # Peso alto (más confiable)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Estadístico W", f"{shapiro_stat:.4f}")
                            st.caption("Rango: [0, 1]. Valores cercanos a 1 indican normalidad")
                        with col2:
                            st.metric("p-valor", f"{shapiro_p:.4f}")
                        
                        # Explicación adicional para Shapiro-Wilk
                        with st.expander("ℹ️ Interpretación de Shapiro-Wilk"):
                            st.markdown(f"""
                            **Cómo funciona Shapiro-Wilk:**
                            
                            Esta prueba compara los datos observados con lo que se esperaría de una distribución normal perfecta.
                            
                            **Estadístico W:**
                            - Valor calculado: **{shapiro_stat:.4f}**
                            - Rango posible: 0 a 1
                            - W = 1 → distribución perfectamente normal
                            - W < 1 → desviación de la normalidad
                            - En la práctica: W > 0.95 es considerado bueno
                            
                            **P-valor:**
                            - Valor calculado: **{shapiro_p:.4f}**
                            - Tu nivel α: **{alpha_normal}**
                            - Si p > α → NO rechazamos normalidad (datos parecen normales)
                            - Si p ≤ α → RECHAZAMOS normalidad (datos NO parecen normales)
                            
                            **En este caso:**
                            - p-valor ({shapiro_p:.4f}) {">" if shapiro_normal else "≤"} α ({alpha_normal})
                            - **Conclusión:** {"Los datos SON consistentes con normalidad" if shapiro_normal else "Los datos NO son consistentes con normalidad"}
                            
                            **Ventajas de Shapiro-Wilk:**
                            - ✅ Más potente que otras pruebas para n < 2000
                            - ✅ Detecta bien desviaciones en las colas
                            - ✅ Fácil de interpretar con p-valor exacto
                            
                            **Limitaciones:**
                            - ⚠️ Muy sensible con muestras grandes (n > 5000)
                            - ⚠️ Puede rechazar normalidad por diferencias triviales
                            {"- ⚠️ **TU MUESTRA ES GRANDE (n=" + str(n) + ")** - complementa con gráficos" if n > 5000 else ""}
                            """)
                        
                        if shapiro_normal:
                            st.success("✅ Los datos parecen normales según Shapiro-Wilk")
                        else:
                            st.error("❌ Los datos NO parecen normales según Shapiro-Wilk")
                        
                        # ==========================================
                        # 2. ANDERSON-DARLING
                        # ==========================================
                        st.markdown("#### 2. Prueba de Anderson-Darling")
                        st.caption("Da más peso a las colas de la distribución")
                        
                        ad_test = anderson(data, dist='norm')
                        ad_statistic = ad_test.statistic
                        
                        # Mapeo de alpha a índice de valores críticos
                        # ad_test.significance_level = [15.0, 10.0, 5.0, 2.5, 1.0]
                        # ad_test.critical_values tiene los valores críticos correspondientes
                        alpha_to_idx = {
                            0.15: 0,
                            0.10: 1,
                            0.05: 2,
                            0.025: 3,
                            0.01: 4
                        }
                        
                        # Encontrar el índice más cercano al alpha seleccionado
                        closest_alpha = min(alpha_to_idx.keys(), key=lambda x: abs(x - alpha_normal))
                        idx = alpha_to_idx[closest_alpha]
                        critical_value = ad_test.critical_values[idx]
                        
                        # ✅ CORRECCIÓN: Calcular rango de p-valor correctamente
                        # Si estadístico < valor crítico → NO se rechaza normalidad → p > nivel
                        # Si estadístico >= valor crítico → SE rechaza normalidad → p < nivel
                        if ad_statistic < ad_test.critical_values[0]:
                            # Estadístico muy pequeño → fuerte evidencia de normalidad
                            p_value_range = f"> {ad_test.significance_level[0]/100:.2f}"
                            p_value_numeric = 0.20  # Para comparaciones numéricas
                            p_value_interpretation = f"El p-valor es mayor a {ad_test.significance_level[0]/100:.2f}"
                        elif ad_statistic >= ad_test.critical_values[-1]:
                            # Estadístico muy grande → fuerte evidencia contra normalidad
                            p_value_range = f"< {ad_test.significance_level[-1]/100:.2f}"
                            p_value_numeric = 0.005  # Para comparaciones numéricas
                            p_value_interpretation = f"El p-valor es menor a {ad_test.significance_level[-1]/100:.2f}"
                        else:
                            # Estadístico está entre dos valores críticos
                            for i in range(len(ad_test.critical_values) - 1):
                                if ad_test.critical_values[i] <= ad_statistic < ad_test.critical_values[i+1]:
                                    lower_sig = ad_test.significance_level[i+1] / 100
                                    upper_sig = ad_test.significance_level[i] / 100
                                    p_value_range = f"{lower_sig:.3f} < p < {upper_sig:.3f}"
                                    p_value_numeric = (lower_sig + upper_sig) / 2
                                    p_value_interpretation = f"El p-valor está entre {lower_sig:.3f} y {upper_sig:.3f}"
                                    break
                        
                        # ✅ CORRECCIÓN: Decisión basada en valores críticos, no en p-valor aproximado
                        ad_normal = ad_statistic < critical_value
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Estadístico A-D", f"{ad_statistic:.4f}")
                            st.caption(f"Valor crítico (α={closest_alpha:.3f}): {critical_value:.3f}")
                        with col2:
                            st.metric("p-valor (rango aproximado)", p_value_range)
                            st.caption("⚠️ A-D proporciona rangos, no p-valores exactos")
                        
                        # Explicación adicional para Anderson-Darling
                        with st.expander("ℹ️ Interpretación de Anderson-Darling"):
                            st.markdown(f"""
                            **Cómo funciona Anderson-Darling:**
                            
                            Esta prueba mide qué tan bien los datos se ajustan a una distribución normal, 
                            dando **más peso a las colas** (valores extremos) que otras pruebas.
                            
                            **Estadístico A-D:**
                            - Valor calculado: **{ad_statistic:.4f}**
                            - Mide la discrepancia entre los datos y la distribución normal
                            - Valores pequeños → buena concordancia con normalidad
                            - Valores grandes → pobre concordancia con normalidad
                            
                            **Valores Críticos (en lugar de p-valor único):**
                            
                            Anderson-Darling NO proporciona un p-valor exacto como otras pruebas. 
                            En su lugar, compara el estadístico con valores críticos precalculados:
                            
                            | Nivel α | Valor Crítico |
                            |---------|---------------|
                            | 15%     | {ad_test.critical_values[0]:.3f} |
                            | 10%     | {ad_test.critical_values[1]:.3f} |
                            | 5%      | {ad_test.critical_values[2]:.3f} |
                            | 2.5%    | {ad_test.critical_values[3]:.3f} |
                            | 1%      | {ad_test.critical_values[4]:.3f} |
                            
                            **Regla de decisión:**
                            - Si Estadístico **<** Valor Crítico → NO rechazamos normalidad
                            - Si Estadístico **≥** Valor Crítico → RECHAZAMOS normalidad
                            
                            **En este caso (α = {closest_alpha}):**
                            - Estadístico A-D: **{ad_statistic:.4f}**
                            - Valor crítico: **{critical_value:.3f}**
                            - {ad_statistic:.4f} {"<" if ad_normal else "≥"} {critical_value:.3f}
                            - {p_value_interpretation}
                            - **Conclusión:** {"Los datos SON consistentes con normalidad" if ad_normal else "Los datos NO son consistentes con normalidad"}
                            
                            **Ventajas de Anderson-Darling:**
                            - ✅ Más sensible en las colas (detecta outliers)
                            - ✅ Funciona bien con muestras pequeñas y grandes
                            - ✅ No tiene límite superior de tamaño de muestra
                            
                            **Limitaciones:**
                            - ⚠️ No proporciona p-valor exacto (solo rangos)
                            - ⚠️ Interpretación menos intuitiva que Shapiro-Wilk
                            - ⚠️ Requiere entender valores críticos
                            """)
                        
                        if ad_normal:
                            st.success("✅ Los datos parecen normales según Anderson-Darling")
                        else:
                            st.error("❌ Los datos NO parecen normales según Anderson-Darling")
                        
                        # ==========================================
                        # 3. LILLIEFORS
                        # ==========================================
                        st.markdown("#### 3. Prueba de Kolmogorov-Smirnov (Lilliefors)")
                        st.caption("Versión mejorada de K-S que estima parámetros de los datos")
                        
                        lilliefors_stat, lilliefors_p = lilliefors(data)
                        lilliefors_normal = lilliefors_p > alpha_normal
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Estadístico D", f"{lilliefors_stat:.4f}")
                            st.caption("Mide la máxima distancia entre distribuciones")
                        with col2:
                            st.metric("p-valor", f"{lilliefors_p:.4f}")
                        
                        # Explicación adicional para Lilliefors
                        with st.expander("ℹ️ Interpretación de Kolmogorov-Smirnov (Lilliefors)"):
                            st.markdown(f"""
                            **Cómo funciona Lilliefors (K-S modificado):**
                            
                            Esta prueba compara la distribución acumulada empírica de tus datos con 
                            la distribución normal acumulada teórica. Es una **versión mejorada** de 
                            la clásica prueba Kolmogorov-Smirnov.
                            
                            **Estadístico D (Distancia):**
                            - Valor calculado: **{lilliefors_stat:.4f}**
                            - Mide la **máxima diferencia vertical** entre:
                            - La distribución acumulada de tus datos (escalera)
                            - La curva normal acumulada teórica (S suave)
                            - D = 0 → Ajuste perfecto (imposible en práctica)
                            - D pequeño (< 0.05) → Buen ajuste
                            - D grande (> 0.15) → Mal ajuste
                            
                            **Interpretación del estadístico D:**
                            """)
                            
                            # Interpretación visual del estadístico D
                            if lilliefors_stat < 0.05:
                                st.success(f"📊 D = {lilliefors_stat:.4f} < 0.05 → **Excelente ajuste** a la normalidad")
                            elif lilliefors_stat < 0.10:
                                st.info(f"📊 D = {lilliefors_stat:.4f} ∈ [0.05, 0.10) → **Buen ajuste** a la normalidad")
                            elif lilliefors_stat < 0.15:
                                st.warning(f"📊 D = {lilliefors_stat:.4f} ∈ [0.10, 0.15) → **Ajuste moderado** a la normalidad")
                            else:
                                st.error(f"📊 D = {lilliefors_stat:.4f} ≥ 0.15 → **Mal ajuste** a la normalidad")
                            
                            st.markdown(f"""
                            **P-valor:**
                            - Valor calculado: **{lilliefors_p:.4f}**
                            - Tu nivel α: **{alpha_normal}**
                            - Representa la probabilidad de obtener un D tan grande (o mayor) si los datos fueran realmente normales
                            
                            **Regla de decisión:**
                            - Si p > α → NO rechazamos normalidad (diferencia podría ser por azar)
                            - Si p ≤ α → RECHAZAMOS normalidad (diferencia es significativa)
                            
                            **En este caso:**
                            - p-valor ({lilliefors_p:.4f}) {">" if lilliefors_normal else "≤"} α ({alpha_normal})
                            - **Conclusión:** {"Los datos SON consistentes con normalidad" if lilliefors_normal else "Los datos NO son consistentes con normalidad"}
                            
                            **Diferencia con K-S clásico:**
                            
                            | Aspecto | K-S Clásico | Lilliefors |
                            |---------|-------------|------------|
                            | Parámetros | Deben ser conocidos | Se estiman de los datos |
                            | Uso típico | Distribuciones específicas | Normalidad con parámetros desconocidos |
                            | Conservadurismo | Más liberal | Más conservador (correcto) |
                            
                            **Ventajas de Lilliefors:**
                            - ✅ No requiere conocer μ y σ de antemano
                            - ✅ Más apropiado que K-S clásico para normalidad
                            - ✅ P-valor exacto y fácil de interpretar
                            - ✅ Funciona bien con muestras pequeñas
                            
                            **Limitaciones:**
                            - ⚠️ Menos potente que Shapiro-Wilk
                            - ⚠️ Sensible a desviaciones en el centro más que en las colas
                            - ⚠️ Con n muy pequeño (< 20) puede tener baja potencia
                            
                            **¿Cuándo usar Lilliefors?**
                            - ✓ Cuando tienes muestras pequeñas a medianas (20-500)
                            - ✓ Como complemento a Shapiro-Wilk
                            - ✓ Cuando quieres una prueba más conservadora
                            - ✓ Para reportar en artículos (ampliamente reconocida)
                            """)
                        
                        if lilliefors_normal:
                            st.success("✅ Los datos parecen normales según Lilliefors")
                        else:
                            st.error("❌ Los datos NO parecen normales según Lilliefors")
                        
                        # ==========================================
                        # CONCLUSIÓN FINAL
                        # ==========================================
                        st.markdown("---")
                        st.subheader("🎯 CONCLUSIÓN INTEGRADA")
                        
                        # Crear tabla resumen con pesos
                        results_data = [
                            {
                                'Prueba': 'Shapiro-Wilk',
                                'Estadístico': f"{shapiro_stat:.4f}",
                                'p-valor': f"{shapiro_p:.4f}" if n <= 5000 else f"{shapiro_p:.4f} ⚠️",
                                'Resultado': '✅ Normal' if shapiro_normal else '❌ No Normal',
                                'Peso': shapiro_weight
                            },
                            {
                                'Prueba': 'Anderson-Darling',
                                'Estadístico': f"{ad_statistic:.4f}",
                                'p-valor': p_value_range,
                                'Resultado': '✅ Normal' if ad_normal else '❌ No Normal',
                                'Peso': 2
                            },
                            {
                                'Prueba': 'Lilliefors',
                                'Estadístico': f"{lilliefors_stat:.4f}",
                                'p-valor': f"{lilliefors_p:.4f}",
                                'Resultado': '✅ Normal' if lilliefors_normal else '❌ No Normal',
                                'Peso': 2
                            }
                        ]
                        
                        results_df = pd.DataFrame(results_data)
                        st.dataframe(results_df, hide_index=True, use_container_width=True)
                        
                        # Calcular consenso ponderado
                        total_weight = shapiro_weight + 2 + 2
                        passed_weight = (
                            (shapiro_weight if shapiro_normal else 0) +
                            (2 if ad_normal else 0) +
                            (2 if lilliefors_normal else 0)
                        )
                        consensus = passed_weight / total_weight
                        
                        st.metric("Consenso Ponderado", f"{consensus*100:.1f}%", 
                                help="Porcentaje de evidencia (ponderado por confiabilidad) que apoya la normalidad")
                        
                        # Decisión final con matices
                        if consensus >= 0.7:
                            st.success(f"""
                            ✅ **CONCLUSIÓN: LOS DATOS PARECEN SEGUIR UNA DISTRIBUCIÓN NORMAL**
                            
                            **Consenso:** {consensus*100:.0f}% de las pruebas ponderadas
                            
                            **✓ Puedes usar pruebas paramétricas:**
                            - Prueba T de Student
                            - ANOVA
                            - Correlación de Pearson
                            - Regresión lineal
                            """)
                        elif consensus >= 0.4:
                            st.warning(f"""
                            ⚠️ **CONCLUSIÓN: EVIDENCIA MIXTA SOBRE NORMALIDAD**
                            
                            **Consenso:** {consensus*100:.0f}% de las pruebas ponderadas
                            
                            **Recomendaciones:**
                            1. 📊 Revisa cuidadosamente los gráficos Q-Q y el histograma
                            2. 🔄 Considera transformaciones de datos:
                            - Logarítmica: para datos con sesgo positivo
                            - Raíz cuadrada: para datos de conteo
                            - Box-Cox: transformación óptima automática
                            3. 📏 Si n > 30, las pruebas paramétricas son robustas (Teorema Central del Límite)
                            4. 🛡️ Como alternativa segura, usa pruebas no paramétricas
                            """)
                        else:
                            st.error(f"""
                            ❌ **CONCLUSIÓN: LOS DATOS NO PARECEN SEGUIR UNA DISTRIBUCIÓN NORMAL**
                            
                            **Consenso:** {consensus*100:.0f}% de las pruebas ponderadas
                            
                            **Opciones recomendadas:**
                            
                            **1. Transformaciones de datos:**
                            - `log(x)` - para datos con sesgo positivo
                            - `sqrt(x)` - para datos de conteo
                            - `1/x` - para tiempos o tasas
                            - Box-Cox o Yeo-Johnson - transformación óptima
                            
                            **2. Usar pruebas no paramétricas:**
                            - Mann-Whitney U (en lugar de prueba T independiente)
                            - Wilcoxon (en lugar de prueba T pareada)
                            - Kruskal-Wallis (en lugar de ANOVA)
                            - Spearman (en lugar de Pearson)
                            
                            **3. Modelos robustos:** Técnicas que no asumen normalidad
                            """)
                        
                        # Consideraciones sobre tamaño de muestra
                        if n < 30:
                            sample_size_msg = f"**n = {n} (< 30):** La normalidad es CRÍTICA. Considera pruebas no paramétricas si hay dudas."
                            sample_size_color = "🔴"
                        elif n < 100:
                            sample_size_msg = f"**n = {n} (30-100):** La normalidad es importante, pero las pruebas paramétricas tienen cierta robustez."
                            sample_size_color = "🟡"
                        elif n < 1000:
                            sample_size_msg = f"**n = {n} (100-1000):** Con este tamaño, las pruebas paramétricas son bastante robustas a desviaciones leves de normalidad."
                            sample_size_color = "🟢"
                        else:
                            sample_size_msg = f"**n = {n} (> 1000):** Las pruebas de normalidad pueden ser hipersensibles. Prioriza la validación visual y el sentido del negocio."
                            sample_size_color = "🔵"
                        
                        st.info(f"""
                        {sample_size_color} **Consideración sobre tamaño de muestra:**
                        
                        {sample_size_msg}
                        
                        **Regla general (Teorema Central del Límite):**
                        - Con muestras grandes (n ≥ 30), la distribución de medias tiende a ser normal
                        - Esto hace que las pruebas paramétricas sean robustas incluso con datos no normales
                        - EXCEPCIÓN: Datos con outliers extremos o distribuciones muy asimétricas
                        """)
                        
                        # ==========================================
                        # VISUALIZACIONES
                        # ==========================================
                        st.markdown("---")
                        st.subheader("📊 Diagnóstico Visual")
                        st.caption("Las visualizaciones son tan importantes como las pruebas estadísticas")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig, ax = plt.subplots(figsize=(8, 6))
                            
                            # Histograma con KDE
                            sns.histplot(data, kde=True, ax=ax, stat='density', alpha=0.7, color='skyblue')
                            
                            # Superponer curva normal teórica
                            mu, sigma = data.mean(), data.std()
                            x = np.linspace(data.min(), data.max(), 100)
                            ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, 
                                label=f'Normal teórica\n(μ={mu:.2f}, σ={sigma:.2f})')
                            
                            ax.set_title(f'Distribución de {selected_normal_var}', fontsize=12, fontweight='bold')
                            ax.set_xlabel(selected_normal_var, fontsize=10)
                            ax.set_ylabel('Densidad', fontsize=10)
                            ax.legend(loc='best')
                            ax.grid(True, alpha=0.3)
                            
                            st.pyplot(fig)
                            plt.close()
                            
                            st.caption("📌 Los datos deberían seguir aproximadamente la curva roja si son normales")
                        
                        with col2:
                            fig, ax = plt.subplots(figsize=(8, 6))
                            
                            # Q-Q plot
                            stats.probplot(data, dist="norm", plot=ax)
                            ax.set_title(f'Q-Q Plot de {selected_normal_var}', fontsize=12, fontweight='bold')
                            ax.grid(True, alpha=0.3)
                            
                            # Añadir caja de interpretación
                            textstr = 'Interpretación:\n• Puntos en línea roja\n  → Normal\n• Curva en extremos\n  → Colas pesadas\n• S invertida\n  → Asimetría'
                            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
                            ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
                                fontsize=9, verticalalignment='top', bbox=props)
                            
                            st.pyplot(fig)
                            plt.close()
                            
                            st.caption("📌 Los puntos deberían caer sobre la línea roja si los datos son normales")
                        
                        # Boxplot adicional para detectar outliers
                        st.markdown("#### Detección de Valores Atípicos")
                        fig, ax = plt.subplots(figsize=(10, 4))
                        sns.boxplot(x=data, ax=ax, color='lightcoral')
                        ax.set_xlabel(selected_normal_var, fontsize=10)
                        ax.set_title(f'Boxplot de {selected_normal_var} (Detección de Outliers)', 
                                    fontsize=12, fontweight='bold')
                        ax.grid(True, alpha=0.3, axis='x')
                        
                        st.pyplot(fig)
                        plt.close()
                        
                        # Calcular outliers
                        Q1 = data.quantile(0.25)
                        Q3 = data.quantile(0.75)
                        IQR = Q3 - Q1
                        outliers = data[(data < Q1 - 1.5*IQR) | (data > Q3 + 1.5*IQR)]
                        
                        if len(outliers) > 0:
                            st.warning(f"⚠️ Se detectaron {len(outliers)} valores atípicos ({len(outliers)/len(data)*100:.1f}% de los datos)")
                            st.caption("Los outliers pueden afectar las pruebas de normalidad. Considera investigar estos valores.")
                        else:
                            st.success("✅ No se detectaron valores atípicos significativos")
                        
                except Exception as e:
                    st.error(f"Error en pruebas de normalidad: {e}")
                    import traceback
                    st.code(traceback.format_exc())
        else:
            st.warning("No hay variables numéricas para analizar")

    with tab4:  # Correlaciones
        st.subheader("📉 Análisis de Correlación")
        st.markdown("Analiza la relación entre dos variables numéricas.")
        
        if len(numeric_cols) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                var1 = st.selectbox("Variable 1:", numeric_cols, key="corr_var1")
            with col2:
                var2 = st.selectbox("Variable 2:", numeric_cols, key="corr_var2")
            
            alpha_corr = st.slider("Nivel de significancia (α):", 0.01, 0.10, 0.05, key="corr_alpha")
            
            if st.button("🔍 Analizar Correlación"):
                try:
                    # Filtrar valores nulos
                    clean_data = df[[var1, var2]].dropna()
                    
                    if len(clean_data) < 3:
                        st.error("Se necesitan al menos 3 observaciones válidas para calcular la correlación")
                    else:
                        # Pruebas de normalidad
                        shapiro_stat1, shapiro_p1 = shapiro(clean_data[var1])
                        shapiro_stat2, shapiro_p2 = shapiro(clean_data[var2])
                        
                        normal1 = shapiro_p1 > alpha_corr
                        normal2 = shapiro_p2 > alpha_corr
                        
                        # Seleccionar método de correlación
                        if normal1 and normal2:
                            corr, p_value = stats.pearsonr(clean_data[var1], clean_data[var2])
                            method = "Pearson"
                            method_explanation = "**Correlación de Pearson:** Mide la relación lineal entre variables normales"
                        else:
                            corr, p_value = stats.spearmanr(clean_data[var1], clean_data[var2])
                            method = "Spearman"
                            method_explanation = "**Correlación de Spearman:** Mide la relación monotónica (no necesariamente lineal)"
                        
                        # Resultados
                        st.subheader("📊 Resultados del Análisis de Correlación")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Método utilizado", method)
                        with col2:
                            st.metric("Coeficiente de correlación", f"{corr:.4f}")
                        with col3:
                            st.metric("p-valor", f"{p_value:.4f}")
                        
                        st.info(method_explanation)
                        
                        # Interpretación de la fuerza
                        if abs(corr) < 0.1:
                            strength = "muy débil o inexistente"
                        elif abs(corr) < 0.3:
                            strength = "débil"
                        elif abs(corr) < 0.5:
                            strength = "moderada"
                        elif abs(corr) < 0.7:
                            strength = "fuerte"
                        else:
                            strength = "muy fuerte"
                        
                        # Dirección
                        direction = "positiva" if corr > 0 else "negativa"
                        
                        st.write(f"**Interpretación:** La correlación entre **{var1}** y **{var2}** es {strength} y {direction}.")
                        
                        # Significancia estadística
                        if p_value < alpha_corr:
                            st.success("✅ **La correlación es estadísticamente significativa**")
                        else:
                            st.warning("⚠️ **La correlación no es estadísticamente significativa**")
                        
                        # Gráfico de dispersión
                        st.subheader("📈 Gráfico de Dispersión")
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.scatterplot(data=clean_data, x=var1, y=var2, alpha=0.6, ax=ax)
                        
                        # Añadir línea de tendencia
                        z = np.polyfit(clean_data[var1], clean_data[var2], 1)
                        p = np.poly1d(z)
                        ax.plot(clean_data[var1], p(clean_data[var1]), "r--", alpha=0.8)
                        
                        ax.set_title(f'Correlación {method}: {var1} vs {var2}\n(r = {corr:.3f}, p = {p_value:.4f})')
                        ax.set_xlabel(var1)
                        ax.set_ylabel(var2)
                        ax.grid(True, alpha=0.3)
                        
                        st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"Error en análisis de correlación: {e}")
        else:
            st.warning("Se necesitan al menos 2 variables numéricas para analizar correlaciones")

    # Las demás pestañas (tab5 a tab8) se mantienen exactamente iguales...

# Mensaje final si no hay datos cargados
else:
    st.info("👆 Por favor, carga un archivo de datos en la barra lateral para comenzar el análisis.")

# Footer
st.markdown("---")
st.markdown(
    "**Analytics Statistics Assistant** - Herramienta para análisis estadísticos generales"
)