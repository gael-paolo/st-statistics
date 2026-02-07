# -*- coding: utf-8 -*-
"""
Análisis Estadístico Universal con Asistente AI
Aplicación para análisis estadísticos descriptivos e inferenciales
con integración de OpenAI GPT para recomendaciones y explicaciones
"""

import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import anderson, shapiro
import statsmodels.api as sm
from statsmodels.formula.api import ols
from ydata_profiling import ProfileReport
import io
from openai import OpenAI
import os
import sys
import traceback
from datetime import datetime

# Configuración de la página
st.set_page_config(
    page_title="Analizador Estadístico Universal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🤖 Analizador Estadístico Inteligente")
st.markdown("""
Esta aplicación te ayuda a realizar análisis estadísticos descriptivos e inferenciales para cualquier tipo de datos.
Carga tus datos y consulta al asistente AI qué análisis realizar, luego ejecuta las funciones disponibles.
""")

# Sidebar para configuración
st.sidebar.header("🔧 Configuración")

# Configuración de OpenAI API
st.sidebar.subheader("Configuración de OpenAI")
openai_api_key = st.sidebar.text_input("Ingresa tu API Key de OpenAI:", type="password")

# Variable global para el cliente OpenAI
openai_client = None

if openai_api_key:
    try:
        openai_client = OpenAI(api_key=openai_api_key)
        st.sidebar.success("✅ OpenAI configurado correctamente")
    except Exception as e:
        st.sidebar.error(f"Error configurando OpenAI: {e}")
else:
    st.sidebar.warning("⚠️ Ingresa tu API Key de OpenAI para usar las recomendaciones")

# Función para consultar a OpenAI (CORREGIDA para OpenAI >= 1.0.0)
def consultar_openai(prompt, max_tokens=1500, temperature=0.7, model="gpt-4"):
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

# Función alternativa para versiones antiguas de OpenAI (mantener compatibilidad)
def consultar_openai_legacy(prompt, max_tokens=1500, temperature=0.7, model="gpt-4"):
    """Función alternativa para compatibilidad con versiones antiguas"""
    try:
        import openai as openai_old
        response = openai_old.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "Eres un experto en estadística aplicada y análisis de datos. Proporcionas explicaciones claras, precisas y prácticas."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error con API antigua: {str(e)}"

# ASISTENTE TEÓRICO EN ESTADÍSTICA
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
                # Preparar prompt para OpenAI
                prompt = f"""
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
                
                response = consultar_openai(prompt)
                st.success("📚 Respuesta del Experto en Estadística:")
                
                # Mejorar la presentación de la respuesta
                st.markdown("---")
                st.markdown(response)
                st.markdown("---")
                
            except Exception as e:
                # Intentar con API antigua como fallback
                try:
                    import openai as openai_old
                    openai_old.api_key = openai_api_key
                    response = consultar_openai_legacy(prompt)
                    st.success("📚 Respuesta del Experto en Estadística (API legacy):")
                    st.markdown("---")
                    st.markdown(response)
                    st.markdown("---")
                except:
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

# ============================================================================
# FUNCIONES DE ANÁLISIS ESTADÍSTICO (REUTILIZABLES)
# ============================================================================

# 1. Funciones de muestreo
def generate_sample(df, sample_size, method="simple", stratify_col=None, random_state=None):
    """
    Genera un muestreo a partir de un DataFrame.
    
    Args:
        df: DataFrame original
        sample_size: Tamaño de la muestra (int o float para porcentaje)
        method: 'simple' o 'stratified'
        stratify_col: Columna para estratificación
        random_state: Semilla aleatoria
        
    Returns:
        DataFrame con la muestra
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
    Calcula el tamaño de muestra requerido.
    
    Args:
        population_size: Tamaño de la población
        margin_of_error: Margen de error deseado
        confidence_level: Nivel de confianza
        proportion: Proporción esperada
        
    Returns:
        Tamaño de muestra mínimo
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

    # Ajuste por población finita
    adjusted_sample_size = sample_size / (1 + (sample_size - 1) / population_size)

    # Redondear al entero superior
    final_sample_size = int(np.ceil(adjusted_sample_size))

    return final_sample_size

# 2. Funciones de cálculo de tamaño del efecto
def calculate_effect_size(test_type, data1=None, data2=None, paired_data=None, var_before=None, var_after=None, pop_mean=0):
    """
    Calcula el tamaño del efecto para diferentes pruebas.
    
    Args:
        test_type: Tipo de prueba ('Una muestra', 'Muestras independientes', 'Muestras pareadas')
        data1, data2: Datos para pruebas independientes
        paired_data: DataFrame para pruebas pareadas
        var_before, var_after: Variables para pruebas pareadas
        pop_mean: Media poblacional para prueba de una muestra
        
    Returns:
        Tamaño del efecto (Cohen's d)
    """
    try:
        if test_type == "Una muestra":
            # Cohen's d para una muestra
            d = (data1.mean() - pop_mean) / data1.std()
            return abs(d)
        elif test_type == "Muestras independientes":
            # Cohen's d para muestras independientes
            n1, n2 = len(data1), len(data2)
            pooled_std = np.sqrt(((n1-1)*data1.std()**2 + (n2-1)*data2.std()**2) / (n1 + n2 - 2))
            d = (data1.mean() - data2.mean()) / pooled_std
            return abs(d)
        elif test_type == "Muestras pareadas":
            # Cohen's d para muestras pareadas
            differences = paired_data[var_after] - paired_data[var_before]
            d = differences.mean() / differences.std()
            return abs(d)
    except:
        return 0

def interpret_effect_size(d):
    """Interpreta el tamaño del efecto según Cohen"""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "Muy pequeño", "#808080"
    elif abs_d < 0.5:
        return "Pequeño", "#FF6B6B"
    elif abs_d < 0.8:
        return "Mediano", "#FFA726"
    elif abs_d < 1.2:
        return "Grande", "#4CAF50"
    else:
        return "Muy grande", "#2E7D32"

# 3. Funciones para pruebas estadísticas
def run_shapiro_wilk(data, alpha=0.05):
    """Ejecuta prueba de Shapiro-Wilk"""
    stat, p = shapiro(data)
    return {
        'statistic': stat,
        'p_value': p,
        'is_normal': p > alpha,
        'test': 'Shapiro-Wilk'
    }

def run_anderson_darling(data, alpha=0.05):
    """Ejecuta prueba de Anderson-Darling"""
    result = anderson(data, dist='norm')
    
    # Encontrar el valor crítico más cercano al alpha
    significance_levels = [0.15, 0.10, 0.05, 0.025, 0.01]
    closest_alpha = min(significance_levels, key=lambda x: abs(x - alpha))
    idx = significance_levels.index(closest_alpha)
    critical_value = result.critical_values[idx]
    
    return {
        'statistic': result.statistic,
        'critical_value': critical_value,
        'is_normal': result.statistic < critical_value,
        'test': 'Anderson-Darling'
    }

def run_levene_test(data1, data2, alpha=0.05):
    """Ejecuta prueba de Levene para igualdad de varianzas"""
    stat, p = stats.levene(data1, data2)
    return {
        'statistic': stat,
        'p_value': p,
        'equal_var': p > alpha,
        'test': 'Levene'
    }

def run_bartlett_test(data1, data2, alpha=0.05):
    """Ejecuta prueba de Bartlett para igualdad de varianzas"""
    stat, p = stats.bartlett(data1, data2)
    return {
        'statistic': stat,
        'p_value': p,
        'equal_var': p > alpha,
        'test': 'Bartlett'
    }

# ============================================================================
# CARGA DE DATOS
# ============================================================================

st.sidebar.subheader("Carga de Datos")
uploaded_file = st.sidebar.file_uploader("Sube tu archivo Excel o CSV", type=['xlsx', 'csv', 'xls'])

@st.cache_data
def load_data(file):
    """Carga datos desde archivo CSV o Excel"""
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, encoding='utf-8')
        else:
            # Intentar diferentes engines para Excel
            try:
                df = pd.read_excel(file, engine='openpyxl')
            except:
                df = pd.read_excel(file, engine='xlrd')
        
        # Limpiar columnas Unnamed
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        return df
    except Exception as e:
        st.error(f"Error cargando archivo: {e}")
        return None

df = None
if uploaded_file is not None:
    df = load_data(uploaded_file)
    if df is not None:
        st.sidebar.success(f"✅ Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
        
        # Mostrar información básica
        st.subheader("📋 Vista previa de los datos")
        st.dataframe(df.head())
        
        # Información básica del dataset
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total de registros", df.shape[0])
        with col2:
            st.metric("Total de variables", df.shape[1])
        with col3:
            st.metric("Valores faltantes", df.isnull().sum().sum())
        with col4:
            st.metric("Memoria usada", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Identificar tipos de variables
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Mostrar resumen de tipos de variables
        st.info(f"**Variables numéricas:** {len(numeric_cols)} | **Variables categóricas:** {len(categorical_cols)}")

# ============================================================================
# ASISTENTE DE ANÁLISIS PARA DATOS ESPECÍFICOS
# ============================================================================

if df is not None:
    st.subheader("🤖 Asistente de Análisis para tus Datos")
    st.markdown("Consulta recomendaciones específicas basadas en los datos que has cargado.")
    
    user_question = st.text_area(
        "Describe tu objetivo de análisis o pregunta qué análisis realizar con tus datos:",
        placeholder="Ej: Quiero analizar si hay diferencias significativas entre grupos, identificar correlaciones, validar hipótesis específicas...",
        height=100,
        key="business_question_main"
    )
    
    if st.button("Obtener recomendaciones de análisis", key="business_recommendations_main") and user_question:
        if openai_api_key:
            with st.spinner("Analizando tus datos y generando recomendaciones..."):
                try:
                    # Preparar resumen de datos para el prompt
                    data_summary = f"""
                    Resumen del dataset:
                    - Dimensiones: {df.shape[0]} filas × {df.shape[1]} columnas
                    - Variables numéricas: {', '.join(numeric_cols[:10])}{'...' if len(numeric_cols) > 10 else ''}
                    - Variables categóricas: {', '.join(categorical_cols[:10])}{'...' if len(categorical_cols) > 10 else ''}
                    - Valores faltantes: {df.isnull().sum().sum()}
                    
                    Pregunta/objetivo del usuario: {user_question}
                    
                    Estadísticas descriptivas básicas:
                    {df.describe().to_string() if len(numeric_cols) > 0 else 'No hay variables numéricas'}
                    """
                    
                    prompt = f"""
                    Como experto en análisis estadístico, analiza el siguiente caso:
                    
                    {data_summary}
                    
                    Basándote en los datos y el objetivo del usuario, recomienda un plan de análisis que incluya:
                    
                    1. **Análisis descriptivo necesario**: Qué estadísticas descriptivas calcular
                    2. **Pruebas de supuestos**: Qué validaciones realizar (normalidad, homogeneidad de varianzas)
                    3. **Análisis inferencial**: Qué pruebas estadísticas aplicar y por qué
                    4. **Visualizaciones recomendadas**: Qué gráficos crear para entender mejor los datos
                    5. **Interpretación esperada**: Cómo interpretar los resultados
                    6. **Limitaciones y consideraciones**: Qué precauciones tomar
                    
                    Para cada recomendación, explica:
                    - Qué pregunta de investigación ayuda a responder
                    - Qué variables utilizar
                    - Cómo interpretar los resultados
                    - Qué alternativas considerar si no se cumplen los supuestos
                    
                    Mantén las recomendaciones prácticas y aplicables.
                    """
                    
                    response = consultar_openai(prompt, max_tokens=2000)
                    st.success("🎯 Recomendaciones de Análisis para tus Datos:")
                    st.markdown("---")
                    st.markdown(response)
                    st.markdown("---")
                    
                except Exception as e:
                    st.error(f"Error consultando a OpenAI: {e}")
        else:
            st.error("🔑 Necesitas configurar tu API Key de OpenAI en la barra lateral")

# ============================================================================
# SECCIÓN DE ANÁLISIS ESTADÍSTICOS
# ============================================================================

if df is not None:
    st.header("📊 Análisis Estadísticos")
    
    # Crear pestañas para diferentes tipos de análisis
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "🎯 Muestreo", 
        "📈 Descriptivos", 
        "🔍 Normalidad", 
        "📉 Correlaciones",
        "⚖️ Homogeneidad",
        "✅ Pruebas T",
        "📊 ANOVA",
        "🔄 No Paramétricas"
    ])
    
    # ========================================================================
    # PESTAÑA 1: MUESTREO
    # ========================================================================
    with tab1:
        st.subheader("🎯 Análisis de Muestreo")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Generar Muestra")
            st.markdown("Genera una muestra representativa de tus datos para análisis.")
            
            sample_method = st.radio(
                "Método de muestreo:",
                ["simple", "stratified"],
                format_func=lambda x: "Aleatorio Simple" if x == "simple" else "Estratificado",
                key="sample_method"
            )
            
            sample_size_type = st.radio(
                "Tipo de tamaño de muestra:",
                ["percentage", "absolute"],
                format_func=lambda x: "Porcentaje" if x == "percentage" else "Número absoluto",
                key="sample_size_type"
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
                    help="La muestra mantendrá las proporciones de esta variable categórica",
                    key="stratify_column"
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
                    
                    # Opción para descargar la muestra
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        sample_df.to_excel(writer, index=False, sheet_name='Muestra')
                    
                    st.download_button(
                        label="📥 Descargar muestra como Excel",
                        data=output.getvalue(),
                        file_name=f"muestra_generada_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
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
                help="Número total de elementos en la población de estudio",
                key="population_size"
            )
            
            margin_error = st.slider(
                "Margen de error (%):",
                min_value=1,
                max_value=10,
                value=5,
                help="Precisión deseada en los resultados (±%)",
                key="margin_error"
            ) / 100.0
            
            confidence_level = st.slider(
                "Nivel de confianza (%):",
                min_value=80,
                max_value=99,
                value=95,
                help="Probabilidad de que el resultado sea correcto",
                key="confidence_level"
            ) / 100.0
            
            proportion = st.slider(
                "Proporción esperada (%):",
                min_value=1,
                max_value=99,
                value=50,
                help="Proporción esperada de la característica en la población (usar 50% si es desconocida)",
                key="proportion"
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
                    
                except Exception as e:
                    st.error(f"❌ Error calculando tamaño de muestra: {e}")
    
    # ========================================================================
    # PESTAÑA 2: ANÁLISIS DESCRIPTIVOS
    # ========================================================================
    with tab2:
        st.subheader("📈 Análisis Descriptivo")
        
        # Selector de variables para análisis descriptivo
        if numeric_cols:
            selected_vars = st.multiselect(
                "Selecciona variables para análisis descriptivo:",
                numeric_cols,
                default=numeric_cols[:min(5, len(numeric_cols))],
                key="desc_vars"
            )
            
            if selected_vars:
                # Estadísticas descriptivas
                st.subheader("Estadísticas Descriptivas")
                desc_stats = df[selected_vars].describe().T
                desc_stats['cv'] = (desc_stats['std'] / desc_stats['mean']) * 100  # Coeficiente de variación
                desc_stats['skew'] = df[selected_vars].skew()
                desc_stats['kurtosis'] = df[selected_vars].kurtosis()
                
                st.dataframe(desc_stats.round(4))
                
                # Visualizaciones
                st.subheader("Visualizaciones")
                
                # Seleccionar variable específica para histograma y boxplot
                if selected_vars:
                    selected_var = st.selectbox(
                        "Selecciona variable para visualización detallada:",
                        selected_vars,
                        key="desc_var_detail"
                    )
                    
                    if selected_var:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Histograma
                            fig, ax = plt.subplots(figsize=(8, 6))
                            sns.histplot(df[selected_var].dropna(), kde=True, ax=ax)
                            ax.set_title(f'Distribución de {selected_var}')
                            ax.set_xlabel(selected_var)
                            ax.set_ylabel('Frecuencia')
                            st.pyplot(fig)
                        
                        with col2:
                            # Boxplot
                            fig, ax = plt.subplots(figsize=(8, 6))
                            sns.boxplot(y=df[selected_var].dropna(), ax=ax)
                            ax.set_title(f'Boxplot de {selected_var}')
                            ax.set_ylabel(selected_var)
                            st.pyplot(fig)
                        
                        # Gráfico de densidad comparativo si hay múltiples variables
                        if len(selected_vars) > 1:
                            st.subheader("Comparación de Distribuciones")
                            fig, ax = plt.subplots(figsize=(10, 6))
                            for var in selected_vars:
                                sns.kdeplot(df[var].dropna(), label=var, ax=ax)
                            ax.set_title('Comparación de Distribuciones')
                            ax.set_xlabel('Valor')
                            ax.set_ylabel('Densidad')
                            ax.legend()
                            st.pyplot(fig)
        else:
            st.warning("No hay variables numéricas para análisis descriptivo")
        
        # Reporte completo
        st.subheader("Reporte Completo de Análisis Exploratorio")
        
        if st.button("📊 Generar Reporte Completo", key="generate_full_report"):
            with st.spinner("Generando reporte exploratorio... Esto puede tomar unos segundos"):
                try:
                    profile = ProfileReport(df, title="Análisis Exploratorio de Datos")
                    html_content = profile.to_html()
                    
                    st.download_button(
                        label="📥 Descargar Reporte Completo (HTML)",
                        data=html_content,
                        file_name=f"reporte_exploratorio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                        mime="text/html"
                    )
                    
                    st.success("✅ Reporte generado correctamente. Haz clic en el botón de descarga.")
                    
                except Exception as e:
                    st.error(f"Error generando reporte: {e}")
    
    # ========================================================================
    # PESTAÑA 3: PRUEBAS DE NORMALIDAD
    # ========================================================================
    with tab3:
        st.subheader("🔍 Pruebas de Normalidad")
        
        if numeric_cols:
            selected_var = st.selectbox(
                "Selecciona variable para prueba de normalidad:",
                numeric_cols,
                key="normal_var"
            )
            
            alpha = st.slider(
                "Nivel de significancia (α):",
                0.01, 0.10, 0.05,
                key="normal_alpha"
            )
            
            if st.button("📊 Ejecutar Pruebas de Normalidad", key="run_normality_tests"):
                try:
                    data = df[selected_var].dropna()
                    
                    if len(data) < 3:
                        st.error("Se necesitan al menos 3 observaciones")
                    else:
                        # Ejecutar pruebas
                        shapiro_result = run_shapiro_wilk(data, alpha)
                        anderson_result = run_anderson_darling(data, alpha)
                        
                        # Mostrar resultados
                        st.subheader("📋 Resultados")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Prueba", shapiro_result['test'])
                            st.metric("Estadístico", f"{shapiro_result['statistic']:.4f}")
                            st.metric("p-valor", f"{shapiro_result['p_value']:.4f}")
                            if shapiro_result['is_normal']:
                                st.success("✅ Los datos parecen normales")
                            else:
                                st.error("❌ Los datos NO parecen normales")
                        
                        with col2:
                            st.metric("Prueba", anderson_result['test'])
                            st.metric("Estadístico", f"{anderson_result['statistic']:.4f}")
                            st.metric("Valor crítico", f"{anderson_result['critical_value']:.4f}")
                            if anderson_result['is_normal']:
                                st.success("✅ Los datos parecen normales")
                            else:
                                st.error("❌ Los datos NO parecen normales")
                        
                        # Visualizaciones
                        st.subheader("📊 Diagnóstico Visual")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Histograma con curva normal
                            fig, ax = plt.subplots(figsize=(8, 6))
                            sns.histplot(data, kde=True, stat='density', ax=ax, alpha=0.7)
                            
                            # Curva normal teórica
                            mu, sigma = data.mean(), data.std()
                            x = np.linspace(data.min(), data.max(), 100)
                            ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, 
                                   label=f'Normal teórica\n(μ={mu:.2f}, σ={sigma:.2f})')
                            
                            ax.set_title(f'Distribución de {selected_var}')
                            ax.legend()
                            st.pyplot(fig)
                        
                        with col2:
                            # Q-Q plot
                            fig, ax = plt.subplots(figsize=(8, 6))
                            stats.probplot(data, dist="norm", plot=ax)
                            ax.set_title(f'Q-Q Plot de {selected_var}')
                            st.pyplot(fig)
                        
                        # Interpretación con OpenAI
                        if openai_api_key and st.button("📝 Obtener interpretación detallada", key="normality_interpretation"):
                            with st.spinner("Generando interpretación..."):
                                prompt = f"""
                                Como experto en estadística, interpreta los siguientes resultados de pruebas de normalidad:
                                
                                Variable analizada: {selected_var}
                                Tamaño de muestra: {len(data)}
                                Nivel de significancia (α): {alpha}
                                
                                Resultados:
                                1. Shapiro-Wilk:
                                   - Estadístico W: {shapiro_result['statistic']:.4f}
                                   - p-valor: {shapiro_result['p_value']:.4f}
                                   - Conclusión: {'Normal' if shapiro_result['is_normal'] else 'No normal'}
                                
                                2. Anderson-Darling:
                                   - Estadístico A-D: {anderson_result['statistic']:.4f}
                                   - Valor crítico: {anderson_result['critical_value']:.4f}
                                   - Conclusión: {'Normal' if anderson_result['is_normal'] else 'No normal'}
                                
                                Proporciona una interpretación detallada que incluya:
                                1. Qué significa cada prueba
                                2. Cómo interpretar los resultados
                                3. Recomendaciones basadas en los resultados (qué pruebas usar)
                                4. Posibles acciones si los datos no son normales
                                """
                                
                                interpretation = consultar_openai(prompt)
                                st.markdown("---")
                                st.markdown("### 📚 Interpretación Detallada")
                                st.markdown(interpretation)
                                st.markdown("---")
                        
                except Exception as e:
                    st.error(f"Error en pruebas de normalidad: {e}")
        else:
            st.warning("No hay variables numéricas para análisis de normalidad")
    
    # ========================================================================
    # PESTAÑA 4: CORRELACIONES
    # ========================================================================
    with tab4:
        st.subheader("📉 Análisis de Correlación")
        
        if len(numeric_cols) >= 2:
            # Selección de variables
            col1, col2 = st.columns(2)
            with col1:
                var1 = st.selectbox("Variable X:", numeric_cols, key="corr_var1")
            with col2:
                # Filtrar para no seleccionar la misma variable
                available_vars = [v for v in numeric_cols if v != var1]
                var2 = st.selectbox("Variable Y:", available_vars, key="corr_var2")
            
            # Opciones de análisis
            col1, col2 = st.columns(2)
            with col1:
                correlation_method = st.radio(
                    "Método de correlación:",
                    ["pearson", "spearman", "kendall"],
                    format_func=lambda x: {
                        "pearson": "Pearson (lineal)",
                        "spearman": "Spearman (monotónica)",
                        "kendall": "Kendall (rangos)"
                    }[x],
                    key="corr_method"
                )
            
            with col2:
                alpha_corr = st.slider(
                    "Nivel de significancia (α):",
                    0.01, 0.10, 0.05,
                    key="corr_alpha"
                )
            
            if st.button("🔍 Analizar Correlación", key="analyze_correlation"):
                try:
                    # Limpiar datos
                    clean_data = df[[var1, var2]].dropna()
                    
                    if len(clean_data) < 3:
                        st.error("Se necesitan al menos 3 observaciones válidas")
                    else:
                        # Calcular correlación según método
                        if correlation_method == "pearson":
                            corr, p_value = stats.pearsonr(clean_data[var1], clean_data[var2])
                        elif correlation_method == "spearman":
                            corr, p_value = stats.spearmanr(clean_data[var1], clean_data[var2])
                        else:  # kendall
                            corr, p_value = stats.kendalltau(clean_data[var1], clean_data[var2])
                        
                        # Resultados
                        st.subheader("📊 Resultados de Correlación")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Método", correlation_method.capitalize())
                        with col2:
                            st.metric("Coeficiente", f"{corr:.4f}")
                        with col3:
                            st.metric("p-valor", f"{p_value:.4f}")
                        
                        # Interpretación de fuerza
                        abs_corr = abs(corr)
                        if abs_corr < 0.1:
                            strength = "muy débil o inexistente"
                        elif abs_corr < 0.3:
                            strength = "débil"
                        elif abs_corr < 0.5:
                            strength = "moderada"
                        elif abs_corr < 0.7:
                            strength = "fuerte"
                        else:
                            strength = "muy fuerte"
                        
                        # Significancia
                        is_significant = p_value < alpha_corr
                        
                        st.info(f"""
                        **Interpretación:**
                        - La correlación entre **{var1}** y **{var2}** es **{strength}**
                        - Dirección: {'Positiva' if corr > 0 else 'Negativa'}
                        - Significancia estadística: {'Significativa' if is_significant else 'No significativa'}
                        """)
                        
                        # Gráfico de dispersión
                        st.subheader("📈 Gráfico de Dispersión")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        sns.scatterplot(data=clean_data, x=var1, y=var2, alpha=0.6, ax=ax, s=50)
                        
                        # Línea de tendencia
                        if correlation_method == "pearson":
                            z = np.polyfit(clean_data[var1], clean_data[var2], 1)
                            p = np.poly1d(z)
                            ax.plot(clean_data[var1], p(clean_data[var1]), "r--", alpha=0.8, 
                                   label=f'Tendencia lineal (r={corr:.3f})')
                        
                        ax.set_title(f'Correlación {correlation_method.capitalize()}: {var1} vs {var2}')
                        ax.set_xlabel(var1)
                        ax.set_ylabel(var2)
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        
                        st.pyplot(fig)
                        
                        # Matriz de correlación si hay múltiples variables
                        if len(numeric_cols) > 2 and st.checkbox("Mostrar matriz de correlación completa", key="show_corr_matrix"):
                            st.subheader("📊 Matriz de Correlación")
                            
                            # Seleccionar variables para la matriz
                            selected_for_matrix = st.multiselect(
                                "Selecciona variables para la matriz:",
                                numeric_cols,
                                default=numeric_cols[:min(10, len(numeric_cols))],
                                key="corr_matrix_vars"
                            )
                            
                            if selected_for_matrix:
                                corr_matrix = df[selected_for_matrix].corr(method=correlation_method)
                                
                                # Mostrar matriz numérica
                                st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm', vmin=-1, vmax=1))
                                
                                # Heatmap visual
                                fig, ax = plt.subplots(figsize=(10, 8))
                                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                                sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", 
                                          cmap='coolwarm', center=0, square=True, ax=ax)
                                ax.set_title(f'Matriz de Correlación ({correlation_method.capitalize()})')
                                st.pyplot(fig)
                        
                        # Interpretación con OpenAI
                        if openai_api_key and st.button("📝 Obtener interpretación experta", key="corr_interpretation"):
                            with st.spinner("Generando interpretación experta..."):
                                prompt = f"""
                                Como experto en estadística, interpreta los siguientes resultados de correlación:
                                
                                Variables: {var1} y {var2}
                                Método: {correlation_method.capitalize()}
                                Coeficiente de correlación: {corr:.4f}
                                p-valor: {p_value:.4f}
                                Nivel de significancia: {alpha_corr}
                                Tamaño de muestra: {len(clean_data)}
                                
                                Proporciona una interpretación completa que incluya:
                                1. Qué significa este coeficiente de correlación
                                2. Cómo interpretar la fuerza y dirección de la relación
                                3. Si la correlación es estadísticamente significativa y qué implica
                                4. Limitaciones y precauciones en la interpretación
                                5. Recomendaciones para análisis adicionales
                                
                                Importante: Explicar claramente que correlación no implica causalidad.
                                """
                                
                                interpretation = consultar_openai(prompt)
                                st.markdown("---")
                                st.markdown("### 📚 Interpretación Experta")
                                st.markdown(interpretation)
                                st.markdown("---")
                        
                except Exception as e:
                    st.error(f"Error en análisis de correlación: {e}")
        else:
            st.warning("Se necesitan al menos 2 variables numéricas para análisis de correlación")
    
    # ========================================================================
    # PESTAÑA 5: HOMOGENEIDAD DE VARIANZAS
    # ========================================================================
    with tab5:
        st.subheader("⚖️ Homogeneidad de Varianzas")
        
        if numeric_cols and categorical_cols:
            # Selección de variables
            col1, col2 = st.columns(2)
            with col1:
                num_var = st.selectbox("Variable numérica:", numeric_cols, key="homo_num_var")
            with col2:
                cat_var = st.selectbox("Variable categórica:", categorical_cols, key="homo_cat_var")
            
            alpha_homo = st.slider(
                "Nivel de significancia (α):",
                0.01, 0.10, 0.05,
                key="homo_alpha"
            )
            
            if st.button("📊 Ejecutar Pruebas de Homogeneidad", key="run_homogeneity_tests"):
                try:
                    # Preparar datos por grupos
                    groups = []
                    group_names = []
                    
                    for group in df[cat_var].dropna().unique():
                        group_data = df[df[cat_var] == group][num_var].dropna()
                        if len(group_data) >= 2:
                            groups.append(group_data)
                            group_names.append(str(group))
                    
                    if len(groups) < 2:
                        st.error("Se necesitan al menos 2 grupos con datos válidos")
                    else:
                        # Ejecutar pruebas
                        levene_result = run_levene_test(*groups, alpha=alpha_homo)
                        bartlett_result = run_bartlett_test(*groups, alpha=alpha_homo)
                        
                        # Mostrar resultados
                        st.subheader("📋 Resultados")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Prueba", levene_result['test'])
                            st.metric("Estadístico", f"{levene_result['statistic']:.4f}")
                            st.metric("p-valor", f"{levene_result['p_value']:.4f}")
                            if levene_result['equal_var']:
                                st.success("✅ Varianzas homogéneas")
                            else:
                                st.error("❌ Varianzas NO homogéneas")
                        
                        with col2:
                            st.metric("Prueba", bartlett_result['test'])
                            st.metric("Estadístico", f"{bartlett_result['statistic']:.4f}")
                            st.metric("p-valor", f"{bartlett_result['p_value']:.4f}")
                            if bartlett_result['equal_var']:
                                st.success("✅ Varianzas homogéneas")
                            else:
                                st.error("❌ Varianzas NO homogéneas")
                        
                        # Estadísticas por grupo
                        st.subheader("📊 Estadísticas por Grupo")
                        stats_data = []
                        for name, data in zip(group_names, groups):
                            stats_data.append({
                                'Grupo': name,
                                'n': len(data),
                                'Media': f"{data.mean():.4f}",
                                'Desviación': f"{data.std():.4f}",
                                'Varianza': f"{data.var():.4f}"
                            })
                        
                        stats_df = pd.DataFrame(stats_data)
                        st.dataframe(stats_df)
                        
                        # Visualizaciones
                        st.subheader("📈 Comparación Visual de Varianzas")
                        
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                        
                        # Boxplot
                        plot_data = []
                        for name, data in zip(group_names, groups):
                            for value in data:
                                plot_data.append({'Grupo': name, 'Valor': value})
                        
                        plot_df = pd.DataFrame(plot_data)
                        sns.boxplot(data=plot_df, x='Grupo', y='Valor', ax=ax1)
                        ax1.set_title(f'Distribución por Grupo\n({num_var})')
                        ax1.tick_params(axis='x', rotation=45)
                        
                        # Gráfico de varianzas
                        variances = [d.var() for d in groups]
                        ax2.bar(group_names, variances, alpha=0.7)
                        ax2.set_title('Varianzas por Grupo')
                        ax2.set_ylabel('Varianza')
                        ax2.tick_params(axis='x', rotation=45)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Interpretación con OpenAI
                        if openai_api_key and st.button("📝 Obtener recomendaciones", key="homo_interpretation"):
                            with st.spinner("Generando recomendaciones..."):
                                prompt = f"""
                                Como experto en estadística, interpreta los siguientes resultados de homogeneidad de varianzas:
                                
                                Variable numérica: {num_var}
                                Variable categórica: {cat_var}
                                Número de grupos: {len(groups)}
                                Nivel de significancia: {alpha_homo}
                                
                                Resultados:
                                1. Prueba de Levene:
                                   - p-valor: {levene_result['p_value']:.4f}
                                   - Conclusión: {'Varianzas homogéneas' if levene_result['equal_var'] else 'Varianzas NO homogéneas'}
                                
                                2. Prueba de Bartlett:
                                   - p-valor: {bartlett_result['p_value']:.4f}
                                   - Conclusión: {'Varianzas homogéneas' if bartlett_result['equal_var'] else 'Varianzas NO homogéneas'}
                                
                                Estadísticas por grupo:
                                {stats_df.to_string()}
                                
                                Proporciona recomendaciones prácticas que incluyan:
                                1. Qué pruebas estadísticas son apropiadas basadas en estos resultados
                                2. Si se deben usar pruebas paramétricas o no paramétricas
                                3. Cómo proceder con análisis adicionales
                                4. Posibles transformaciones de datos si es necesario
                                """
                                
                                interpretation = consultar_openai(prompt)
                                st.markdown("---")
                                st.markdown("### 📚 Recomendaciones Expertas")
                                st.markdown(interpretation)
                                st.markdown("---")
                        
                except Exception as e:
                    st.error(f"Error en pruebas de homogeneidad: {e}")
        else:
            st.warning("Se necesitan variables numéricas y categóricas para evaluar homogeneidad")
    
    # ========================================================================
    # PESTAÑA 6: PRUEBAS T
    # ========================================================================
    with tab6:
        st.subheader("✅ Pruebas T")
        
        # Selección del tipo de prueba
        test_type = st.radio(
            "Tipo de prueba T:",
            ["Una muestra", "Muestras independientes", "Muestras pareadas"],
            key="ttest_type"
        )
        
        # Configuración común
        col1, col2 = st.columns(2)
        with col1:
            alpha_ttest = st.slider(
                "Nivel de significancia (α):",
                0.01, 0.10, 0.05,
                key="ttest_alpha"
            )
        with col2:
            alternative = st.selectbox(
                "Hipótesis alternativa:",
                ["two-sided", "less", "greater"],
                format_func=lambda x: {
                    "two-sided": "Bilateral (≠)",
                    "less": "Unilateral izquierda (<)",
                    "greater": "Unilateral derecha (>)"
                }[x],
                key="ttest_alternative"
            )
        
        # Prueba T para una muestra
        if test_type == "Una muestra" and numeric_cols:
            st.subheader("Prueba T para Una Muestra")
            
            var_onesample = st.selectbox("Variable numérica:", numeric_cols, key="onesample_var")
            pop_mean = st.number_input("Media poblacional de referencia:", value=0.0, key="pop_mean")
            
            if st.button("📊 Ejecutar Prueba T Una Muestra", key="run_onesample_ttest"):
                try:
                    data = df[var_onesample].dropna()
                    
                    if len(data) < 2:
                        st.error("Se necesitan al menos 2 observaciones")
                    else:
                        # Ejecutar prueba
                        t_stat, p_value = stats.ttest_1samp(data, pop_mean)
                        
                        # Ajuste para pruebas unilaterales
                        if alternative == "less" and data.mean() < pop_mean:
                            p_value = p_value / 2
                        elif alternative == "greater" and data.mean() > pop_mean:
                            p_value = p_value / 2
                        elif alternative != "two-sided":
                            p_value = 1 - p_value / 2
                        
                        # Calcular intervalo de confianza
                        ci_low, ci_high = stats.t.interval(
                            1 - alpha_ttest, 
                            len(data) - 1, 
                            loc=data.mean(), 
                            scale=stats.sem(data)
                        )
                        
                        # Tamaño del efecto
                        effect_size = calculate_effect_size("Una muestra", data1=data, pop_mean=pop_mean)
                        effect_magnitude, effect_color = interpret_effect_size(effect_size)
                        
                        # Mostrar resultados
                        st.subheader("📋 Resultados")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Estadístico t", f"{t_stat:.4f}")
                        with col2:
                            st.metric("p-valor", f"{p_value:.4f}")
                        with col3:
                            st.metric("Significativo", "Sí" if p_value < alpha_ttest else "No")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Media muestral", f"{data.mean():.4f}")
                            st.metric("Desviación estándar", f"{data.std():.4f}")
                        with col2:
                            st.metric("Media referencia", f"{pop_mean:.4f}")
                            st.metric("Tamaño muestra", len(data))
                        
                        st.metric(f"IC del {(1-alpha_ttest)*100:.0f}%", f"[{ci_low:.4f}, {ci_high:.4f}]")
                        st.metric("Tamaño efecto (d)", f"{effect_size:.4f}")
                        
                        # Interpretación
                        st.info(f"""
                        **Interpretación:**
                        - Tamaño del efecto: **{effect_magnitude}**
                        - {'Diferencias significativas' if p_value < alpha_ttest else 'Sin diferencias significativas'}
                        """)
                        
                except Exception as e:
                    st.error(f"Error en prueba T una muestra: {e}")
        
        # Prueba T para muestras independientes
        elif test_type == "Muestras independientes" and numeric_cols and categorical_cols:
            st.subheader("Prueba T para Muestras Independientes")
            
            var_independent = st.selectbox("Variable numérica:", numeric_cols, key="indep_var")
            group_var = st.selectbox("Variable categórica:", categorical_cols, key="group_var")
            
            # Verificar grupos
            unique_groups = df[group_var].dropna().unique()
            if len(unique_groups) == 2:
                group1, group2 = unique_groups
                
                if st.button("📊 Ejecutar Prueba T Independiente", key="run_independent_ttest"):
                    try:
                        data1 = df[df[group_var] == group1][var_independent].dropna()
                        data2 = df[df[group_var] == group2][var_independent].dropna()
                        
                        if len(data1) < 2 or len(data2) < 2:
                            st.error("Cada grupo necesita al menos 2 observaciones")
                        else:
                            # Verificar homogeneidad de varianzas
                            levene_stat, levene_p = stats.levene(data1, data2)
                            equal_var = levene_p > 0.05
                            
                            # Ejecutar prueba T
                            t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=equal_var)
                            
                            # Ajuste para pruebas unilaterales
                            if alternative == "less" and data1.mean() < data2.mean():
                                p_value = p_value / 2
                            elif alternative == "greater" and data1.mean() > data2.mean():
                                p_value = p_value / 2
                            elif alternative != "two-sided":
                                p_value = 1 - p_value / 2
                            
                            # Tamaño del efecto
                            effect_size = calculate_effect_size("Muestras independientes", data1=data1, data2=data2)
                            effect_magnitude, effect_color = interpret_effect_size(effect_size)
                            
                            # Mostrar resultados
                            st.subheader("📋 Resultados")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Estadístico t", f"{t_stat:.4f}")
                            with col2:
                                st.metric("p-valor", f"{p_value:.4f}")
                            with col3:
                                st.metric("Prueba", "Student" if equal_var else "Welch")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric(f"Media {group1}", f"{data1.mean():.4f}")
                                st.metric(f"Desviación {group1}", f"{data1.std():.4f}")
                                st.metric(f"n {group1}", len(data1))
                            with col2:
                                st.metric(f"Media {group2}", f"{data2.mean():.4f}")
                                st.metric(f"Desviación {group2}", f"{data2.std():.4f}")
                                st.metric(f"n {group2}", len(data2))
                            
                            st.metric("Diferencia medias", f"{data1.mean() - data2.mean():.4f}")
                            st.metric("Tamaño efecto (d)", f"{effect_size:.4f}")
                            
                            st.info(f"""
                            **Homogeneidad de varianzas (Levene):**
                            - p-valor: {levene_p:.4f}
                            - {'Varianzas homogéneas' if equal_var else 'Varianzas diferentes'}
                            """)
                            
                    except Exception as e:
                        st.error(f"Error en prueba T independiente: {e}")
            else:
                st.warning(f"La variable categórica debe tener exactamente 2 grupos (tiene {len(unique_groups)})")
        
        # Prueba T para muestras pareadas
        elif test_type == "Muestras pareadas" and len(numeric_cols) >= 2:
            st.subheader("Prueba T para Muestras Pareadas")
            
            col1, col2 = st.columns(2)
            with col1:
                var_before = st.selectbox("Variable 'Antes':", numeric_cols, key="before_var")
            with col2:
                var_after = st.selectbox("Variable 'Después':", numeric_cols, key="after_var")
            
            if st.button("📊 Ejecutar Prueba T Pareada", key="run_paired_ttest"):
                try:
                    # Filtrar pares completos
                    paired_data = df[[var_before, var_after]].dropna()
                    
                    if len(paired_data) < 2:
                        st.error("Se necesitan al menos 2 pares completos")
                    else:
                        # Ejecutar prueba
                        t_stat, p_value = stats.ttest_rel(paired_data[var_before], paired_data[var_after])
                        
                        # Ajuste para pruebas unilaterales
                        differences = paired_data[var_after] - paired_data[var_before]
                        if alternative == "less" and differences.mean() < 0:
                            p_value = p_value / 2
                        elif alternative == "greater" and differences.mean() > 0:
                            p_value = p_value / 2
                        elif alternative != "two-sided":
                            p_value = 1 - p_value / 2
                        
                        # Tamaño del efecto
                        effect_size = calculate_effect_size(
                            "Muestras pareadas", 
                            paired_data=paired_data, 
                            var_before=var_before, 
                            var_after=var_after
                        )
                        effect_magnitude, effect_color = interpret_effect_size(effect_size)
                        
                        # Mostrar resultados
                        st.subheader("📋 Resultados")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Estadístico t", f"{t_stat:.4f}")
                        with col2:
                            st.metric("p-valor", f"{p_value:.4f}")
                        with col3:
                            st.metric("Número de pares", len(paired_data))
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(f"Media '{var_before}'", f"{paired_data[var_before].mean():.4f}")
                            st.metric(f"Desviación '{var_before}'", f"{paired_data[var_before].std():.4f}")
                        with col2:
                            st.metric(f"Media '{var_after}'", f"{paired_data[var_after].mean():.4f}")
                            st.metric(f"Desviación '{var_after}'", f"{paired_data[var_after].std():.4f}")
                        
                        st.metric("Diferencia media", f"{differences.mean():.4f}")
                        st.metric("Tamaño efecto (d)", f"{effect_size:.4f}")
                        
                except Exception as e:
                    st.error(f"Error en prueba T pareada: {e}")
    
    # ========================================================================
    # PESTAÑA 7: ANOVA
    # ========================================================================
    with tab7:
        st.subheader("📊 Análisis de Varianza (ANOVA)")
        
        if numeric_cols and categorical_cols:
            # Selección de variables
            num_var = st.selectbox("Variable numérica:", numeric_cols, key="anova_num_var")
            cat_var = st.selectbox("Variable categórica:", categorical_cols, key="anova_cat_var")
            
            alpha_anova = st.slider(
                "Nivel de significancia (α):",
                0.01, 0.10, 0.05,
                key="anova_alpha"
            )
            
            if st.button("📊 Ejecutar ANOVA", key="run_anova"):
                try:
                    # Preparar datos
                    groups_data = []
                    group_names = []
                    
                    for group in df[cat_var].dropna().unique():
                        group_data = df[df[cat_var] == group][num_var].dropna()
                        if len(group_data) >= 2:
                            groups_data.append(group_data)
                            group_names.append(str(group))
                    
                    if len(groups_data) < 2:
                        st.error("Se necesitan al menos 2 grupos con datos válidos")
                    else:
                        # Ejecutar ANOVA
                        f_stat, p_value = stats.f_oneway(*groups_data)
                        
                        # Mostrar resultados
                        st.subheader("📋 Resultados del ANOVA")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Estadístico F", f"{f_stat:.4f}")
                        with col2:
                            st.metric("p-valor", f"{p_value:.4f}")
                        with col3:
                            st.metric("Significativo", "Sí" if p_value < alpha_anova else "No")
                        
                        # Estadísticas por grupo
                        st.subheader("📊 Estadísticas por Grupo")
                        stats_data = []
                        for name, data in zip(group_names, groups_data):
                            stats_data.append({
                                'Grupo': name,
                                'n': len(data),
                                'Media': f"{data.mean():.4f}",
                                'Desviación': f"{data.std():.4f}",
                                'Mínimo': f"{data.min():.4f}",
                                'Máximo': f"{data.max():.4f}"
                            })
                        
                        stats_df = pd.DataFrame(stats_data)
                        st.dataframe(stats_df)
                        
                        # Visualización
                        st.subheader("📈 Comparación Visual")
                        fig, ax = plt.subplots(figsize=(12, 6))
                        
                        plot_data = []
                        for name, data in zip(group_names, groups_data):
                            for value in data:
                                plot_data.append({'Grupo': name, 'Valor': value})
                        
                        plot_df = pd.DataFrame(plot_data)
                        sns.boxplot(data=plot_df, x='Grupo', y='Valor', ax=ax)
                        ax.set_title(f'ANOVA: {num_var} por {cat_var}\n(F={f_stat:.3f}, p={p_value:.4f})')
                        ax.tick_params(axis='x', rotation=45)
                        
                        # Agregar medias
                        for i, name in enumerate(group_names):
                            mean_val = groups_data[i].mean()
                            ax.text(i, mean_val, f'{mean_val:.2f}', 
                                  ha='center', va='bottom', fontweight='bold')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Pruebas post-hoc si ANOVA es significativo
                        if p_value < alpha_anova and len(groups_data) > 2:
                            st.subheader("🔍 Comparaciones Múltiples (Post-hoc)")
                            
                            try:
                                # Usar Tukey HSD de scipy si está disponible
                                from statsmodels.stats.multicomp import pairwise_tukeyhsd
                                
                                # Preparar datos para Tukey
                                tukey_data = df[[num_var, cat_var]].dropna()
                                tukey = pairwise_tukeyhsd(
                                    tukey_data[num_var], 
                                    tukey_data[cat_var], 
                                    alpha=alpha_anova
                                )
                                
                                # Mostrar resultados
                                result_df = pd.DataFrame(
                                    data=tukey._results_table.data[1:],
                                    columns=tukey._results_table.data[0]
                                )
                                st.dataframe(result_df)
                                
                                # Identificar diferencias significativas
                                sig_pairs = result_df[result_df['p-adj'] < alpha_anova]
                                if not sig_pairs.empty:
                                    st.write("**Diferencias significativas:**")
                                    for _, row in sig_pairs.iterrows():
                                        st.write(f"- {row['group1']} vs {row['group2']} (p-adj = {row['p-adj']:.4f})")
                                else:
                                    st.info("No hay diferencias significativas entre pares específicos")
                                    
                            except ImportError:
                                st.warning("Para comparaciones post-hoc, instala statsmodels: pip install statsmodels")
                            except Exception as e:
                                st.warning(f"No se pudo realizar análisis post-hoc: {e}")
                        
                except Exception as e:
                    st.error(f"Error en ANOVA: {e}")
        else:
            st.warning("Se necesitan variables numéricas y categóricas para ANOVA")
    
    # ========================================================================
    # PESTAÑA 8: PRUEBAS NO PARAMÉTRICAS
    # ========================================================================
    with tab8:
        st.subheader("🔄 Pruebas No Paramétricas")
        
        # Selección de prueba
        nonpar_test = st.radio(
            "Selecciona la prueba:",
            ["Mann-Whitney U", "Wilcoxon (pareado)", "Kruskal-Wallis", "Chi-cuadrado"],
            key="nonpar_test"
        )
        
        alpha_nonpar = st.slider(
            "Nivel de significancia (α):",
            0.01, 0.10, 0.05,
            key="nonpar_alpha"
        )
        
        # Mann-Whitney U
        if nonpar_test == "Mann-Whitney U" and numeric_cols and categorical_cols:
            st.subheader("Prueba de Mann-Whitney U")
            
            mw_num_var = st.selectbox("Variable numérica:", numeric_cols, key="mw_num_var")
            mw_cat_var = st.selectbox("Variable categórica (2 grupos):", categorical_cols, key="mw_cat_var")
            
            # Verificar grupos
            unique_groups = df[mw_cat_var].dropna().unique()
            if len(unique_groups) == 2:
                group1, group2 = unique_groups
                
                if st.button("📊 Ejecutar Mann-Whitney U", key="run_mannwhitney"):
                    try:
                        data1 = df[df[mw_cat_var] == group1][mw_num_var].dropna()
                        data2 = df[df[mw_cat_var] == group2][mw_num_var].dropna()
                        
                        if len(data1) < 3 or len(data2) < 3:
                            st.error("Cada grupo necesita al menos 3 observaciones")
                        else:
                            # Ejecutar prueba
                            u_stat, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                            
                            # Mostrar resultados
                            st.subheader("📋 Resultados")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Estadístico U", f"{u_stat:.4f}")
                            with col2:
                                st.metric("p-valor", f"{p_value:.4f}")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric(f"Mediana {group1}", f"{data1.median():.4f}")
                                st.metric(f"Rango IQ {group1}", 
                                        f"{data1.quantile(0.75) - data1.quantile(0.25):.4f}")
                            with col2:
                                st.metric(f"Mediana {group2}", f"{data2.median():.4f}")
                                st.metric(f"Rango IQ {group2}", 
                                        f"{data2.quantile(0.75) - data2.quantile(0.25):.4f}")
                            
                    except Exception as e:
                        st.error(f"Error en Mann-Whitney U: {e}")
            else:
                st.warning("La variable categórica debe tener exactamente 2 grupos")
        
        # Wilcoxon (pareado)
        elif nonpar_test == "Wilcoxon (pareado)" and len(numeric_cols) >= 2:
            st.subheader("Prueba de Wilcoxon para Muestras Pareadas")
            
            col1, col2 = st.columns(2)
            with col1:
                wilcoxon_var1 = st.selectbox("Variable 1:", numeric_cols, key="wilcoxon_var1")
            with col2:
                wilcoxon_var2 = st.selectbox("Variable 2:", numeric_cols, key="wilcoxon_var2")
            
            if st.button("📊 Ejecutar Wilcoxon", key="run_wilcoxon"):
                try:
                    # Filtrar pares completos
                    paired_data = df[[wilcoxon_var1, wilcoxon_var2]].dropna()
                    
                    if len(paired_data) < 3:
                        st.error("Se necesitan al menos 3 pares completos")
                    else:
                        # Ejecutar prueba
                        w_stat, p_value = stats.wilcoxon(
                            paired_data[wilcoxon_var1], 
                            paired_data[wilcoxon_var2]
                        )
                        
                        # Mostrar resultados
                        st.subheader("📋 Resultados")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Estadístico W", f"{w_stat:.4f}")
                        with col2:
                            st.metric("p-valor", f"{p_value:.4f}")
                        
                        # Diferencias
                        differences = paired_data[wilcoxon_var2] - paired_data[wilcoxon_var1]
                        st.metric("Mediana diferencias", f"{differences.median():.4f}")
                        
                except Exception as e:
                    st.error(f"Error en prueba de Wilcoxon: {e}")
        
        # Kruskal-Wallis
        elif nonpar_test == "Kruskal-Wallis" and numeric_cols and categorical_cols:
            st.subheader("Prueba de Kruskal-Wallis")
            
            kw_num_var = st.selectbox("Variable numérica:", numeric_cols, key="kw_num_var")
            kw_cat_var = st.selectbox("Variable categórica:", categorical_cols, key="kw_cat_var")
            
            if st.button("📊 Ejecutar Kruskal-Wallis", key="run_kruskal"):
                try:
                    # Preparar datos
                    groups_data = []
                    group_names = []
                    
                    for group in df[kw_cat_var].dropna().unique():
                        group_data = df[df[kw_cat_var] == group][kw_num_var].dropna()
                        if len(group_data) >= 3:
                            groups_data.append(group_data)
                            group_names.append(str(group))
                    
                    if len(groups_data) < 2:
                        st.error("Se necesitan al menos 2 grupos con datos válidos")
                    else:
                        # Ejecutar prueba
                        h_stat, p_value = stats.kruskal(*groups_data)
                        
                        # Mostrar resultados
                        st.subheader("📋 Resultados")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Estadístico H", f"{h_stat:.4f}")
                        with col2:
                            st.metric("p-valor", f"{p_value:.4f}")
                        
                except Exception as e:
                    st.error(f"Error en Kruskal-Wallis: {e}")
        
        # Chi-cuadrado
        elif nonpar_test == "Chi-cuadrado" and len(categorical_cols) >= 2:
            st.subheader("Prueba de Chi-cuadrado")
            
            chi_var1 = st.selectbox("Variable categórica 1:", categorical_cols, key="chi_var1")
            chi_var2 = st.selectbox("Variable categórica 2:", categorical_cols, key="chi_var2")
            
            if st.button("📊 Ejecutar Chi-cuadrado", key="run_chisquare"):
                try:
                    # Crear tabla de contingencia
                    contingency = pd.crosstab(df[chi_var1], df[chi_var2])
                    
                    if contingency.size == 0:
                        st.error("No hay datos suficientes")
                    else:
                        # Ejecutar prueba
                        chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency)
                        
                        # Mostrar resultados
                        st.subheader("📋 Resultados")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Estadístico χ²", f"{chi2_stat:.4f}")
                        with col2:
                            st.metric("p-valor", f"{p_value:.4f}")
                        with col3:
                            st.metric("Grados libertad", dof)
                        
                        # Mostrar tabla de contingencia
                        st.subheader("📊 Tabla de Contingencia")
                        st.dataframe(contingency)
                        
                except Exception as e:
                    st.error(f"Error en Chi-cuadrado: {e}")

# ============================================================================
# SECCIÓN FINAL: EXPORTACIÓN Y REPORTE
# ============================================================================

if df is not None:
    st.markdown("---")
    st.subheader("📤 Exportar Resultados")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📝 Generar Reporte Resumen", key="generate_summary_report"):
            with st.spinner("Generando reporte..."):
                try:
                    # Crear reporte básico
                    report_content = f"""
                    # Reporte de Análisis Estadístico
                    Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                    
                    ## Resumen del Dataset
                    - Filas: {df.shape[0]}
                    - Columnas: {df.shape[1]}
                    - Variables numéricas: {len(numeric_cols)}
                    - Variables categóricas: {len(categorical_cols)}
                    - Valores faltantes: {df.isnull().sum().sum()}
                    
                    ## Estadísticas Descriptivas
                    """
                    
                    if numeric_cols:
                        report_content += "\n" + df[numeric_cols].describe().to_string()
                    
                    # Crear archivo de texto
                    st.download_button(
                        label="📥 Descargar Reporte (TXT)",
                        data=report_content,
                        file_name=f"reporte_analisis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
                    
                    st.success("Reporte generado correctamente")
                    
                except Exception as e:
                    st.error(f"Error generando reporte: {e}")
    
    with col2:
        if st.button("💾 Exportar Datos Procesados", key="export_processed_data"):
            try:
                # Crear archivo Excel con múltiples hojas
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='Datos Originales', index=False)
                    
                    # Agregar estadísticas descriptivas
                    if numeric_cols:
                        df[numeric_cols].describe().to_excel(
                            writer, sheet_name='Estadísticas Descriptivas'
                        )
                
                st.download_button(
                    label="📥 Descargar Datos (Excel)",
                    data=output.getvalue(),
                    file_name=f"datos_procesados_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
            except Exception as e:
                st.error(f"Error exportando datos: {e}")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    """
    **Analizador Estadístico Universal** - Herramienta para análisis estadísticos descriptivos e inferenciales  
    Desarrollado con Streamlit, SciPy, Statsmodels y OpenAI GPT
    """
)

# Mensaje cuando no hay datos cargados
if df is None:
    st.info("👆 **Para comenzar:** Carga un archivo de datos en la barra lateral para acceder a todas las funciones de análisis.")
    
    # Ejemplos de datos de demostración
    with st.expander("💡 ¿Necesitas datos de ejemplo para probar?"):
        st.markdown("""
        Puedes usar estos conjuntos de datos de ejemplo para probar la aplicación:
        
        1. **Iris Dataset** (Clásico para clasificación)
           - Variables: sepal_length, sepal_width, petal_length, petal_width, species
           - URL: https://archive.ics.uci.edu/ml/datasets/iris
        
        2. **Titanic Dataset** (Supervivencia de pasajeros)
           - Variables: survived, pclass, sex, age, fare, embarked, etc.
           - URL: https://www.kaggle.com/c/titanic/data
        
        3. **Boston Housing** (Precios de viviendas)
           - Variables: crim, zn, indus, nox, rm, age, dis, tax, ptratio, lstat, medv
           - URL: https://www.kaggle.com/c/boston-housing
        
        4. **Wine Dataset** (Análisis de vinos)
           - Variables: alcohol, malic_acid, ash, alcalinity_of_ash, magnesium, etc.
           - URL: https://archive.ics.uci.edu/ml/datasets/wine
        """)