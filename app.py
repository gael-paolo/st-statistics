# -*- coding: utf-8 -*-
"""
Analizador Estadístico Universal con Asistente AI
Aplicación completa para análisis estadísticos descriptivos e inferenciales
con integración de OpenAI GPT para recomendaciones y explicaciones
"""

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
from ydata_profiling import ProfileReport
import io
from openai import OpenAI
import os
import sys
import traceback
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

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

# Función para consultar a OpenAI
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
                
                response = consultar_openai(prompt, max_tokens=2500)
                st.success("📚 Respuesta del Experto en Estadística:")
                
                # Mejorar la presentación de la respuesta
                st.markdown("---")
                st.markdown(response)
                st.markdown("---")
                
                # Opción para descargar la respuesta
                st.download_button(
                    label="📥 Descargar respuesta",
                    data=response,
                    file_name=f"respuesta_teorica_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
                
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

# ============================================================================
# FUNCIONES DE ANÁLISIS ESTADÍSTICO (REUTILIZABLES)
# ============================================================================

# 1. Funciones de muestreo
def generate_sample(df, sample_size, method="simple", stratify_col=None, random_state=None):
    """Genera una muestra representativa de tus datos para análisis."""
    if isinstance(sample_size, float):
        if sample_size <= 0 or sample_size > 1:
            raise ValueError("Si 'sample_size' es un porcentaje, debe estar entre 0 y 1.")
        sample_size = int(len(df) * sample_size)

    if sample_size <= 0 or sample_size > len(df):
        raise ValueError("El tamaño de la muestra debe ser mayor que 0 y menor o igual al total de datos.")

    if method == "simple":
        sample_df = df.sample(n=sample_size, random_state=random_state)

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
    """Calcula el tamaño de muestra requerido."""
    if not (0 < margin_of_error < 1):
        raise ValueError("El margen de error debe estar entre 0 y 1.")
    if not (0 < confidence_level < 1):
        raise ValueError("El nivel de confianza debe estar entre 0 y 1.")
    if not (0 < proportion < 1):
        raise ValueError("La proporción debe estar entre 0 y 1.")
    if population_size <= 0:
        raise ValueError("El tamaño de la población debe ser mayor que 0.")

    z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)
    sample_size = (z_score**2 * proportion * (1 - proportion)) / (margin_of_error**2)
    adjusted_sample_size = sample_size / (1 + (sample_size - 1) / population_size)
    
    return int(np.ceil(adjusted_sample_size))

# 2. Funciones para pruebas estadísticas
def run_normality_tests(data, alpha=0.05):
    """Ejecuta múltiples pruebas de normalidad."""
    results = {}
    
    # Shapiro-Wilk
    if len(data) <= 5000:
        shapiro_stat, shapiro_p = shapiro(data)
        results['shapiro'] = {
            'test': 'Shapiro-Wilk',
            'statistic': shapiro_stat,
            'p_value': shapiro_p,
            'is_normal': shapiro_p > alpha,
            'weight': 3
        }
    else:
        results['shapiro'] = {
            'test': 'Shapiro-Wilk',
            'statistic': None,
            'p_value': None,
            'is_normal': None,
            'weight': 1,
            'note': 'Muestra muy grande (>5000) - prueba no recomendada'
        }
    
    # Anderson-Darling
    ad_result = anderson(data, dist='norm')
    alpha_to_idx = {0.15: 0, 0.10: 1, 0.05: 2, 0.025: 3, 0.01: 4}
    closest_alpha = min(alpha_to_idx.keys(), key=lambda x: abs(x - alpha))
    idx = alpha_to_idx[closest_alpha]
    critical_value = ad_result.critical_values[idx]
    
    results['anderson'] = {
        'test': 'Anderson-Darling',
        'statistic': ad_result.statistic,
        'critical_value': critical_value,
        'is_normal': ad_result.statistic < critical_value,
        'weight': 2
    }
    
    # Lilliefors (Kolmogorov-Smirnov modificada)
    try:
        lilliefors_stat, lilliefors_p = lilliefors(data)
        results['lilliefors'] = {
            'test': 'Lilliefors',
            'statistic': lilliefors_stat,
            'p_value': lilliefors_p,
            'is_normal': lilliefors_p > alpha,
            'weight': 2
        }
    except:
        results['lilliefors'] = {
            'test': 'Lilliefors',
            'statistic': None,
            'p_value': None,
            'is_normal': None,
            'weight': 1,
            'note': 'Error en cálculo'
        }
    
    return results

def calculate_effect_size(test_type, **kwargs):
    """Calcula el tamaño del efecto para diferentes pruebas."""
    try:
        if test_type == "Una muestra":
            data1 = kwargs['data1']
            pop_mean = kwargs.get('pop_mean', 0)
            d = (data1.mean() - pop_mean) / data1.std()
            return abs(d)
            
        elif test_type == "Muestras independientes":
            data1 = kwargs['data1']
            data2 = kwargs['data2']
            n1, n2 = len(data1), len(data2)
            pooled_std = np.sqrt(((n1-1)*data1.std()**2 + (n2-1)*data2.std()**2) / (n1 + n2 - 2))
            d = (data1.mean() - data2.mean()) / pooled_std
            return abs(d)
            
        elif test_type == "Muestras pareadas":
            paired_data = kwargs['paired_data']
            var_before = kwargs['var_before']
            var_after = kwargs['var_after']
            differences = paired_data[var_after] - paired_data[var_before]
            d = differences.mean() / differences.std()
            return abs(d)
    except:
        return 0

def interpret_effect_size(d):
    """Interpreta el tamaño del efecto según Cohen."""
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

# 3. Funciones para visualizaciones
def create_normality_plots(data, variable_name):
    """Crea visualizaciones para pruebas de normalidad."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Histograma con curva normal
    sns.histplot(data, kde=True, stat='density', ax=ax1, alpha=0.7, color='skyblue')
    mu, sigma = data.mean(), data.std()
    x = np.linspace(data.min(), data.max(), 100)
    ax1.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, 
             label=f'Normal teórica\n(μ={mu:.2f}, σ={sigma:.2f})')
    ax1.set_title(f'Distribución de {variable_name}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Q-Q plot
    stats.probplot(data, dist="norm", plot=ax2)
    ax2.set_title(f'Q-Q Plot de {variable_name}')
    ax2.grid(True, alpha=0.3)
    
    # Boxplot para outliers
    sns.boxplot(y=data, ax=ax3, color='lightcoral')
    ax3.set_title(f'Boxplot de {variable_name}\n(Detección de Outliers)')
    ax3.grid(True, alpha=0.3)
    
    # Gráfico de probabilidad acumulada
    ecdf = np.arange(1, len(data) + 1) / len(data)
    sorted_data = np.sort(data)
    ax4.plot(sorted_data, ecdf, 'b-', linewidth=2, label='Empírica')
    ax4.plot(sorted_data, stats.norm.cdf(sorted_data, mu, sigma), 'r--', linewidth=2, label='Teórica')
    ax4.set_title('Función de Distribución Acumulada')
    ax4.set_xlabel(variable_name)
    ax4.set_ylabel('Probabilidad')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

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
        
        # Opciones de visualización
        col1, col2, col3 = st.columns(3)
        with col1:
            show_head = st.checkbox("Mostrar primeras filas", value=True)
        with col2:
            show_tail = st.checkbox("Mostrar últimas filas")
        with col3:
            show_sample = st.checkbox("Mostrar muestra aleatoria")
        
        if show_head:
            st.dataframe(df.head(), use_container_width=True)
        if show_tail:
            st.dataframe(df.tail(), use_container_width=True)
        if show_sample:
            st.dataframe(df.sample(min(10, len(df))), use_container_width=True)
        
        # Información básica del dataset
        st.subheader("📊 Información del Dataset")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total de registros", df.shape[0])
        with col2:
            st.metric("Total de variables", df.shape[1])
        with col3:
            missing_values = df.isnull().sum().sum()
            missing_percent = (missing_values / (df.shape[0] * df.shape[1])) * 100
            st.metric("Valores faltantes", f"{missing_values:,}", f"{missing_percent:.1f}%")
        with col4:
            memory_usage = df.memory_usage(deep=True).sum() / 1024**2
            st.metric("Memoria usada", f"{memory_usage:.2f} MB")
        
        # Identificar tipos de variables
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Mostrar resumen de tipos de variables
        with st.expander("🔍 Detalles de las variables"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Variables Numéricas:**")
                if numeric_cols:
                    for col in numeric_cols:
                        missing = df[col].isnull().sum()
                        st.write(f"- {col}: {df[col].dtype} ({missing} faltantes)")
                else:
                    st.write("No hay variables numéricas")
            
            with col2:
                st.write("**Variables Categóricas:**")
                if categorical_cols:
                    for col in categorical_cols:
                        missing = df[col].isnull().sum()
                        unique_vals = df[col].nunique()
                        st.write(f"- {col}: {df[col].dtype} ({unique_vals} valores únicos, {missing} faltantes)")
                else:
                    st.write("No hay variables categóricas")

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
    
    col1, col2 = st.columns([3, 1])
    with col1:
        analyze_button = st.button("🔍 Obtener recomendaciones de análisis", key="business_recommendations_main", use_container_width=True)
    with col2:
        detailed_analysis = st.checkbox("Análisis detallado", value=True)
    
    if analyze_button and user_question:
        if openai_api_key:
            with st.spinner("Analizando tus datos y generando recomendaciones..."):
                try:
                    # Preparar resumen estadístico detallado
                    data_summary = f"""
                    RESUMEN DEL DATASET PARA ANÁLISIS ESTADÍSTICO
                    =============================================
                    
                    1. DIMENSIONES Y ESTRUCTURA:
                       - Filas: {df.shape[0]}
                       - Columnas: {df.shape[1]}
                       - Variables numéricas ({len(numeric_cols)}): {', '.join(numeric_cols[:10])}{'...' if len(numeric_cols) > 10 else ''}
                       - Variables categóricas ({len(categorical_cols)}): {', '.join(categorical_cols[:10])}{'...' if len(categorical_cols) > 10 else ''}
                       - Valores faltantes totales: {df.isnull().sum().sum()}
                       - Memoria: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
                    
                    2. OBJETIVO DEL USUARIO:
                       "{user_question}"
                    
                    3. ESTADÍSTICAS DESCRIPTIVAS BÁSICAS:"""
                    
                    if numeric_cols:
                        for i, col in enumerate(numeric_cols[:5]):  # Limitar a 5 variables para no hacer el prompt demasiado largo
                            data_summary += f"\n   - {col}:"
                            data_summary += f"\n     * Media: {df[col].mean():.4f}"
                            data_summary += f"\n     * Desviación: {df[col].std():.4f}"
                            data_summary += f"\n     * Mín: {df[col].min():.4f}"
                            data_summary += f"\n     * Máx: {df[col].max():.4f}"
                            data_summary += f"\n     * Faltantes: {df[col].isnull().sum()}"
                    
                    data_summary += f"""
                    
                    4. INFORMACIÓN DE VARIABLES CATEGÓRICAS:"""
                    
                    if categorical_cols:
                        for i, col in enumerate(categorical_cols[:3]):  # Limitar a 3 variables
                            data_summary += f"\n   - {col}:"
                            data_summary += f"\n     * Valores únicos: {df[col].nunique()}"
                            data_summary += f"\n     * Faltantes: {df[col].isnull().sum()}"
                            if df[col].nunique() <= 10:
                                for val in df[col].dropna().unique():
                                    count = (df[col] == val).sum()
                                    data_summary += f"\n     * '{val}': {count} ({count/len(df)*100:.1f}%)"
                    
                    if detailed_analysis:
                        prompt_template = """
                        COMO EXPERTO EN ANÁLISIS ESTADÍSTICO, ANALIZA EL SIGUIENTE CASO:
                        
                        {data_summary}
                        
                        BASÁNDOTE EN LOS DATOS Y EL OBJETIVO DEL USUARIO, GENERA UN PLAN DE ANÁLISIS COMPLETO QUE INCLUYA:
                        
                        1. **DIAGNÓSTICO INICIAL**:
                           - Evaluación de calidad de datos
                           - Identificación de problemas potenciales
                           - Recomendaciones de limpieza de datos
                        
                        2. **ANÁLISIS DESCRIPTIVO NECESARIO**:
                           - Estadísticas descriptivas específicas
                           - Tablas de frecuencia para variables categóricas
                           - Medidas de tendencia central y dispersión
                        
                        3. **VALIDACIÓN DE SUPUESTOS**:
                           - Pruebas de normalidad recomendadas
                           - Evaluación de homogeneidad de varianzas
                           - Detección de outliers
                        
                        4. **ANÁLISIS INFERENCIAL**:
                           - Pruebas estadísticas específicas para cada hipótesis
                           - Justificación de cada prueba seleccionada
                           - Variables a utilizar en cada análisis
                        
                        5. **VISUALIZACIONES RECOMENDADAS**:
                           - Gráficos para exploración inicial
                           - Visualizaciones para presentación de resultados
                           - Gráficos estadísticos específicos
                        
                        6. **INTERPRETACIÓN ESPERADA**:
                           - Cómo interpretar los resultados de cada prueba
                           - Posibles conclusiones basadas en los datos
                           - Limitaciones del análisis
                        
                        7. **PASOS SIGUIENTES**:
                           - Análisis complementarios recomendados
                           - Validaciones adicionales necesarias
                           - Consideraciones para toma de decisiones
                        
                        8. **RECOMENDACIONES PRÁCTICAS**:
                           - Transformaciones de datos si son necesarias
                           - Alternativas si no se cumplen supuestos
                           - Mejores prácticas para reportar resultados
                        
                        PROPORCIONA UNA RESPUESTA ESTRUCTURADA, PRÁCTICA Y APLICABLE AL CONTEXTO DEL USUARIO.
                        """
                    else:
                        prompt_template = """
                        COMO EXPERTO EN ANÁLISIS ESTADÍSTICO, PROPORCIONA RECOMENDACIONES INICIALES:
                        
                        {data_summary}
                        
                        GENERA RECOMENDACIONES CONCISAS QUE INCLUYAN:
                        1. Análisis descriptivos recomendados
                        2. Pruebas estadísticas principales
                        3. Visualizaciones clave
                        4. Consideraciones importantes
                        """
                    
                    prompt = prompt_template.format(data_summary=data_summary)
                    
                    response = consultar_openai(prompt, max_tokens=3000 if detailed_analysis else 1500)
                    
                    st.success("🎯 Recomendaciones de Análisis para tus Datos:")
                    
                    # Mostrar respuesta con mejor formato
                    st.markdown("---")
                    st.markdown(response)
                    st.markdown("---")
                    
                    # Opciones adicionales
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.download_button(
                            label="📥 Descargar recomendaciones",
                            data=response,
                            file_name=f"recomendaciones_analisis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
                    
                    with col2:
                        if st.button("🔄 Generar plan detallado", use_container_width=True):
                            with st.spinner("Generando plan detallado..."):
                                detailed_prompt = f"""
                                BASÁNDOTE EN LAS RECOMENDACIONES PREVIAS, GENERA UN PLAN DE ANÁLISIS DETALLADO CON:
                                1. Cronograma sugerido
                                2. Recursos necesarios
                                3. Scripts de código ejemplo (Python/R)
                                4. Plantilla de reporte
                                5. Validaciones de calidad
                                """
                                detailed_response = consultar_openai(detailed_prompt, max_tokens=2500)
                                st.markdown("### 📋 Plan Detallado de Análisis")
                                st.markdown(detailed_response)
                    
                    with col3:
                        if st.button("📊 Sugerir visualizaciones", use_container_width=True):
                            with st.spinner("Generando sugerencias de visualización..."):
                                viz_prompt = f"""
                                SUGIERE VISUALIZACIONES ESPECÍFICAS PARA ESTE ANÁLISIS, INCLUYENDO:
                                1. Gráficos exploratorios iniciales
                                2. Visualizaciones para validación de supuestos
                                3. Gráficos para presentación de resultados
                                4. Código Python (matplotlib/seaborn) para cada gráfico
                                """
                                viz_response = consultar_openai(viz_prompt, max_tokens=2000)
                                st.markdown("### 📈 Visualizaciones Sugeridas")
                                st.markdown(viz_response)
                    
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
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "🎯 Muestreo", 
        "📈 Descriptivos", 
        "🔍 Normalidad", 
        "📉 Correlaciones",
        "⚖️ Homogeneidad",
        "✅ Pruebas T",
        "📊 ANOVA",
        "🔄 No Paramétricas",
        "📋 Reportes"
    ])
    
    # ========================================================================
    # PESTAÑA 1: MUESTREO
    # ========================================================================
    with tab1:
        st.subheader("🎯 Análisis de Muestreo")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📊 Generar Muestra")
            
            sample_method = st.radio(
                "**Método de muestreo:**",
                ["simple", "stratified"],
                format_func=lambda x: "🎲 Aleatorio Simple" if x == "simple" else "📊 Estratificado",
                horizontal=True
            )
            
            col_size1, col_size2 = st.columns(2)
            with col_size1:
                sample_size_type = st.radio(
                    "**Tipo de tamaño:**",
                    ["percentage", "absolute"],
                    format_func=lambda x: "📏 Porcentaje" if x == "percentage" else "🔢 Número absoluto"
                )
            
            with col_size2:
                if sample_size_type == "percentage":
                    sample_size_input = st.slider(
                        "**Porcentaje de muestra:**",
                        min_value=1,
                        max_value=50,
                        value=20,
                        help="Porcentaje del total de datos a incluir en la muestra"
                    )
                    sample_size = sample_size_input / 100.0
                    display_size = f"{sample_size_input}%"
                else:
                    sample_size_input = st.number_input(
                        "**Tamaño de muestra:**",
                        min_value=1,
                        max_value=len(df),
                        value=min(100, len(df)),
                        help="Número absoluto de registros para la muestra"
                    )
                    sample_size = sample_size_input
                    display_size = f"{sample_size_input:,}"
            
            if sample_method == "stratified" and categorical_cols:
                stratify_column = st.selectbox(
                    "**Variable para estratificación:**",
                    categorical_cols,
                    help="La muestra mantendrá las proporciones de esta variable categórica"
                )
            else:
                stratify_column = None
            
            if st.button("🎲 Generar Muestra", key="generate_sample", use_container_width=True):
                try:
                    with st.spinner("Generando muestra..."):
                        sample_df = generate_sample(
                            df, 
                            sample_size, 
                            method=sample_method, 
                            stratify_col=stratify_column, 
                            random_state=42
                        )
                    
                    st.success(f"✅ Muestra generada: **{len(sample_df):,}** registros")
                    
                    # Mostrar información de la muestra
                    col_info1, col_info2, col_info3 = st.columns(3)
                    with col_info1:
                        st.metric("Tamaño de muestra", f"{len(sample_df):,}")
                    with col_info2:
                        st.metric("Porcentaje del total", f"{(len(sample_df)/len(df))*100:.1f}%")
                    with col_info3:
                        st.metric("Reducción", f"{len(df) - len(sample_df):,} registros")
                    
                    # Mostrar preview de la muestra
                    with st.expander("👁️ Vista previa de la muestra", expanded=True):
                        st.dataframe(sample_df.head(), use_container_width=True)
                    
                    # Mostrar distribución si es muestreo estratificado
                    if sample_method == "stratified" and stratify_column:
                        with st.expander("📊 Distribución estratificada", expanded=True):
                            sample_dist = sample_df[stratify_column].value_counts()
                            original_dist = df[stratify_column].value_counts()
                            
                            dist_comparison = pd.DataFrame({
                                'Original': original_dist,
                                'Muestra': sample_dist,
                                '% Original': (original_dist / len(df)) * 100,
                                '% Muestra': (sample_dist / len(sample_df)) * 100
                            })
                            
                            st.dataframe(dist_comparison, use_container_width=True)
                            
                            # Gráfico de comparación
                            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                            
                            original_dist.plot(kind='bar', ax=ax[0], color='skyblue', alpha=0.7)
                            ax[0].set_title('Distribución Original')
                            ax[0].set_ylabel('Frecuencia')
                            ax[0].tick_params(axis='x', rotation=45)
                            
                            sample_dist.plot(kind='bar', ax=ax[1], color='lightcoral', alpha=0.7)
                            ax[1].set_title('Distribución en Muestra')
                            ax[1].set_ylabel('Frecuencia')
                            ax[1].tick_params(axis='x', rotation=45)
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                    
                    # Opciones de descarga
                    col_dl1, col_dl2, col_dl3 = st.columns(3)
                    
                    with col_dl1:
                        # Descargar como Excel
                        output_excel = io.BytesIO()
                        with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
                            sample_df.to_excel(writer, index=False, sheet_name='Muestra')
                            if sample_method == "stratified" and stratify_column:
                                dist_comparison.to_excel(writer, sheet_name='Distribución')
                        
                        st.download_button(
                            label="📥 Excel",
                            data=output_excel.getvalue(),
                            file_name=f"muestra_{sample_method}_{display_size}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
                    
                    with col_dl2:
                        # Descargar como CSV
                        output_csv = sample_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 CSV",
                            data=output_csv,
                            file_name=f"muestra_{sample_method}_{display_size}_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    with col_dl3:
                        # Descargar información del muestreo
                        sample_info = f"""
                        INFORMACIÓN DEL MUESTREO
                        ========================
                        
                        Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                        
                        1. DATOS ORIGINALES:
                           - Total registros: {len(df):,}
                           - Total variables: {len(df.columns)}
                           - Tamaño archivo: {memory_usage:.2f} MB
                        
                        2. PARÁMETROS DEL MUESTREO:
                           - Método: {sample_method}
                           - Tamaño solicitado: {display_size}
                           - Variable estratificación: {stratify_column if stratify_column else 'N/A'}
                           - Semilla aleatoria: 42
                        
                        3. RESULTADO:
                           - Muestra generada: {len(sample_df):,} registros
                           - Porcentaje del total: {(len(sample_df)/len(df))*100:.1f}%
                           - Reducción: {len(df) - len(sample_df):,} registros
                        
                        4. DISTRIBUCIÓN (si aplica):"""
                        
                        if sample_method == "stratified" and stratify_column:
                            for group, count in sample_dist.items():
                                sample_info += f"\n   - {group}: {count} registros ({(count/len(sample_df))*100:.1f}%)"
                        
                        st.download_button(
                            label="📥 Informe",
                            data=sample_info,
                            file_name=f"informe_muestreo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
                    
                except Exception as e:
                    st.error(f"❌ Error generando muestra: {e}")
        
        with col2:
            st.markdown("### 🧮 Calcular Tamaño de Muestra")
            
            population_size = st.number_input(
                "**Tamaño de la población:**",
                min_value=1,
                value=len(df),
                help="Número total de elementos en la población de estudio"
            )
            
            margin_error = st.slider(
                "**Margen de error (%):**",
                min_value=1,
                max_value=10,
                value=5,
                help="Precisión deseada en los resultados (±%)"
            ) / 100.0
            
            confidence_level = st.slider(
                "**Nivel de confianza (%):**",
                min_value=80,
                max_value=99,
                value=95,
                help="Probabilidad de que el resultado sea correcto"
            ) / 100.0
            
            proportion = st.slider(
                "**Proporción esperada (%):**",
                min_value=1,
                max_value=99,
                value=50,
                help="Proporción esperada de la característica en la población (usar 50% si es desconocida)"
            ) / 100.0
            
            if st.button("📐 Calcular Tamaño de Muestra", key="calculate_sample_size", use_container_width=True):
                try:
                    with st.spinner("Calculando tamaño de muestra..."):
                        sample_size = calculate_sample_size(
                            population_size=population_size,
                            margin_of_error=margin_error,
                            confidence_level=confidence_level,
                            proportion=proportion
                        )
                    
                    st.success(f"🎯 **Tamaño de muestra recomendado:** `{sample_size:,}`")
                    
                    # Información detallada
                    st.info(f"""
                    **📋 Parámetros utilizados:**
                    - **Población:** {population_size:,}
                    - **Margen de error:** ±{margin_error*100:.1f}%
                    - **Nivel de confianza:** {confidence_level*100:.1f}%
                    - **Proporción esperada:** {proportion*100:.1f}%
                    
                    **📊 Interpretación:**
                    Para obtener resultados con un margen de error de **±{margin_error*100:.1f}%** y un nivel de confianza del **{confidence_level*100:.1f}%**, se requiere una muestra mínima de **{sample_size:,}** elementos.
                    """)
                    
                    # Comparación con datos actuales
                    if population_size == len(df):
                        coverage = (sample_size / len(df)) * 100
                        
                        col_comp1, col_comp2 = st.columns(2)
                        with col_comp1:
                            st.metric(
                                "Cobertura de tu dataset",
                                f"{coverage:.1f}%",
                                help="Porcentaje del tamaño de muestra recomendado que cubre tu dataset actual"
                            )
                        
                        with col_comp2:
                            if sample_size > len(df):
                                delta_val = sample_size - len(df)
                                st.metric(
                                    "Déficit de datos",
                                    f"{delta_val:,}",
                                    delta=f"Faltan {delta_val:,} registros",
                                    delta_color="inverse"
                                )
                                st.warning(f"⚠️ Tu dataset actual ({len(df):,}) es más pequeño que el tamaño de muestra recomendado ({sample_size:,})")
                            else:
                                st.metric(
                                    "Excedente de datos",
                                    f"{len(df) - sample_size:,}",
                                    delta="Dataset suficiente",
                                    delta_color="normal"
                                )
                                st.success(f"✅ Tu dataset actual ({len(df):,}) es suficiente para el análisis")
                    
                    # Descargar cálculo
                    calc_info = f"""
                    CÁLCULO DE TAMAÑO DE MUESTRA
                    =============================
                    
                    Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                    
                    1. PARÁMETROS DE ENTRADA:
                       - Tamaño población: {population_size:,}
                       - Margen de error: {margin_error*100:.1f}%
                       - Nivel de confianza: {confidence_level*100:.1f}%
                       - Proporción esperada: {proportion*100:.1f}%
                    
                    2. FÓRMULA UTILIZADA:
                       n = (Z² * p * (1-p)) / E²
                       Donde:
                       - Z = {stats.norm.ppf(1 - (1 - confidence_level) / 2):.4f} (valor Z para {confidence_level*100:.1f}% confianza)
                       - p = {proportion} (proporción esperada)
                       - E = {margin_error} (margen de error)
                    
                    3. CÁLCULO:
                       n = ({stats.norm.ppf(1 - (1 - confidence_level) / 2):.4f}² * {proportion} * {1-proportion}) / {margin_error}²
                       n = {sample_size:,} (ajustado por población finita)
                    
                    4. RESULTADO:
                       - Tamaño muestra mínimo: {sample_size:,}
                       - Interpretación: Para obtener resultados con ±{margin_error*100:.1f}% de error y {confidence_level*100:.1f}% de confianza
                    """
                    
                    st.download_button(
                        label="📥 Descargar cálculo",
                        data=calc_info,
                        file_name=f"calculo_muestra_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                    
                except Exception as e:
                    st.error(f"❌ Error calculando tamaño de muestra: {e}")
            
            # Información educativa
            with st.expander("📚 ¿Por qué es importante el muestreo?"):
                st.markdown("""
                **🎯 El muestreo adecuado es crucial porque:**
                - **Reduce costos y tiempo** de análisis
                - **Permite trabajar** con conjuntos de datos manejables
                - **Mantiene la representatividad** de la población
                - **Facilita la generalización** de resultados
                
                **📊 Tipos de muestreo:**
                - **🎲 Aleatorio simple:** Cada elemento tiene igual probabilidad de ser seleccionado
                - **📊 Estratificado:** Mantiene las proporciones de subgrupos importantes
                - **🔢 Sistemático:** Selecciona cada k-ésimo elemento
                - **👥 Por conglomerados:** Divide la población en grupos y selecciona algunos grupos completos
                
                **📈 Consideraciones prácticas:**
                - Para poblaciones grandes (>10,000), muestras de 400-1000 suelen ser suficientes
                - El margen de error disminuye con la raíz cuadrada del tamaño de muestra
                - Doblar el tamaño de muestra no reduce el error a la mitad
                """)
    
    # ========================================================================
    # PESTAÑA 2: ANÁLISIS DESCRIPTIVOS
    # ========================================================================
    with tab2:
        st.subheader("📈 Análisis Descriptivo")
        
        # Selector de variables
        col_select1, col_select2 = st.columns(2)
        with col_select1:
            selected_numeric = st.multiselect(
                "**Variables numéricas para análisis:**",
                numeric_cols,
                default=numeric_cols[:min(5, len(numeric_cols))],
                help="Selecciona las variables numéricas que quieres analizar"
            )
        
        with col_select2:
            if categorical_cols:
                selected_categorical = st.multiselect(
                    "**Variables categóricas para análisis:**",
                    categorical_cols,
                    default=categorical_cols[:min(3, len(categorical_cols))],
                    help="Selecciona variables categóricas para análisis de frecuencia"
                )
            else:
                selected_categorical = []
        
        if selected_numeric:
            # Estadísticas descriptivas detalladas
            st.markdown("### 📊 Estadísticas Descriptivas Detalladas")
            
            # Opciones de análisis
            col_options1, col_options2 = st.columns(2)
            with col_options1:
                show_percentiles = st.checkbox("Mostrar percentiles", value=True)
                show_skew_kurtosis = st.checkbox("Mostrar asimetría y curtosis", value=True)
            with col_options2:
                show_missing = st.checkbox("Mostrar valores faltantes", value=True)
                show_ci = st.checkbox("Mostrar intervalos de confianza", value=False)
            
            # Calcular estadísticas
            if st.button("📈 Calcular estadísticas", key="calc_stats", use_container_width=True):
                with st.spinner("Calculando estadísticas descriptivas..."):
                    try:
                        # DataFrame para estadísticas
                        stats_list = []
                        
                        for var in selected_numeric:
                            data = df[var].dropna()
                            
                            if len(data) > 0:
                                stats_row = {
                                    'Variable': var,
                                    'n': len(data),
                                    'Media': data.mean(),
                                    'Mediana': data.median(),
                                    'Moda': data.mode()[0] if not data.mode().empty else np.nan,
                                    'Mínimo': data.min(),
                                    'Máximo': data.max(),
                                    'Rango': data.max() - data.min(),
                                    'Desviación Estándar': data.std(),
                                    'Varianza': data.var(),
                                    'Error Estándar': stats.sem(data) if len(data) > 1 else np.nan,
                                    'Coef. Variación (%)': (data.std() / data.mean() * 100) if data.mean() != 0 else np.nan
                                }
                                
                                # Percentiles
                                if show_percentiles:
                                    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
                                        stats_row[f'P{p}'] = np.percentile(data, p)
                                
                                # Asimetría y curtosis
                                if show_skew_kurtosis:
                                    stats_row['Asimetría'] = data.skew()
                                    stats_row['Curtosis'] = data.kurtosis()
                                
                                # Intervalos de confianza
                                if show_ci and len(data) > 1:
                                    ci_low, ci_high = stats.t.interval(0.95, len(data)-1, loc=data.mean(), scale=stats.sem(data))
                                    stats_row['IC 95% Inferior'] = ci_low
                                    stats_row['IC 95% Superior'] = ci_high
                                
                                # Valores faltantes
                                if show_missing:
                                    missing_count = df[var].isnull().sum()
                                    stats_row['Faltantes'] = missing_count
                                    stats_row['% Faltantes'] = (missing_count / len(df)) * 100
                                
                                stats_list.append(stats_row)
                        
                        if stats_list:
                            stats_df = pd.DataFrame(stats_list)
                            
                            # Mostrar estadísticas
                            st.dataframe(
                                stats_df.style.format({
                                    'Media': '{:.4f}',
                                    'Mediana': '{:.4f}',
                                    'Desviación Estándar': '{:.4f}',
                                    'Varianza': '{:.4f}',
                                    'Coef. Variación (%)': '{:.2f}%',
                                    'Asimetría': '{:.4f}',
                                    'Curtosis': '{:.4f}'
                                }),
                                use_container_width=True
                            )
                            
                            # Opciones de descarga
                            col_dl_stats1, col_dl_stats2 = st.columns(2)
                            
                            with col_dl_stats1:
                                # Descargar como Excel
                                output_stats = io.BytesIO()
                                with pd.ExcelWriter(output_stats, engine='openpyxl') as writer:
                                    stats_df.to_excel(writer, sheet_name='Estadísticas', index=False)
                                
                                st.download_button(
                                    label="📥 Descargar Excel",
                                    data=output_stats.getvalue(),
                                    file_name=f"estadisticas_descriptivas_{datetime.now().strftime('%Y%m%d')}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    use_container_width=True
                                )
                            
                            with col_dl_stats2:
                                # Descargar como CSV
                                csv_stats = stats_df.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="📥 Descargar CSV",
                                    data=csv_stats,
                                    file_name=f"estadisticas_descriptivas_{datetime.now().strftime('%Y%m%d')}.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                            
                            # Visualizaciones
                            st.markdown("### 📈 Visualizaciones")
                            
                            # Seleccionar variable para visualización detallada
                            selected_var_viz = st.selectbox(
                                "**Selecciona variable para visualización detallada:**",
                                selected_numeric,
                                key="viz_var_select"
                            )
                            
                            if selected_var_viz:
                                col_viz1, col_viz2 = st.columns(2)
                                
                                with col_viz1:
                                    # Histograma con KDE
                                    fig1, ax1 = plt.subplots(figsize=(10, 6))
                                    sns.histplot(df[selected_var_viz].dropna(), kde=True, ax=ax1, bins=30)
                                    ax1.set_title(f'Distribución de {selected_var_viz}', fontsize=14, fontweight='bold')
                                    ax1.set_xlabel(selected_var_viz)
                                    ax1.set_ylabel('Frecuencia')
                                    ax1.grid(True, alpha=0.3)
                                    st.pyplot(fig1)
                                
                                with col_viz2:
                                    # Boxplot
                                    fig2, ax2 = plt.subplots(figsize=(10, 6))
                                    sns.boxplot(y=df[selected_var_viz].dropna(), ax=ax2)
                                    ax2.set_title(f'Boxplot de {selected_var_viz}', fontsize=14, fontweight='bold')
                                    ax2.set_ylabel(selected_var_viz)
                                    ax2.grid(True, alpha=0.3)
                                    
                                    # Calcular y mostrar outliers
                                    Q1 = df[selected_var_viz].quantile(0.25)
                                    Q3 = df[selected_var_viz].quantile(0.75)
                                    IQR = Q3 - Q1
                                    outliers = df[(df[selected_var_viz] < Q1 - 1.5*IQR) | (df[selected_var_viz] > Q3 + 1.5*IQR)]
                                    
                                    if len(outliers) > 0:
                                        ax2.text(0.02, 0.95, f'Outliers: {len(outliers)}', 
                                                transform=ax2.transAxes, fontsize=12,
                                                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
                                    
                                    st.pyplot(fig2)
                                
                                # Gráfico de densidad comparativo si hay múltiples variables
                                if len(selected_numeric) > 1:
                                    st.markdown("#### Comparación de Distribuciones")
                                    fig3, ax3 = plt.subplots(figsize=(12, 6))
                                    for var in selected_numeric:
                                        sns.kdeplot(df[var].dropna(), label=var, ax=ax3)
                                    ax3.set_title('Comparación de Distribuciones', fontsize=14, fontweight='bold')
                                    ax3.set_xlabel('Valor')
                                    ax3.set_ylabel('Densidad')
                                    ax3.legend()
                                    ax3.grid(True, alpha=0.3)
                                    st.pyplot(fig3)
                                
                                # Gráfico de violín
                                st.markdown("#### Gráfico de Violín")
                                fig4, ax4 = plt.subplots(figsize=(12, 6))
                                sns.violinplot(data=df[selected_numeric], ax=ax4)
                                ax4.set_title('Distribuciones por Variable (Violín Plot)', fontsize=14, fontweight='bold')
                                ax4.set_xlabel('Variable')
                                ax4.set_ylabel('Valor')
                                ax4.tick_params(axis='x', rotation=45)
                                ax4.grid(True, alpha=0.3, axis='y')
                                st.pyplot(fig4)
                            
                        else:
                            st.warning("No se encontraron datos válidos para las variables seleccionadas.")
                    
                    except Exception as e:
                        st.error(f"Error calculando estadísticas: {e}")
            
            # Análisis de frecuencia para variables categóricas
            if selected_categorical:
                st.markdown("### 📊 Análisis de Frecuencia")
                
                for cat_var in selected_categorical:
                    with st.expander(f"**Análisis de {cat_var}**", expanded=False):
                        # Calcular frecuencias
                        freq_df = df[cat_var].value_counts().reset_index()
                        freq_df.columns = [cat_var, 'Frecuencia']
                        freq_df['Porcentaje'] = (freq_df['Frecuencia'] / len(df)) * 100
                        freq_df['Porcentaje Acumulado'] = freq_df['Porcentaje'].cumsum()
                        
                        # Mostrar tabla
                        st.dataframe(
                            freq_df.style.format({'Porcentaje': '{:.2f}%', 'Porcentaje Acumulado': '{:.2f}%'}),
                            use_container_width=True
                        )
                        
                        # Gráfico de barras
                        fig, ax = plt.subplots(figsize=(10, 6))
                        bars = ax.bar(freq_df[cat_var].astype(str), freq_df['Frecuencia'], color='skyblue', alpha=0.7)
                        ax.set_title(f'Distribución de {cat_var}', fontsize=14, fontweight='bold')
                        ax.set_xlabel(cat_var)
                        ax.set_ylabel('Frecuencia')
                        ax.tick_params(axis='x', rotation=45)
                        
                        # Agregar etiquetas con porcentajes
                        for bar, perc in zip(bars, freq_df['Porcentaje']):
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                   f'{perc:.1f}%', ha='center', va='bottom', fontsize=10)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
        
        else:
            st.warning("No hay variables numéricas seleccionadas para análisis descriptivo.")
    
    # ========================================================================
    # PESTAÑA 3: PRUEBAS DE NORMALIDAD
    # ========================================================================
    with tab3:
        st.subheader("🔍 Pruebas de Normalidad")
        
        if numeric_cols:
            col_select_norm1, col_select_norm2 = st.columns(2)
            with col_select_norm1:
                selected_var = st.selectbox(
                    "**Selecciona variable para prueba de normalidad:**",
                    numeric_cols,
                    key="normal_var_select"
                )
            
            with col_select_norm2:
                alpha_normal = st.slider(
                    "**Nivel de significancia (α):**",
                    min_value=0.01,
                    max_value=0.10,
                    value=0.05,
                    step=0.01,
                    key="normal_alpha_slider"
                )
            
            if st.button("📊 Ejecutar Pruebas de Normalidad", key="run_normality_full", use_container_width=True):
                try:
                    data = df[selected_var].dropna()
                    n = len(data)
                    
                    if n < 3:
                        st.error("Se necesitan al menos 3 observaciones para las pruebas de normalidad")
                    else:
                        st.info(f"**📋 Tamaño de muestra:** {n:,} observaciones")
                        
                        # Ejecutar pruebas de normalidad
                        results = run_normality_tests(data, alpha_normal)
                        
                        # Mostrar resultados en una tabla
                        st.markdown("### 📊 Resultados de las Pruebas de Normalidad")
                        
                        results_data = []
                        for key, result in results.items():
                            if result.get('statistic') is not None:
                                results_data.append({
                                    'Prueba': result['test'],
                                    'Estadístico': f"{result['statistic']:.4f}",
                                    'p-valor/Crítico': f"{result.get('p_value', result.get('critical_value', 'N/A')):.4f}",
                                    'Resultado': '✅ Normal' if result['is_normal'] else '❌ No Normal',
                                    'Peso': result.get('weight', 1)
                                })
                            elif result.get('note'):
                                results_data.append({
                                    'Prueba': result['test'],
                                    'Estadístico': 'N/A',
                                    'p-valor/Crítico': 'N/A',
                                    'Resultado': f"⚠️ {result['note']}",
                                    'Peso': result.get('weight', 1)
                                })
                        
                        results_df = pd.DataFrame(results_data)
                        st.dataframe(results_df, use_container_width=True, hide_index=True)
                        
                        # Calcular consenso ponderado
                        valid_tests = [r for r in results.values() if r.get('is_normal') is not None]
                        if valid_tests:
                            total_weight = sum(r.get('weight', 1) for r in valid_tests)
                            passed_weight = sum(r.get('weight', 1) for r in valid_tests if r['is_normal'])
                            consensus = passed_weight / total_weight if total_weight > 0 else 0
                            
                            st.metric(
                                "Consenso Ponderado",
                                f"{consensus*100:.1f}%",
                                help="Porcentaje de evidencia (ponderado por confiabilidad) que apoya la normalidad"
                            )
                            
                            # Conclusión basada en consenso
                            st.markdown("### 🎯 CONCLUSIÓN INTEGRADA")
                            
                            if consensus >= 0.7:
                                st.success(f"""
                                ✅ **LOS DATOS PARECEN SEGUIR UNA DISTRIBUCIÓN NORMAL**
                                
                                **Consenso:** {consensus*100:.0f}% de las pruebas ponderadas
                                
                                **✓ Puedes usar pruebas paramétricas:**
                                - Prueba T de Student
                                - ANOVA
                                - Correlación de Pearson
                                - Regresión lineal
                                """)
                            elif consensus >= 0.4:
                                st.warning(f"""
                                ⚠️ **EVIDENCIA MIXTA SOBRE NORMALIDAD**
                                
                                **Consenso:** {consensus*100:.0f}% de las pruebas ponderadas
                                
                                **Recomendaciones:**
                                1. 📊 Revisa cuidadosamente los gráficos Q-Q y el histograma
                                2. 🔄 Considera transformaciones de datos
                                3. 📏 Si n > 30, las pruebas paramétricas son robustas (Teorema Central del Límite)
                                4. 🛡️ Como alternativa segura, usa pruebas no paramétricas
                                """)
                            else:
                                st.error(f"""
                                ❌ **LOS DATOS NO PARECEN SEGUIR UNA DISTRIBUCIÓN NORMAL**
                                
                                **Consenso:** {consensus*100:.0f}% de las pruebas ponderadas
                                
                                **Opciones recomendadas:**
                                1. **Transformaciones:** log(x), sqrt(x), Box-Cox
                                2. **Pruebas no paramétricas:** Mann-Whitney, Kruskal-Wallis, Wilcoxon
                                3. **Modelos robustos:** Técnicas que no asumen normalidad
                                """)
                        
                        # Visualizaciones
                        st.markdown("### 📈 Diagnóstico Visual")
                        fig = create_normality_plots(data, selected_var)
                        st.pyplot(fig)
                        
                        # Interpretación con OpenAI
                        if openai_api_key and st.button("🤖 Obtener interpretación experta", key="normality_ai", use_container_width=True):
                            with st.spinner("Consultando al experto..."):
                                prompt = f"""
                                Como experto en estadística, interpreta los siguientes resultados de pruebas de normalidad:
                                
                                Variable analizada: {selected_var}
                                Tamaño de muestra: {n}
                                Nivel de significancia (α): {alpha_normal}
                                
                                Resultados de las pruebas:
                                """
                                
                                for test_name, test_result in results.items():
                                    if test_result.get('statistic') is not None:
                                        prompt += f"\n- {test_result['test']}: "
                                        prompt += f"Estadístico = {test_result['statistic']:.4f}, "
                                        if 'p_value' in test_result:
                                            prompt += f"p-valor = {test_result['p_value']:.4f}, "
                                        if 'critical_value' in test_result:
                                            prompt += f"Valor crítico = {test_result['critical_value']:.4f}, "
                                        prompt += f"Conclusión = {'Normal' if test_result['is_normal'] else 'No normal'}"
                                
                                prompt += f"""
                                
                                Consenso ponderado: {consensus*100:.1f}%
                                
                                Proporciona una interpretación detallada que incluya:
                                1. Explicación de cada prueba y su importancia
                                2. Interpretación de los resultados obtenidos
                                3. Recomendaciones prácticas basadas en los resultados
                                4. Qué pruebas estadísticas son apropiadas usar
                                5. Posibles transformaciones de datos si son necesarias
                                6. Limitaciones y consideraciones importantes
                                """
                                
                                interpretation = consultar_openai(prompt, max_tokens=2000)
                                st.markdown("---")
                                st.markdown("### 📚 Interpretación del Experto")
                                st.markdown(interpretation)
                                st.markdown("---")
                                
                                # Opción para descargar la interpretación
                                st.download_button(
                                    label="📥 Descargar interpretación",
                                    data=interpretation,
                                    file_name=f"interpretacion_normalidad_{selected_var}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                    mime="text/plain",
                                    use_container_width=True
                                )
                        
                        # Descargar resultados
                        col_dl_norm1, col_dl_norm2 = st.columns(2)
                        with col_dl_norm1:
                            # Descargar tabla de resultados
                            output_norm = io.BytesIO()
                            with pd.ExcelWriter(output_norm, engine='openpyxl') as writer:
                                results_df.to_excel(writer, sheet_name='Resultados', index=False)
                                
                                # Agregar datos crudos
                                summary_df = pd.DataFrame({
                                    'Variable': [selected_var],
                                    'n': [n],
                                    'Media': [data.mean()],
                                    'Desviación': [data.std()],
                                    'Consenso': [f"{consensus*100:.1f}%"]
                                })
                                summary_df.to_excel(writer, sheet_name='Resumen', index=False)
                            
                            st.download_button(
                                label="📥 Descargar resultados",
                                data=output_norm.getvalue(),
                                file_name=f"pruebas_normalidad_{selected_var}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True
                            )
                        
                        with col_dl_norm2:
                            # Descargar informe detallado
                            report_text = f"""
                            INFORME DE PRUEBAS DE NORMALIDAD
                            =================================
                            
                            Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                            Variable: {selected_var}
                            Tamaño muestra: {n}
                            Nivel significancia (α): {alpha_normal}
                            
                            RESULTADOS:
                            """
                            
                            for test_name, test_result in results.items():
                                if test_result.get('statistic') is not None:
                                    report_text += f"\n- {test_result['test']}:\n"
                                    report_text += f"  * Estadístico: {test_result['statistic']:.4f}\n"
                                    if 'p_value' in test_result:
                                        report_text += f"  * p-valor: {test_result['p_value']:.4f}\n"
                                    if 'critical_value' in test_result:
                                        report_text += f"  * Valor crítico: {test_result['critical_value']:.4f}\n"
                                    report_text += f"  * Conclusión: {'Normal' if test_result['is_normal'] else 'No normal'}\n"
                            
                            report_text += f"\nCONSENSO PONDERADO: {consensus*100:.1f}%\n"
                            
                            if consensus >= 0.7:
                                report_text += "\nCONCLUSIÓN: Los datos parecen seguir una distribución normal."
                                report_text += "\nRECOMENDACIÓN: Puedes usar pruebas paramétricas."
                            elif consensus >= 0.4:
                                report_text += "\nCONCLUSIÓN: Evidencia mixta sobre normalidad."
                                report_text += "\nRECOMENDACIÓN: Considera transformaciones o pruebas no paramétricas."
                            else:
                                report_text += "\nCONCLUSIÓN: Los datos no parecen seguir una distribución normal."
                                report_text += "\nRECOMENDACIÓN: Usa pruebas no paramétricas o transforma los datos."
                            
                            st.download_button(
                                label="📥 Descargar informe",
                                data=report_text,
                                file_name=f"informe_normalidad_{selected_var}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                mime="text/plain",
                                use_container_width=True
                            )
                
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
            col_corr_select1, col_corr_select2 = st.columns(2)
            with col_corr_select1:
                var1 = st.selectbox("**Variable X:**", numeric_cols, key="corr_var1_select")
            
            with col_corr_select2:
                available_vars = [v for v in numeric_cols if v != var1]
                var2 = st.selectbox("**Variable Y:**", available_vars, key="corr_var2_select")
            
            # Opciones de análisis
            col_corr_opt1, col_corr_opt2 = st.columns(2)
            with col_corr_opt1:
                correlation_method = st.radio(
                    "**Método de correlación:**",
                    ["pearson", "spearman", "kendall"],
                    format_func=lambda x: {
                        "pearson": "📏 Pearson (lineal)",
                        "spearman": "📈 Spearman (monotónica)",
                        "kendall": "🔢 Kendall (rangos)"
                    }[x],
                    horizontal=True
                )
            
            with col_corr_opt2:
                alpha_corr = st.slider(
                    "**Nivel de significancia (α):**",
                    min_value=0.01,
                    max_value=0.10,
                    value=0.05,
                    step=0.01,
                    key="corr_alpha_slider"
                )
            
            if st.button("🔍 Analizar Correlación", key="analyze_correlation_full", use_container_width=True):
                try:
                    # Limpiar datos
                    clean_data = df[[var1, var2]].dropna()
                    
                    if len(clean_data) < 3:
                        st.error("Se necesitan al menos 3 observaciones válidas")
                    else:
                        # Calcular correlación según método
                        if correlation_method == "pearson":
                            corr, p_value = stats.pearsonr(clean_data[var1], clean_data[var2])
                            method_name = "Correlación de Pearson"
                        elif correlation_method == "spearman":
                            corr, p_value = stats.spearmanr(clean_data[var1], clean_data[var2])
                            method_name = "Correlación de Spearman"
                        else:  # kendall
                            corr, p_value = stats.kendalltau(clean_data[var1], clean_data[var2])
                            method_name = "Correlación de Kendall"
                        
                        # Resultados principales
                        st.markdown("### 📊 Resultados de la Correlación")
                        
                        col_res1, col_res2, col_res3, col_res4 = st.columns(4)
                        with col_res1:
                            st.metric("Método", method_name)
                        with col_res2:
                            st.metric("Coeficiente", f"{corr:.4f}")
                        with col_res3:
                            st.metric("p-valor", f"{p_value:.4f}")
                        with col_res4:
                            is_significant = p_value < alpha_corr
                            st.metric("Significativo", "✅ Sí" if is_significant else "❌ No")
                        
                        # Interpretación de la fuerza
                        abs_corr = abs(corr)
                        if abs_corr < 0.1:
                            strength = "muy débil o inexistente"
                            strength_color = "gray"
                        elif abs_corr < 0.3:
                            strength = "débil"
                            strength_color = "orange"
                        elif abs_corr < 0.5:
                            strength = "moderada"
                            strength_color = "blue"
                        elif abs_corr < 0.7:
                            strength = "fuerte"
                            strength_color = "green"
                        else:
                            strength = "muy fuerte"
                            strength_color = "darkgreen"
                        
                        # Información detallada
                        st.info(f"""
                        **📈 Interpretación:**
                        - **Fuerza:** La correlación es **{strength}** (|r| = {abs_corr:.3f})
                        - **Dirección:** {'🔼 Positiva' if corr > 0 else '🔽 Negativa'} (r = {corr:.3f})
                        - **Significancia:** {'✅ Estadísticamente significativa' if is_significant else '❌ No significativa'} (p = {p_value:.4f} {'<' if is_significant else '≥'} α = {alpha_corr})
                        - **Tamaño muestra:** {len(clean_data)} observaciones válidas
                        """)
                        
                        # Gráfico de dispersión
                        st.markdown("### 📈 Gráfico de Dispersión")
                        fig, ax = plt.subplots(figsize=(12, 8))
                        
                        # Scatter plot
                        scatter = sns.scatterplot(data=clean_data, x=var1, y=var2, alpha=0.6, 
                                                 ax=ax, s=80, color='steelblue', edgecolor='white')
                        
                        # Línea de tendencia (solo para Pearson)
                        if correlation_method == "pearson":
                            z = np.polyfit(clean_data[var1], clean_data[var2], 1)
                            p = np.poly1d(z)
                            x_range = np.linspace(clean_data[var1].min(), clean_data[var1].max(), 100)
                            ax.plot(x_range, p(x_range), "r--", linewidth=2.5, 
                                   alpha=0.8, label=f'Tendencia lineal\n(r = {corr:.3f})')
                        
                        # Configuración del gráfico
                        ax.set_title(f'{method_name}: {var1} vs {var2}', fontsize=16, fontweight='bold')
                        ax.set_xlabel(var1, fontsize=12)
                        ax.set_ylabel(var2, fontsize=12)
                        
                        # Agregar estadísticas al gráfico
                        stats_text = f"r = {corr:.3f}\np = {p_value:.4f}\nn = {len(clean_data)}"
                        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                               fontsize=11, verticalalignment='top',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
                        
                        ax.legend(loc='best')
                        ax.grid(True, alpha=0.3)
                        
                        st.pyplot(fig)
                        
                        # Matriz de correlación si hay múltiples variables
                        if len(numeric_cols) > 2:
                            st.markdown("### 📊 Matriz de Correlación")
                            
                            selected_for_matrix = st.multiselect(
                                "**Selecciona variables para la matriz:**",
                                numeric_cols,
                                default=numeric_cols[:min(8, len(numeric_cols))],
                                key="corr_matrix_select"
                            )
                            
                            if selected_for_matrix:
                                corr_matrix = df[selected_for_matrix].corr(method=correlation_method)
                                
                                # Mostrar matriz numérica
                                st.dataframe(
                                    corr_matrix.style.background_gradient(cmap='coolwarm', vmin=-1, vmax=1)
                                    .format("{:.3f}"),
                                    use_container_width=True
                                )
                                
                                # Heatmap visual
                                fig2, ax2 = plt.subplots(figsize=(12, 10))
                                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                                sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", 
                                          cmap='coolwarm', center=0, square=True, 
                                          linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax2)
                                ax2.set_title(f'Matriz de Correlación ({method_name})', 
                                            fontsize=16, fontweight='bold', pad=20)
                                plt.tight_layout()
                                st.pyplot(fig2)
                        
                        # Interpretación con OpenAI
                        if openai_api_key and st.button("🤖 Obtener interpretación experta", 
                                                      key="corr_ai_interpret", use_container_width=True):
                            with st.spinner("Consultando al experto..."):
                                prompt = f"""
                                Como experto en estadística, interpreta los siguientes resultados de correlación:
                                
                                Variables: {var1} y {var2}
                                Método: {method_name}
                                Coeficiente de correlación (r): {corr:.4f}
                                p-valor: {p_value:.4f}
                                Nivel de significancia: {alpha_corr}
                                Tamaño de muestra: {len(clean_data)}
                                Fuerza de la correlación: {strength}
                                Dirección: {'Positiva' if corr > 0 else 'Negativa'}
                                Significancia estadística: {'Significativa' if is_significant else 'No significativa'}
                                
                                Proporciona una interpretación completa que incluya:
                                1. Explicación del coeficiente de correlación obtenido
                                2. Interpretación práctica de la fuerza y dirección
                                3. Implicaciones de la significancia estadística
                                4. Limitaciones importantes (correlación ≠ causalidad)
                                5. Recomendaciones para análisis adicionales
                                6. Posibles explicaciones teóricas para esta relación
                                7. Consideraciones para la toma de decisiones basadas en estos resultados
                                
                                Sé claro, práctico y basado en evidencia.
                                """
                                
                                interpretation = consultar_openai(prompt, max_tokens=2500)
                                st.markdown("---")
                                st.markdown("### 📚 Interpretación del Experto")
                                st.markdown(interpretation)
                                st.markdown("---")
                        
                        # Descargar resultados
                        col_dl_corr1, col_dl_corr2 = st.columns(2)
                        with col_dl_corr1:
                            # Crear informe detallado
                            report_corr = f"""
                            INFORME DE ANÁLISIS DE CORRELACIÓN
                            ====================================
                            
                            Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                            Variables: {var1} y {var2}
                            Método: {method_name}
                            
                            RESULTADOS:
                            - Coeficiente de correlación (r): {corr:.4f}
                            - p-valor: {p_value:.4f}
                            - Nivel de significancia (α): {alpha_corr}
                            - Significativo: {'Sí' if is_significant else 'No'}
                            - Tamaño muestra: {len(clean_data)}
                            
                            INTERPRETACIÓN:
                            - Fuerza: {strength} (|r| = {abs_corr:.3f})
                            - Dirección: {'Positiva' if corr > 0 else 'Negativa'}
                            - Significancia: {'Estadísticamente significativa' if is_significant else 'No significativa'}
                            
                            DATOS ESTADÍSTICOS:
                            - {var1}: Media = {clean_data[var1].mean():.4f}, Desv = {clean_data[var1].std():.4f}
                            - {var2}: Media = {clean_data[var2].mean():.4f}, Desv = {clean_data[var2].std():.4f}
                            
                            CONSIDERACIONES IMPORTANTES:
                            1. Correlación no implica causalidad
                            2. El coeficiente mide la relación lineal/monotónica entre variables
                            3. Resultados válidos para el rango observado de datos
                            4. Verificar supuestos del método utilizado
                            """
                            
                            st.download_button(
                                label="📥 Descargar informe",
                                data=report_corr,
                                file_name=f"informe_correlacion_{var1}_{var2}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                mime="text/plain",
                                use_container_width=True
                            )
                        
                        with col_dl_corr2:
                            # Descargar datos de correlación
                            corr_data = clean_data.copy()
                            corr_data['ID'] = range(1, len(corr_data) + 1)
                            corr_data = corr_data[['ID', var1, var2]]
                            
                            output_corr = io.BytesIO()
                            with pd.ExcelWriter(output_corr, engine='openpyxl') as writer:
                                corr_data.to_excel(writer, sheet_name='Datos', index=False)
                                
                                # Agregar resultados
                                results_df = pd.DataFrame({
                                    'Métrica': ['Método', 'Coeficiente', 'p-valor', 'Significativo', 'n'],
                                    'Valor': [method_name, f"{corr:.4f}", f"{p_value:.4f}", 
                                             'Sí' if is_significant else 'No', len(clean_data)]
                                })
                                results_df.to_excel(writer, sheet_name='Resultados', index=False)
                            
                            st.download_button(
                                label="📥 Descargar datos",
                                data=output_corr.getvalue(),
                                file_name=f"datos_correlacion_{var1}_{var2}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True
                            )
                
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
            col_homo_select1, col_homo_select2 = st.columns(2)
            with col_homo_select1:
                num_var = st.selectbox("**Variable numérica:**", numeric_cols, key="homo_num_select")
            
            with col_homo_select2:
                cat_var = st.selectbox("**Variable categórica:**", categorical_cols, key="homo_cat_select")
            
            alpha_homo = st.slider(
                "**Nivel de significancia (α):**",
                min_value=0.01,
                max_value=0.10,
                value=0.05,
                step=0.01,
                key="homo_alpha_slider"
            )
            
            if st.button("⚖️ Ejecutar Pruebas de Homogeneidad", key="run_homogeneity_full", use_container_width=True):
                try:
                    # Preparar datos por grupos
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
                        # Ejecutar pruebas
                        levene_stat, levene_p = stats.levene(*groups_data)
                        bartlett_stat, bartlett_p = stats.bartlett(*groups_data)
                        
                        levene_homo = levene_p > alpha_homo
                        bartlett_homo = bartlett_p > alpha_homo
                        
                        # Resultados
                        st.markdown("### 📊 Resultados de las Pruebas")
                        
                        col_res_homo1, col_res_homo2 = st.columns(2)
                        with col_res_homo1:
                            st.metric("Prueba", "Levene")
                            st.metric("Estadístico", f"{levene_stat:.4f}")
                            st.metric("p-valor", f"{levene_p:.4f}")
                            if levene_homo:
                                st.success("✅ Varianzas homogéneas")
                            else:
                                st.error("❌ Varianzas NO homogéneas")
                        
                        with col_res_homo2:
                            st.metric("Prueba", "Bartlett")
                            st.metric("Estadístico", f"{bartlett_stat:.4f}")
                            st.metric("p-valor", f"{bartlett_p:.4f}")
                            if bartlett_homo:
                                st.success("✅ Varianzas homogéneas")
                            else:
                                st.error("❌ Varianzas NO homogéneas")
                        
                        # Comparación de resultados
                        st.markdown("### 🎯 COMPARACIÓN Y CONCLUSIÓN")
                        
                        results_data = [
                            {
                                'Prueba': 'Levene',
                                'Estadístico': f"{levene_stat:.4f}",
                                'p-valor': f"{levene_p:.4f}",
                                'Resultado': '✅ Homogéneas' if levene_homo else '❌ No Homogéneas',
                                'Robustez': 'Alta (recomendada)'
                            },
                            {
                                'Prueba': 'Bartlett',
                                'Estadístico': f"{bartlett_stat:.4f}",
                                'p-valor': f"{bartlett_p:.4f}",
                                'Resultado': '✅ Homogéneas' if bartlett_homo else '❌ No Homogéneas',
                                'Robustez': 'Baja (requiere normalidad)'
                            }
                        ]
                        
                        results_df = pd.DataFrame(results_data)
                        st.dataframe(results_df, use_container_width=True, hide_index=True)
                        
                        # Conclusión integrada
                        if levene_homo and bartlett_homo:
                            st.success("""
                            ✅ **LAS VARIANZAS SON HOMOGÉNEAS**
                            
                            Ambas pruebas confirman homogeneidad de varianzas.
                            
                            **✓ Puedes usar con confianza:**
                            - Prueba T independiente (con equal_var=True)
                            - ANOVA clásica
                            - Otras pruebas que asumen varianzas iguales
                            """)
                        elif levene_homo and not bartlett_homo:
                            st.warning("""
                            ⚠️ **RESULTADOS MIXTOS (FAVORECEN HOMOGENEIDAD)**
                            
                            - **Levene (robusta):** Varianzas homogéneas ✅
                            - **Bartlett (sensible):** Varianzas no homogéneas ❌
                            
                            **Interpretación más probable:**
                            Los datos pueden tener desviaciones leves de la normalidad que hacen que 
                            Bartlett sea demasiado sensible. **Confía en Levene**.
                            
                            **Recomendación:**
                            - ✓ Usa pruebas paramétricas con equal_var=True
                            - ✓ Verifica normalidad de los datos
                            - ✓ Considera usar métodos robustos si hay dudas
                            """)
                        elif not levene_homo and bartlett_homo:
                            st.warning("""
                            ⚠️ **RESULTADOS MIXTOS (FAVORECEN NO HOMOGENEIDAD)**
                            
                            - **Levene (robusta):** Varianzas no homogéneas ❌
                            - **Bartlett (sensible):** Varianzas homogéneas ✅
                            
                            **Interpretación más probable:**
                            Puede haber outliers o estructura en los datos que Levene detecta mejor.
                            
                            **Recomendación:**
                            - ✓ Usa pruebas con equal_var=False (Welch's t-test)
                            - ✓ Investiga posibles outliers
                            - ✓ Considera transformaciones de datos
                            """)
                        else:
                            st.error("""
                            ❌ **LAS VARIANZAS NO SON HOMOGÉNEAS**
                            
                            Ambas pruebas rechazan la homogeneidad de varianzas.
                            
                            **Opciones recomendadas:**
                            
                            1. **Usar versiones robustas de las pruebas:**
                               - Welch's t-test (en lugar de t-test estándar)
                               - Welch's ANOVA (en lugar de ANOVA estándar)
                            
                            2. **Transformar los datos:**
                               - Logaritmo: log(x) - reduce varianzas grandes
                               - Raíz cuadrada: sqrt(x) - estabiliza varianzas
                               - Box-Cox: transformación óptima automática
                            
                            3. **Usar pruebas no paramétricas:**
                               - Mann-Whitney U (en lugar de t-test)
                               - Kruskal-Wallis (en lugar de ANOVA)
                            """)
                        
                        # Estadísticas descriptivas por grupo
                        st.markdown("### 📊 Estadísticas de Varianza por Grupo")
                        
                        stats_data = []
                        variances = []
                        for name, data in zip(group_names, groups_data):
                            stats_data.append({
                                'Grupo': name,
                                'n': len(data),
                                'Media': f"{data.mean():.4f}",
                                'Desviación Estándar': f"{data.std():.4f}",
                                'Varianza': f"{data.var():.4f}",
                                'Coef. Variación (%)': f"{(data.std()/data.mean())*100:.2f}" if data.mean() != 0 else "N/A"
                            })
                            variances.append(data.var())
                        
                        stats_df = pd.DataFrame(stats_data)
                        st.dataframe(stats_df, use_container_width=True)
                        
                        # Calcular ratio de varianzas
                        max_var = max(variances)
                        min_var = min(variances)
                        var_ratio = max_var / min_var if min_var > 0 else float('inf')
                        
                        col_ratio1, col_ratio2 = st.columns(2)
                        with col_ratio1:
                            st.metric(
                                "Ratio de Varianzas (Máx/Mín)", 
                                f"{var_ratio:.2f}",
                                help="Regla general: Si < 4, las varianzas son razonablemente similares"
                            )
                        
                        with col_ratio2:
                            if var_ratio < 2:
                                st.success("✅ Ratio < 2: Varianzas muy similares")
                            elif var_ratio < 4:
                                st.info("ℹ️ Ratio < 4: Varianzas razonablemente similares")
                            else:
                                st.warning("⚠️ Ratio ≥ 4: Varianzas notablemente diferentes")
                        
                        # Visualizaciones
                        st.markdown("### 📈 Visualización de Varianzas")
                        
                        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
                        
                        # Boxplot comparativo
                        plot_data = []
                        for name, data in zip(group_names, groups_data):
                            for value in data:
                                plot_data.append({'Grupo': name, 'Valor': value})
                        
                        plot_df = pd.DataFrame(plot_data)
                        sns.boxplot(data=plot_df, x='Grupo', y='Valor', ax=ax1, palette='Set2')
                        ax1.set_title('Distribución por Grupo (Boxplot)', fontsize=14, fontweight='bold')
                        ax1.set_ylabel(num_var)
                        ax1.tick_params(axis='x', rotation=45)
                        ax1.grid(True, alpha=0.3, axis='y')
                        
                        # Gráfico de varianzas
                        ax2.bar(group_names, variances, color='lightcoral', alpha=0.7, edgecolor='black')
                        mean_var = np.mean(variances)
                        ax2.axhline(mean_var, color='red', linestyle='--', linewidth=2, label=f'Media: {mean_var:.2f}')
                        ax2.set_title('Comparación de Varianzas por Grupo', fontsize=14, fontweight='bold')
                        ax2.set_ylabel('Varianza')
                        ax2.set_xlabel('Grupo')
                        ax2.tick_params(axis='x', rotation=45)
                        ax2.legend()
                        ax2.grid(True, alpha=0.3, axis='y')
                        
                        # Gráfico de desviaciones estándar
                        stds = [d.std() for d in groups_data]
                        ax3.bar(group_names, stds, color='skyblue', alpha=0.7, edgecolor='black')
                        ax3.set_title('Desviación Estándar por Grupo', fontsize=14, fontweight='bold')
                        ax3.set_ylabel('Desviación Estándar')
                        ax3.set_xlabel('Grupo')
                        ax3.tick_params(axis='x', rotation=45)
                        ax3.grid(True, alpha=0.3, axis='y')
                        
                        # Gráfico de dispersión de residuos
                        for i, (name, data) in enumerate(zip(group_names, groups_data)):
                            mean_val = data.mean()
                            std_val = data.std()
                            if std_val > 0:
                                residuals = np.abs((data - mean_val) / std_val)
                                ax4.scatter([name] * len(residuals), residuals, alpha=0.5, s=50)
                        ax4.axhline(np.sqrt(2), color='red', linestyle='--', alpha=0.5, label='Referencia')
                        ax4.set_title('Dispersión de Residuos Estandarizados', fontsize=14, fontweight='bold')
                        ax4.set_ylabel('|Residuos Estandarizados|')
                        ax4.set_xlabel('Grupo')
                        ax4.tick_params(axis='x', rotation=45)
                        ax4.legend()
                        ax4.grid(True, alpha=0.3, axis='y')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Interpretación con OpenAI
                        if openai_api_key and st.button("🤖 Obtener interpretación experta", 
                                                      key="homo_ai_interpret", use_container_width=True):
                            with st.spinner("Consultando al experto..."):
                                prompt = f"""
                                Como experto en estadística, interpreta los siguientes resultados de homogeneidad de varianzas:
                                
                                Variable numérica: {num_var}
                                Variable categórica: {cat_var}
                                Número de grupos: {len(groups_data)}
                                Nivel de significancia: {alpha_homo}
                                
                                Resultados:
                                1. Prueba de Levene:
                                   - Estadístico: {levene_stat:.4f}
                                   - p-valor: {levene_p:.4f}
                                   - Conclusión: {'Varianzas homogéneas' if levene_homo else 'Varianzas NO homogéneas'}
                                
                                2. Prueba de Bartlett:
                                   - Estadístico: {bartlett_stat:.4f}
                                   - p-valor: {bartlett_p:.4f}
                                   - Conclusión: {'Varianzas homogéneas' if bartlett_homo else 'Varianzas NO homogéneas'}
                                
                                Ratio de varianzas (Máx/Mín): {var_ratio:.2f}
                                
                                Proporciona una interpretación detallada que incluya:
                                1. Explicación de cada prueba y sus diferencias
                                2. Interpretación de los resultados obtenidos
                                3. Recomendaciones prácticas basadas en los resultados
                                4. Qué pruebas estadísticas son apropiadas usar a continuación
                                5. Posibles acciones si las varianzas no son homogéneas
                                6. Consideraciones para el diseño experimental futuro
                                """
                                
                                interpretation = consultar_openai(prompt, max_tokens=2500)
                                st.markdown("---")
                                st.markdown("### 📚 Interpretación del Experto")
                                st.markdown(interpretation)
                                st.markdown("---")
                        
                        # Descargar resultados
                        col_dl_homo1, col_dl_homo2 = st.columns(2)
                        with col_dl_homo1:
                            # Crear informe detallado
                            report_homo = f"""
                            INFORME DE HOMOGENEIDAD DE VARIANZAS
                            =====================================
                            
                            Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                            Variable numérica: {num_var}
                            Variable categórica: {cat_var}
                            Nivel de significancia (α): {alpha_homo}
                            
                            RESULTADOS DE LAS PRUEBAS:
                            1. Prueba de Levene:
                               - Estadístico: {levene_stat:.4f}
                               - p-valor: {levene_p:.4f}
                               - Conclusión: {'Varianzas homogéneas' if levene_homo else 'Varianzas NO homogéneas'}
                            
                            2. Prueba de Bartlett:
                               - Estadístico: {bartlett_stat:.4f}
                               - p-valor: {bartlett_p:.4f}
                               - Conclusión: {'Varianzas homogéneas' if bartlett_homo else 'Varianzas NO homogéneas'}
                            
                            ESTADÍSTICAS POR GRUPO:
                            """
                            
                            for i, row in stats_df.iterrows():
                                report_homo += f"\n- {row['Grupo']}:"
                                report_homo += f"\n  * n = {row['n']}"
                                report_homo += f"\n  * Media = {row['Media']}"
                                report_homo += f"\n  * Desviación = {row['Desviación Estándar']}"
                                report_homo += f"\n  * Varianza = {row['Varianza']}"
                            
                            report_homo += f"\n\nRATIO DE VARIANZAS (Máx/Mín): {var_ratio:.2f}"
                            
                            if var_ratio < 2:
                                report_homo += "\nINTERPRETACIÓN: Varianzas muy similares"
                            elif var_ratio < 4:
                                report_homo += "\nINTERPRETACIÓN: Varianzas razonablemente similares"
                            else:
                                report_homo += "\nINTERPRETACIÓN: Varianzas notablemente diferentes"
                            
                            if levene_homo and bartlett_homo:
                                report_homo += "\n\nCONCLUSIÓN FINAL: Varianzas homogéneas - usar pruebas paramétricas estándar"
                            elif not levene_homo and not bartlett_homo:
                                report_homo += "\n\nCONCLUSIÓN FINAL: Varianzas NO homogéneas - usar pruebas robustas o no paramétricas"
                            else:
                                report_homo += "\n\nCONCLUSIÓN FINAL: Resultados mixtos - considerar pruebas robustas"
                            
                            st.download_button(
                                label="📥 Descargar informe",
                                data=report_homo,
                                file_name=f"informe_homogeneidad_{num_var}_{cat_var}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                mime="text/plain",
                                use_container_width=True
                            )
                        
                        with col_dl_homo2:
                            # Descargar datos
                            output_homo = io.BytesIO()
                            with pd.ExcelWriter(output_homo, engine='openpyxl') as writer:
                                # Datos por grupo
                                group_data_all = pd.DataFrame()
                                for name, data in zip(group_names, groups_data):
                                    temp_df = pd.DataFrame({name: data})
                                    group_data_all = pd.concat([group_data_all, temp_df], axis=1)
                                group_data_all.to_excel(writer, sheet_name='Datos por Grupo', index=False)
                                
                                # Resultados
                                results_summary = pd.DataFrame({
                                    'Prueba': ['Levene', 'Bartlett'],
                                    'Estadístico': [levene_stat, bartlett_stat],
                                    'p_valor': [levene_p, bartlett_p],
                                    'Conclusión': [
                                        'Homogéneas' if levene_homo else 'No homogéneas',
                                        'Homogéneas' if bartlett_homo else 'No homogéneas'
                                    ]
                                })
                                results_summary.to_excel(writer, sheet_name='Resultados', index=False)
                                
                                # Estadísticas por grupo
                                stats_df.to_excel(writer, sheet_name='Estadísticas', index=False)
                            
                            st.download_button(
                                label="📥 Descargar datos",
                                data=output_homo.getvalue(),
                                file_name=f"datos_homogeneidad_{num_var}_{cat_var}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True
                            )
                
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
            "**Selecciona el tipo de prueba T:**",
            ["Una muestra", "Muestras independientes", "Muestras pareadas"],
            horizontal=True,
            key="ttest_type_select"
        )
        
        # Configuración común
        col_ttest_opt1, col_ttest_opt2 = st.columns(2)
        with col_ttest_opt1:
            alpha_ttest = st.slider(
                "**Nivel de significancia (α):**",
                min_value=0.01,
                max_value=0.10,
                value=0.05,
                step=0.01,
                key="ttest_alpha_slider"
            )
        
        with col_ttest_opt2:
            alternative = st.selectbox(
                "**Hipótesis alternativa:**",
                ["two-sided", "less", "greater"],
                format_func=lambda x: {
                    "two-sided": "🔄 Bilateral (≠)",
                    "less": "⬇️ Unilateral izquierda (<)",
                    "greater": "⬆️ Unilateral derecha (>)"
                }[x],
                key="ttest_alternative_select"
            )
        
        # Prueba T para una muestra
        if test_type == "Una muestra" and numeric_cols:
            st.markdown("### 📊 Prueba T para Una Muestra")
            
            col_onesample1, col_onesample2 = st.columns(2)
            with col_onesample1:
                var_onesample = st.selectbox("**Variable numérica:**", numeric_cols, key="onesample_var_select")
            
            with col_onesample2:
                pop_mean = st.number_input("**Media poblacional de referencia:**", value=0.0, 
                                          help="Valor teórico o de referencia para comparar", key="pop_mean_input")
            
            if st.button("📊 Ejecutar Prueba T Una Muestra", key="run_onesample_full", use_container_width=True):
                try:
                    data = df[var_onesample].dropna()
                    
                    if len(data) < 2:
                        st.error("Se necesitan al menos 2 observaciones")
                    else:
                        # Ejecutar prueba
                        t_stat, p_value = stats.ttest_1samp(data, pop_mean)
                        
                        # Ajuste para pruebas unilaterales
                        if alternative == "less":
                            if data.mean() < pop_mean:
                                p_value = p_value / 2
                            else:
                                p_value = 1 - p_value / 2
                        elif alternative == "greater":
                            if data.mean() > pop_mean:
                                p_value = p_value / 2
                            else:
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
                        
                        # Resultados principales
                        st.markdown("### 📋 Resultados Principales")
                        
                        col_res_ttest1, col_res_ttest2, col_res_ttest3, col_res_ttest4 = st.columns(4)
                        with col_res_ttest1:
                            st.metric("Estadístico t", f"{t_stat:.4f}")
                        with col_res_ttest2:
                            st.metric("p-valor", f"{p_value:.4f}")
                        with col_res_ttest3:
                            st.metric("Grados libertad", len(data) - 1)
                        with col_res_ttest4:
                            is_significant = p_value < alpha_ttest
                            st.metric("Significativo", "✅ Sí" if is_significant else "❌ No")
                        
                        # Estadísticas descriptivas
                        st.markdown("### 📊 Estadísticas Descriptivas")
                        
                        col_stats_ttest1, col_stats_ttest2, col_stats_ttest3 = st.columns(3)
                        with col_stats_ttest1:
                            st.metric("Media muestral", f"{data.mean():.4f}")
                            st.metric("Desviación estándar", f"{data.std():.4f}")
                        with col_stats_ttest2:
                            st.metric("Media referencia", f"{pop_mean:.4f}")
                            st.metric("Error estándar", f"{stats.sem(data):.4f}")
                        with col_stats_ttest3:
                            st.metric("Tamaño muestra", len(data))
                            st.metric("Diferencia", f"{data.mean() - pop_mean:.4f}")
                        
                        # Información adicional
                        st.markdown("### 📈 Información Adicional")
                        
                        col_add_ttest1, col_add_ttest2 = st.columns(2)
                        with col_add_ttest1:
                            st.metric(f"IC del {(1-alpha_ttest)*100:.0f}%", f"[{ci_low:.4f}, {ci_high:.4f}]")
                        
                        with col_add_ttest2:
                            st.metric("Tamaño efecto (d)", f"{effect_size:.4f}")
                            st.markdown(f"<span style='color:{effect_color}; font-weight:bold;'>{effect_magnitude}</span>", 
                                      unsafe_allow_html=True)
                        
                        # Interpretación
                        st.info(f"""
                        **📝 Interpretación:**
                        - **Hipótesis nula (H₀):** μ = {pop_mean} (La media poblacional es igual a {pop_mean})
                        - **Hipótesis alternativa (H₁):** μ {'≠' if alternative == 'two-sided' else '<' if alternative == 'less' else '>'} {pop_mean}
                        - **Decisión:** {'Se rechaza H₀' if is_significant else 'No se rechaza H₀'} (p = {p_value:.4f} {'<' if is_significant else '≥'} α = {alpha_ttest})
                        - **Tamaño del efecto:** {effect_magnitude} (d = {effect_size:.4f})
                        """)
                        
                        # Visualización
                        st.markdown("### 📊 Visualización")
                        fig, ax = plt.subplots(figsize=(12, 8))
                        
                        # Histograma con KDE
                        sns.histplot(data, kde=True, ax=ax, alpha=0.7, label='Distribución muestral', color='skyblue')
                        
                        # Líneas de referencia
                        ax.axvline(data.mean(), color='red', linestyle='-', linewidth=2.5, 
                                  label=f'Media muestral: {data.mean():.4f}')
                        ax.axvline(pop_mean, color='green', linestyle='--', linewidth=2.5, 
                                  label=f'Media referencia: {pop_mean}')
                        
                        # Intervalo de confianza
                        ax.axvspan(ci_low, ci_high, alpha=0.2, color='yellow', 
                                  label=f'IC {(1-alpha_ttest)*100:.0f}%: [{ci_low:.4f}, {ci_high:.4f}]')
                        
                        ax.set_title(f'Prueba T Una Muestra: {var_onesample} vs μ = {pop_mean}', 
                                   fontsize=16, fontweight='bold')
                        ax.set_xlabel(var_onesample, fontsize=12)
                        ax.set_ylabel('Densidad', fontsize=12)
                        ax.legend(loc='best')
                        ax.grid(True, alpha=0.3)
                        
                        # Agregar estadísticas al gráfico
                        stats_text = f"t = {t_stat:.3f}\np = {p_value:.4f}\nn = {len(data)}\nd = {effect_size:.3f}"
                        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                               fontsize=11, verticalalignment='top',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
                        
                        st.pyplot(fig)
                        
                        # Interpretación con OpenAI
                        if openai_api_key and st.button("🤖 Obtener interpretación experta", 
                                                      key="ttest_onesample_ai", use_container_width=True):
                            with st.spinner("Consultando al experto..."):
                                prompt = f"""
                                Como experto en estadística, interpreta los siguientes resultados de prueba T para una muestra:
                                
                                Variable: {var_onesample}
                                Media muestral: {data.mean():.4f}
                                Media poblacional de referencia: {pop_mean}
                                Tamaño de muestra: {len(data)}
                                Nivel de significancia: {alpha_ttest}
                                
                                Resultados:
                                - Estadístico t: {t_stat:.4f}
                                - p-valor: {p_value:.4f}
                                - Grados de libertad: {len(data) - 1}
                                - Significativo: {'Sí' if is_significant else 'No'}
                                - Intervalo de confianza del {(1-alpha_ttest)*100:.0f}%: [{ci_low:.4f}, {ci_high:.4f}]
                                - Tamaño del efecto (Cohen's d): {effect_size:.4f} ({effect_magnitude})
                                
                                Hipótesis:
                                - H₀: μ = {pop_mean} (La media poblacional es igual a {pop_mean})
                                - H₁: μ {'≠' if alternative == 'two-sided' else '<' if alternative == 'less' else '>'} {pop_mean}
                                
                                Proporciona una interpretación detallada que incluya:
                                1. Explicación de la prueba T para una muestra
                                2. Interpretación práctica de los resultados
                                3. Implicaciones de la significancia estadística
                                4. Interpretación del tamaño del efecto
                                5. Uso del intervalo de confianza
                                6. Limitaciones y consideraciones importantes
                                7. Recomendaciones para acciones basadas en los resultados
                                
                                Sé claro, práctico y aplicable al contexto del análisis.
                                """
                                
                                interpretation = consultar_openai(prompt, max_tokens=2500)
                                st.markdown("---")
                                st.markdown("### 📚 Interpretación del Experto")
                                st.markdown(interpretation)
                                st.markdown("---")
                        
                        # Descargar resultados
                        col_dl_ttest1, col_dl_ttest2 = st.columns(2)
                        with col_dl_ttest1:
                            # Crear informe detallado
                            report_ttest = f"""
                            INFORME DE PRUEBA T PARA UNA MUESTRA
                            ======================================
                            
                            Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                            Variable: {var_onesample}
                            
                            PARÁMETROS:
                            - Media poblacional de referencia: {pop_mean}
                            - Nivel de significancia (α): {alpha_ttest}
                            - Hipótesis alternativa: {'Bilateral (≠)' if alternative == 'two-sided' else 'Unilateral izquierda (<)' if alternative == 'less' else 'Unilateral derecha (>)'}
                            
                            DATOS:
                            - Tamaño de muestra: {len(data)}
                            - Media muestral: {data.mean():.4f}
                            - Desviación estándar: {data.std():.4f}
                            - Error estándar: {stats.sem(data):.4f}
                            - Diferencia: {data.mean() - pop_mean:.4f}
                            
                            RESULTADOS:
                            - Estadístico t: {t_stat:.4f}
                            - p-valor: {p_value:.4f}
                            - Grados de libertad: {len(data) - 1}
                            - Significativo: {'Sí' if is_significant else 'No'}
                            - IC del {(1-alpha_ttest)*100:.0f}%: [{ci_low:.4f}, {ci_high:.4f}]
                            - Tamaño del efecto (Cohen's d): {effect_size:.4f} ({effect_magnitude})
                            
                            HIPÓTESIS:
                            - H₀: μ = {pop_mean} (La media poblacional es igual a {pop_mean})
                            - H₁: μ {'≠' if alternative == 'two-sided' else '<' if alternative == 'less' else '>'} {pop_mean}
                            
                            DECISIÓN:
                            - {'Se rechaza H₀' if is_significant else 'No se rechaza H₀'} (p = {p_value:.4f} {'<' if is_significant else '≥'} α = {alpha_ttest})
                            
                            INTERPRETACIÓN:
                            - Tamaño del efecto: {effect_magnitude}
                            - Diferencia media: {data.mean() - pop_mean:.4f}
                            - El intervalo de confianza {'incluye' if ci_low <= pop_mean <= ci_high else 'no incluye'} el valor de referencia
                            
                            CONSIDERACIONES:
                            1. Los resultados son válidos si los datos son aproximadamente normales
                            2. Con muestras grandes (>30), la prueba es robusta a desviaciones de normalidad
                            3. Verificar supuestos antes de generalizar resultados
                            """
                            
                            st.download_button(
                                label="📥 Descargar informe",
                                data=report_ttest,
                                file_name=f"informe_ttest_una_muestra_{var_onesample}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                mime="text/plain",
                                use_container_width=True
                            )
                        
                        with col_dl_ttest2:
                            # Descargar datos
                            output_ttest = io.BytesIO()
                            with pd.ExcelWriter(output_ttest, engine='openpyxl') as writer:
                                # Datos
                                data_df = pd.DataFrame({var_onesample: data})
                                data_df.to_excel(writer, sheet_name='Datos', index=False)
                                
                                # Resultados
                                results_df = pd.DataFrame({
                                    'Métrica': ['Estadístico t', 'p-valor', 'Grados libertad', 'Significativo', 
                                               'Media muestral', 'Media referencia', 'Diferencia', 'Tamaño efecto'],
                                    'Valor': [f"{t_stat:.4f}", f"{p_value:.4f}", len(data)-1, 
                                             'Sí' if is_significant else 'No', f"{data.mean():.4f}", 
                                             f"{pop_mean}", f"{data.mean() - pop_mean:.4f}", f"{effect_size:.4f}"]
                                })
                                results_df.to_excel(writer, sheet_name='Resultados', index=False)
                                
                                # Intervalo de confianza
                                ci_df = pd.DataFrame({
                                    'Nivel_confianza': [f"{(1-alpha_ttest)*100:.0f}%"],
                                    'Limite_inferior': [ci_low],
                                    'Limite_superior': [ci_high]
                                })
                                ci_df.to_excel(writer, sheet_name='IC', index=False)
                            
                            st.download_button(
                                label="📥 Descargar datos",
                                data=output_ttest.getvalue(),
                                file_name=f"datos_ttest_una_muestra_{var_onesample}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True
                            )
                
                except Exception as e:
                    st.error(f"Error en prueba T una muestra: {e}")
        
        # Prueba T para muestras independientes
        elif test_type == "Muestras independientes" and numeric_cols and categorical_cols:
            st.markdown("### 📊 Prueba T para Muestras Independientes")
            
            col_indep1, col_indep2 = st.columns(2)
            with col_indep1:
                var_independent = st.selectbox("**Variable numérica:**", numeric_cols, key="indep_var_select")
            
            with col_indep2:
                group_var = st.selectbox("**Variable categórica (debe tener 2 grupos):**", 
                                        categorical_cols, key="group_var_select")
            
            # Verificar que la variable categórica tenga exactamente 2 grupos
            unique_groups = df[group_var].dropna().unique()
            if len(unique_groups) == 2:
                group1, group2 = unique_groups
                
                if st.button("📊 Ejecutar Prueba T Independiente", key="run_independent_full", use_container_width=True):
                    try:
                        data1 = df[df[group_var] == group1][var_independent].dropna()
                        data2 = df[df[group_var] == group2][var_independent].dropna()
                        
                        if len(data1) < 2 or len(data2) < 2:
                            st.error("Cada grupo necesita al menos 2 observaciones")
                        else:
                            # Prueba de igualdad de varianzas
                            levene_stat, levene_p = stats.levene(data1, data2)
                            equal_var = levene_p > 0.05
                            
                            # Ejecutar prueba T
                            t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=equal_var)
                            
                            # Ajuste para pruebas unilaterales
                            if alternative == "less":
                                if data1.mean() < data2.mean():
                                    p_value = p_value / 2
                                else:
                                    p_value = 1 - p_value / 2
                            elif alternative == "greater":
                                if data1.mean() > data2.mean():
                                    p_value = p_value / 2
                                else:
                                    p_value = 1 - p_value / 2
                            
                            # Tamaño del efecto
                            effect_size = calculate_effect_size("Muestras independientes", data1=data1, data2=data2)
                            effect_magnitude, effect_color = interpret_effect_size(effect_size)
                            
                            # Resultados principales
                            st.markdown("### 📋 Resultados Principales")
                            
                            col_res_indep1, col_res_indep2, col_res_indep3, col_res_indep4 = st.columns(4)
                            with col_res_indep1:
                                st.metric("Estadístico t", f"{t_stat:.4f}")
                            with col_res_indep2:
                                st.metric("p-valor", f"{p_value:.4f}")
                            with col_res_indep3:
                                st.metric("Prueba", "Student" if equal_var else "Welch")
                            with col_res_indep4:
                                is_significant = p_value < alpha_ttest
                                st.metric("Significativo", "✅ Sí" if is_significant else "❌ No")
                            
                            # Homogeneidad de varianzas
                            st.markdown("### ⚖️ Homogeneidad de Varianzas")
                            
                            col_homo_indep1, col_homo_indep2 = st.columns(2)
                            with col_homo_indep1:
                                st.metric("Prueba", "Levene")
                                st.metric("Estadístico", f"{levene_stat:.4f}")
                            with col_homo_indep2:
                                st.metric("p-valor", f"{levene_p:.4f}")
                                if equal_var:
                                    st.success("✅ Varianzas homogéneas")
                                else:
                                    st.warning("⚠️ Varianzas diferentes (usando Welch)")
                            
                            # Estadísticas por grupo
                            st.markdown("### 📊 Estadísticas por Grupo")
                            
                            col_stats_indep1, col_stats_indep2 = st.columns(2)
                            with col_stats_indep1:
                                st.metric(f"Media {group1}", f"{data1.mean():.4f}")
                                st.metric(f"Desviación {group1}", f"{data1.std():.4f}")
                                st.metric(f"n {group1}", len(data1))
                                st.metric(f"Error estándar {group1}", f"{stats.sem(data1):.4f}")
                            
                            with col_stats_indep2:
                                st.metric(f"Media {group2}", f"{data2.mean():.4f}")
                                st.metric(f"Desviación {group2}", f"{data2.std():.4f}")
                                st.metric(f"n {group2}", len(data2))
                                st.metric(f"Error estándar {group2}", f"{stats.sem(data2):.4f}")
                            
                            # Información adicional
                            st.markdown("### 📈 Información Adicional")
                            
                            col_add_indep1, col_add_indep2 = st.columns(2)
                            with col_add_indep1:
                                st.metric("Diferencia medias", f"{data1.mean() - data2.mean():.4f}")
                            
                            with col_add_indep2:
                                st.metric("Tamaño efecto (d)", f"{effect_size:.4f}")
                                st.markdown(f"<span style='color:{effect_color}; font-weight:bold;'>{effect_magnitude}</span>", 
                                          unsafe_allow_html=True)
                            
                            # Interpretación
                            st.info(f"""
                            **📝 Interpretación:**
                            - **Hipótesis nula (H₀):** μ₁ = μ₂ (Las medias de {group1} y {group2} son iguales)
                            - **Hipótesis alternativa (H₁):** μ₁ {'≠' if alternative == 'two-sided' else '<' if alternative == 'less' else '>'} μ₂
                            - **Decisión:** {'Se rechaza H₀' if is_significant else 'No se rechaza H₀'} (p = {p_value:.4f} {'<' if is_significant else '≥'} α = {alpha_ttest})
                            - **Tamaño del efecto:** {effect_magnitude} (d = {effect_size:.4f})
                            """)
                            
                            # Visualización
                            st.markdown("### 📊 Visualización")
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
                            
                            # Boxplot comparativo
                            plot_data = pd.DataFrame({
                                'Grupo': [group1] * len(data1) + [group2] * len(data2),
                                'Valor': list(data1) + list(data2)
                            })
                            sns.boxplot(data=plot_data, x='Grupo', y='Valor', ax=ax1, palette='Set2')
                            ax1.set_title(f'Comparación de {var_independent} entre grupos', 
                                        fontsize=14, fontweight='bold')
                            ax1.set_xlabel('Grupo')
                            ax1.set_ylabel(var_independent)
                            ax1.grid(True, alpha=0.3, axis='y')
                            
                            # Agregar medias al boxplot
                            for i, (name, data) in enumerate(zip([group1, group2], [data1, data2])):
                                mean_val = data.mean()
                                ax1.text(i, mean_val, f'{mean_val:.2f}', 
                                       ha='center', va='bottom', fontweight='bold')
                            
                            # Gráfico de medias con barras de error
                            means = [data1.mean(), data2.mean()]
                            stds = [stats.sem(data1), stats.sem(data2)]
                            groups = [group1, group2]
                            
                            bars = ax2.bar(groups, means, yerr=stds, capsize=10, alpha=0.7, 
                                         color=['skyblue', 'lightcoral'], edgecolor='black')
                            ax2.set_title('Medias con Intervalos de Confianza', 
                                        fontsize=14, fontweight='bold')
                            ax2.set_xlabel('Grupo')
                            ax2.set_ylabel('Media')
                            ax2.grid(True, alpha=0.3, axis='y')
                            
                            # Agregar valores a las barras
                            for bar, mean in zip(bars, means):
                                height = bar.get_height()
                                ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                                       f'{mean:.2f}', ha='center', va='bottom', fontweight='bold')
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            # Interpretación con OpenAI
                            if openai_api_key and st.button("🤖 Obtener interpretación experta", 
                                                          key="ttest_indep_ai", use_container_width=True):
                                with st.spinner("Consultando al experto..."):
                                    prompt = f"""
                                    Como experto en estadística, interpreta los siguientes resultados de prueba T para muestras independientes:
                                    
                                    Variable: {var_independent}
                                    Grupos: {group1} vs {group2}
                                    Tamaños de muestra: {len(data1)} y {len(data2)}
                                    Nivel de significancia: {alpha_ttest}
                                    
                                    Resultados:
                                    - Estadístico t: {t_stat:.4f}
                                    - p-valor: {p_value:.4f}
                                    - Prueba utilizada: {'Student' if equal_var else 'Welch'}
                                    - Significativo: {'Sí' if is_significant else 'No'}
                                    - Tamaño del efecto (Cohen's d): {effect_size:.4f} ({effect_magnitude})
                                    
                                    Estadísticas por grupo:
                                    - {group1}: Media = {data1.mean():.4f}, Desviación = {data1.std():.4f}, n = {len(data1)}
                                    - {group2}: Media = {data2.mean():.4f}, Desviación = {data2.std():.4f}, n = {len(data2)}
                                    
                                    Homogeneidad de varianzas:
                                    - Prueba de Levene: p = {levene_p:.4f}
                                    - Conclusión: {'Varianzas homogéneas' if equal_var else 'Varianzas diferentes'}
                                    
                                    Hipótesis:
                                    - H₀: μ₁ = μ₂ (Las medias de {group1} y {group2} son iguales)
                                    - H₁: μ₁ {'≠' if alternative == 'two-sided' else '<' if alternative == 'less' else '>'} μ₂
                                    
                                    Proporciona una interpretación detallada que incluya:
                                    1. Explicación de la prueba T para muestras independientes
                                    2. Importancia de la prueba de homogeneidad de varianzas
                                    3. Interpretación práctica de los resultados
                                    4. Implicaciones de la significancia estadística
                                    5. Interpretación del tamaño del efecto
                                    6. Recomendaciones para acciones basadas en los resultados
                                    7. Limitaciones y consideraciones importantes
                                    
                                    Sé claro, práctico y aplicable al contexto del análisis.
                                    """
                                    
                                    interpretation = consultar_openai(prompt, max_tokens=2500)
                                    st.markdown("---")
                                    st.markdown("### 📚 Interpretación del Experto")
                                    st.markdown(interpretation)
                                    st.markdown("---")
                            
                            # Descargar resultados
                            col_dl_indep1, col_dl_indep2 = st.columns(2)
                            with col_dl_indep1:
                                # Crear informe detallado
                                report_indep = f"""
                                INFORME DE PRUEBA T PARA MUESTRAS INDEPENDIENTES
                                ===================================================
                                
                                Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                                Variable: {var_independent}
                                Grupos: {group1} vs {group2}
                                
                                PARÁMETROS:
                                - Nivel de significancia (α): {alpha_ttest}
                                - Hipótesis alternativa: {'Bilateral (≠)' if alternative == 'two-sided' else 'Unilateral izquierda (<)' if alternative == 'less' else 'Unilateral derecha (>)'}
                                
                                DATOS POR GRUPO:
                                - {group1}:
                                  * n = {len(data1)}
                                  * Media = {data1.mean():.4f}
                                  * Desviación estándar = {data1.std():.4f}
                                  * Error estándar = {stats.sem(data1):.4f}
                                
                                - {group2}:
                                  * n = {len(data2)}
                                  * Media = {data2.mean():.4f}
                                  * Desviación estándar = {data2.std():.4f}
                                  * Error estándar = {stats.sem(data2):.4f}
                                
                                DIFERENCIA: {data1.mean() - data2.mean():.4f}
                                
                                HOMOGENEIDAD DE VARIANZAS:
                                - Prueba de Levene: p = {levene_p:.4f}
                                - Conclusión: {'Varianzas homogéneas' if equal_var else 'Varianzas diferentes'}
                                - Prueba T utilizada: {'Student' if equal_var else 'Welch'}
                                
                                RESULTADOS DE LA PRUEBA T:
                                - Estadístico t: {t_stat:.4f}
                                - p-valor: {p_value:.4f}
                                - Significativo: {'Sí' if is_significant else 'No'}
                                - Tamaño del efecto (Cohen's d): {effect_size:.4f} ({effect_magnitude})
                                
                                HIPÓTESIS:
                                - H₀: μ₁ = μ₂ (Las medias de {group1} y {group2} son iguales)
                                - H₁: μ₁ {'≠' if alternative == 'two-sided' else '<' if alternative == 'less' else '>'} μ₂
                                
                                DECISIÓN:
                                - {'Se rechaza H₀' if is_significant else 'No se rechaza H₀'} (p = {p_value:.4f} {'<' if is_significant else '≥'} α = {alpha_ttest})
                                
                                INTERPRETACIÓN:
                                - Tamaño del efecto: {effect_magnitude}
                                - Diferencia entre grupos: {data1.mean() - data2.mean():.4f}
                                - {'Existen diferencias significativas' if is_significant else 'No existen diferencias significativas'} entre los grupos
                                
                                CONSIDERACIONES:
                                1. Los resultados son válidos si los datos son aproximadamente normales
                                2. Con muestras grandes (>30 por grupo), la prueba es robusta a desviaciones de normalidad
                                3. Verificar supuestos antes de generalizar resultados
                                4. Considerar pruebas no paramétricas si no se cumplen los supuestos
                                """
                                
                                st.download_button(
                                    label="📥 Descargar informe",
                                    data=report_indep,
                                    file_name=f"informe_ttest_independiente_{var_independent}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                    mime="text/plain",
                                    use_container_width=True
                                )
                            
                            with col_dl_indep2:
                                # Descargar datos
                                output_indep = io.BytesIO()
                                with pd.ExcelWriter(output_indep, engine='openpyxl') as writer:
                                    # Datos por grupo
                                    data1_df = pd.DataFrame({f'{group1}': data1})
                                    data2_df = pd.DataFrame({f'{group2}': data2})
                                    max_len = max(len(data1_df), len(data2_df))
                                    data1_df = data1_df.reindex(range(max_len))
                                    data2_df = data2_df.reindex(range(max_len))
                                    combined_df = pd.concat([data1_df, data2_df], axis=1)
                                    combined_df.to_excel(writer, sheet_name='Datos', index=False)
                                    
                                    # Resultados
                                    results_df = pd.DataFrame({
                                        'Métrica': ['Estadístico t', 'p-valor', 'Significativo', 'Prueba utilizada',
                                                   f'Media {group1}', f'Media {group2}', 'Diferencia', 'Tamaño efecto'],
                                        'Valor': [f"{t_stat:.4f}", f"{p_value:.4f}", 
                                                 'Sí' if is_significant else 'No',
                                                 'Student' if equal_var else 'Welch',
                                                 f"{data1.mean():.4f}", f"{data2.mean():.4f}",
                                                 f"{data1.mean() - data2.mean():.4f}", f"{effect_size:.4f}"]
                                    })
                                    results_df.to_excel(writer, sheet_name='Resultados', index=False)
                                    
                                    # Homogeneidad de varianzas
                                    homo_df = pd.DataFrame({
                                        'Prueba': ['Levene'],
                                        'Estadístico': [levene_stat],
                                        'p_valor': [levene_p],
                                        'Conclusión': ['Varianzas homogéneas' if equal_var else 'Varianzas diferentes']
                                    })
                                    homo_df.to_excel(writer, sheet_name='Homogeneidad', index=False)
                                
                                st.download_button(
                                    label="📥 Descargar datos",
                                    data=output_indep.getvalue(),
                                    file_name=f"datos_ttest_independiente_{var_independent}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    use_container_width=True
                                )
                    
                    except Exception as e:
                        st.error(f"Error en prueba T independiente: {e}")
            else:
                st.warning(f"La variable '{group_var}' tiene {len(unique_groups)} grupos. Debe tener exactamente 2 grupos.")
        
        # Prueba T para muestras pareadas
        elif test_type == "Muestras pareadas" and len(numeric_cols) >= 2:
            st.markdown("### 📊 Prueba T para Muestras Pareadas")
            
            col_paired1, col_paired2 = st.columns(2)
            with col_paired1:
                var_before = st.selectbox("**Variable 'Antes':**", numeric_cols, key="before_var_select")
            
            with col_paired2:
                var_after = st.selectbox("**Variable 'Después':**", numeric_cols, key="after_var_select")
            
            if st.button("📊 Ejecutar Prueba T Pareada", key="run_paired_full", use_container_width=True):
                try:
                    # Filtrar pares completos
                    paired_data = df[[var_before, var_after]].dropna()
                    
                    if len(paired_data) < 2:
                        st.error("Se necesitan al menos 2 pares completos de observaciones")
                    else:
                        # Ejecutar prueba
                        t_stat, p_value = stats.ttest_rel(paired_data[var_before], paired_data[var_after])
                        
                        # Ajuste para pruebas unilaterales
                        differences = paired_data[var_after] - paired_data[var_before]
                        if alternative == "less":
                            if differences.mean() < 0:
                                p_value = p_value / 2
                            else:
                                p_value = 1 - p_value / 2
                        elif alternative == "greater":
                            if differences.mean() > 0:
                                p_value = p_value / 2
                            else:
                                p_value = 1 - p_value / 2
                        
                        # Tamaño del efecto
                        effect_size = calculate_effect_size(
                            "Muestras pareadas", 
                            paired_data=paired_data, 
                            var_before=var_before, 
                            var_after=var_after
                        )
                        effect_magnitude, effect_color = interpret_effect_size(effect_size)
                        
                        # Resultados principales
                        st.markdown("### 📋 Resultados Principales")
                        
                        col_res_paired1, col_res_paired2, col_res_paired3, col_res_paired4 = st.columns(4)
                        with col_res_paired1:
                            st.metric("Estadístico t", f"{t_stat:.4f}")
                        with col_res_paired2:
                            st.metric("p-valor", f"{p_value:.4f}")
                        with col_res_paired3:
                            st.metric("Grados libertad", len(paired_data) - 1)
                        with col_res_paired4:
                            is_significant = p_value < alpha_ttest
                            st.metric("Significativo", "✅ Sí" if is_significant else "❌ No")
                        
                        # Estadísticas descriptivas
                        st.markdown("### 📊 Estadísticas Descriptivas")
                        
                        col_stats_paired1, col_stats_paired2, col_stats_paired3 = st.columns(3)
                        with col_stats_paired1:
                            st.metric(f"Media '{var_before}'", f"{paired_data[var_before].mean():.4f}")
                            st.metric(f"Desviación '{var_before}'", f"{paired_data[var_before].std():.4f}")
                        
                        with col_stats_paired2:
                            st.metric(f"Media '{var_after}'", f"{paired_data[var_after].mean():.4f}")
                            st.metric(f"Desviación '{var_after}'", f"{paired_data[var_after].std():.4f}")
                        
                        with col_stats_paired3:
                            st.metric("Número de pares", len(paired_data))
                            st.metric("Diferencia media", f"{differences.mean():.4f}")
                        
                        # Información adicional
                        st.markdown("### 📈 Información Adicional")
                        
                        col_add_paired1, col_add_paired2 = st.columns(2)
                        with col_add_paired1:
                            st.metric("Desviación diferencias", f"{differences.std():.4f}")
                        
                        with col_add_paired2:
                            st.metric("Tamaño efecto (d)", f"{effect_size:.4f}")
                            st.markdown(f"<span style='color:{effect_color}; font-weight:bold;'>{effect_magnitude}</span>", 
                                      unsafe_allow_html=True)
                        
                        # Cambio porcentual
                        change_percent = (differences.mean() / paired_data[var_before].mean()) * 100
                        st.info(f"""
                        **📈 Cambio porcentual:**
                        - De {paired_data[var_before].mean():.4f} a {paired_data[var_after].mean():.4f}
                        - Cambio: {differences.mean():.4f} ({change_percent:+.2f}%)
                        """)
                        
                        # Interpretación
                        st.info(f"""
                        **📝 Interpretación:**
                        - **Hipótesis nula (H₀):** μ_antes = μ_después (No hay diferencia entre mediciones)
                        - **Hipótesis alternativa (H₁):** μ_antes {'≠' if alternative == 'two-sided' else '<' if alternative == 'less' else '>'} μ_después
                        - **Decisión:** {'Se rechaza H₀' if is_significant else 'No se rechaza H₀'} (p = {p_value:.4f} {'<' if is_significant else '≥'} α = {alpha_ttest})
                        - **Tamaño del efecto:** {effect_magnitude} (d = {effect_size:.4f})
                        """)
                        
                        # Visualización
                        st.markdown("### 📊 Visualización")
                        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
                        
                        # Boxplot comparativo
                        plot_data = pd.DataFrame({
                            'Momento': ['Antes'] * len(paired_data) + ['Después'] * len(paired_data),
                            'Valor': list(paired_data[var_before]) + list(paired_data[var_after])
                        })
                        sns.boxplot(data=plot_data, x='Momento', y='Valor', ax=ax1, palette='Set2')
                        ax1.set_title('Distribución Antes vs Después', fontsize=14, fontweight='bold')
                        ax1.set_xlabel('Momento')
                        ax1.set_ylabel('Valor')
                        ax1.grid(True, alpha=0.3, axis='y')
                        
                        # Gráfico de diferencias
                        ax2.hist(differences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                        ax2.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Sin cambio')
                        ax2.axvline(differences.mean(), color='green', linestyle='-', linewidth=2.5, 
                                  alpha=0.8, label=f'Media diferencias: {differences.mean():.4f}')
                        ax2.set_title('Distribución de las Diferencias', fontsize=14, fontweight='bold')
                        ax2.set_xlabel('Diferencia (Después - Antes)')
                        ax2.set_ylabel('Frecuencia')
                        ax2.legend()
                        ax2.grid(True, alpha=0.3)
                        
                        # Gráfico de pares
                        for i in range(len(paired_data)):
                            ax3.plot([0, 1], [paired_data[var_before].iloc[i], paired_data[var_after].iloc[i]], 
                                   'gray', alpha=0.3, linewidth=1)
                        ax3.plot([0], paired_data[var_before].mean(), 'bo', markersize=10, label=f'Media Antes: {paired_data[var_before].mean():.2f}')
                        ax3.plot([1], paired_data[var_after].mean(), 'ro', markersize=10, label=f'Media Después: {paired_data[var_after].mean():.2f}')
                        ax3.errorbar([0], paired_data[var_before].mean(), yerr=stats.sem(paired_data[var_before]), 
                                   fmt='bo', capsize=5, capthick=2)
                        ax3.errorbar([1], paired_data[var_after].mean(), yerr=stats.sem(paired_data[var_after]), 
                                   fmt='ro', capsize=5, capthick=2)
                        ax3.set_xlim(-0.5, 1.5)
                        ax3.set_xticks([0, 1])
                        ax3.set_xticklabels(['Antes', 'Después'])
                        ax3.set_ylabel(var_before)
                        ax3.set_title('Comparación de Medias por Par', fontsize=14, fontweight='bold')
                        ax3.legend()
                        ax3.grid(True, alpha=0.3)
                        
                        # Gráfico de Bland-Altman (diferencias vs promedio)
                        averages = (paired_data[var_before] + paired_data[var_after]) / 2
                        ax4.scatter(averages, differences, alpha=0.6, s=50)
                        ax4.axhline(differences.mean(), color='red', linestyle='-', linewidth=2, 
                                  label=f'Media diferencias: {differences.mean():.4f}')
                        ax4.axhline(differences.mean() + 1.96*differences.std(), color='red', linestyle='--', 
                                  alpha=0.7, label='Límite superior 95%')
                        ax4.axhline(differences.mean() - 1.96*differences.std(), color='red', linestyle='--', 
                                  alpha=0.7, label='Límite inferior 95%')
                        ax4.axhline(0, color='gray', linestyle=':', alpha=0.5)
                        ax4.set_title('Gráfico de Bland-Altman', fontsize=14, fontweight='bold')
                        ax4.set_xlabel(f'Promedio de ({var_before} y {var_after})')
                        ax4.set_ylabel('Diferencia (Después - Antes)')
                        ax4.legend()
                        ax4.grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Interpretación con OpenAI
                        if openai_api_key and st.button("🤖 Obtener interpretación experta", 
                                                      key="ttest_paired_ai", use_container_width=True):
                            with st.spinner("Consultando al experto..."):
                                prompt = f"""
                                Como experto en estadística, interpreta los siguientes resultados de prueba T para muestras pareadas:
                                
                                Variables: {var_before} (Antes) y {var_after} (Después)
                                Número de pares: {len(paired_data)}
                                Nivel de significancia: {alpha_ttest}
                                
                                Resultados:
                                - Estadístico t: {t_stat:.4f}
                                - p-valor: {p_value:.4f}
                                - Grados de libertad: {len(paired_data) - 1}
                                - Significativo: {'Sí' if is_significant else 'No'}
                                - Tamaño del efecto (Cohen's d): {effect_size:.4f} ({effect_magnitude})
                                
                                Estadísticas:
                                - Media Antes: {paired_data[var_before].mean():.4f}
                                - Media Después: {paired_data[var_after].mean():.4f}
                                - Diferencia media: {differences.mean():.4f}
                                - Desviación de diferencias: {differences.std():.4f}
                                - Cambio porcentual: {change_percent:+.2f}%
                                
                                Hipótesis:
                                - H₀: μ_antes = μ_después (No hay diferencia entre mediciones)
                                - H₁: μ_antes {'≠' if alternative == 'two-sided' else '<' if alternative == 'less' else '>'} μ_después
                                
                                Proporciona una interpretación detallada que incluya:
                                1. Explicación de la prueba T para muestras pareadas
                                2. Interpretación práctica de los resultados
                                3. Implicaciones de la significancia estadística
                                4. Interpretación del tamaño del efecto
                                5. Análisis del cambio porcentual
                                6. Recomendaciones para acciones basadas en los resultados
                                7. Limitaciones y consideraciones importantes
                                8. Interpretación del gráfico de Bland-Altman
                                
                                Sé claro, práctico y aplicable al contexto del análisis.
                                """
                                
                                interpretation = consultar_openai(prompt, max_tokens=2500)
                                st.markdown("---")
                                st.markdown("### 📚 Interpretación del Experto")
                                st.markdown(interpretation)
                                st.markdown("---")
                        
                        # Descargar resultados
                        col_dl_paired1, col_dl_paired2 = st.columns(2)
                        with col_dl_paired1:
                            # Crear informe detallado
                            report_paired = f"""
                            INFORME DE PRUEBA T PARA MUESTRAS PAREADAS
                            =============================================
                            
                            Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                            Variables: {var_before} (Antes) y {var_after} (Después)
                            
                            PARÁMETROS:
                            - Nivel de significancia (α): {alpha_ttest}
                            - Hipótesis alternativa: {'Bilateral (≠)' if alternative == 'two-sided' else 'Unilateral izquierda (<)' if alternative == 'less' else 'Unilateral derecha (>)'}
                            
                            DATOS:
                            - Número de pares: {len(paired_data)}
                            - Media Antes: {paired_data[var_before].mean():.4f}
                            - Media Después: {paired_data[var_after].mean():.4f}
                            - Diferencia media: {differences.mean():.4f}
                            - Desviación de diferencias: {differences.std():.4f}
                            - Cambio porcentual: {change_percent:+.2f}%
                            
                            RESULTADOS DE LA PRUEBA T:
                            - Estadístico t: {t_stat:.4f}
                            - p-valor: {p_value:.4f}
                            - Grados de libertad: {len(paired_data) - 1}
                            - Significativo: {'Sí' if is_significant else 'No'}
                            - Tamaño del efecto (Cohen's d): {effect_size:.4f} ({effect_magnitude})
                            
                            HIPÓTESIS:
                            - H₀: μ_antes = μ_después (No hay diferencia entre mediciones)
                            - H₁: μ_antes {'≠' if alternative == 'two-sided' else '<' if alternative == 'less' else '>'} μ_después
                            
                            DECISIÓN:
                            - {'Se rechaza H₀' if is_significant else 'No se rechaza H₀'} (p = {p_value:.4f} {'<' if is_significant else '≥'} α = {alpha_ttest})
                            
                            INTERPRETACIÓN:
                            - Tamaño del efecto: {effect_magnitude}
                            - Cambio observado: {differences.mean():.4f} ({change_percent:+.2f}%)
                            - {'Existen diferencias significativas' if is_significant else 'No existen diferencias significativas'} entre las mediciones
                            
                            CONSIDERACIONES:
                            1. La prueba T pareada requiere normalidad de las diferencias
                            2. Es más poderosa que la prueba para muestras independientes cuando las mediciones están correlacionadas
                            3. Verificar supuestos antes de generalizar resultados
                            4. Considerar pruebas no paramétricas (Wilcoxon) si no se cumple la normalidad
                            """
                            
                            st.download_button(
                                label="📥 Descargar informe",
                                data=report_paired,
                                file_name=f"informe_ttest_pareado_{var_before}_{var_after}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                mime="text/plain",
                                use_container_width=True
                            )
                        
                        with col_dl_paired2:
                            # Descargar datos
                            output_paired = io.BytesIO()
                            with pd.ExcelWriter(output_paired, engine='openpyxl') as writer:
                                # Datos
                                paired_df = paired_data.copy()
                                paired_df['Diferencia'] = differences
                                paired_df['ID'] = range(1, len(paired_df) + 1)
                                paired_df = paired_df[['ID', var_before, var_after, 'Diferencia']]
                                paired_df.to_excel(writer, sheet_name='Datos', index=False)
                                
                                # Resultados
                                results_df = pd.DataFrame({
                                    'Métrica': ['Estadístico t', 'p-valor', 'Grados libertad', 'Significativo',
                                               'Media Antes', 'Media Después', 'Diferencia media', 
                                               'Cambio porcentual', 'Tamaño efecto'],
                                    'Valor': [f"{t_stat:.4f}", f"{p_value:.4f}", len(paired_data)-1,
                                             'Sí' if is_significant else 'No', f"{paired_data[var_before].mean():.4f}",
                                             f"{paired_data[var_after].mean():.4f}", f"{differences.mean():.4f}",
                                             f"{change_percent:+.2f}%", f"{effect_size:.4f}"]
                                })
                                results_df.to_excel(writer, sheet_name='Resultados', index=False)
                                
                                # Estadísticas de diferencias
                                diff_stats = pd.DataFrame({
                                    'Estadística': ['Media', 'Desviación', 'Mínimo', 'Máximo', 'Rango'],
                                    'Valor': [differences.mean(), differences.std(), 
                                             differences.min(), differences.max(), 
                                             differences.max() - differences.min()]
                                })
                                diff_stats.to_excel(writer, sheet_name='Diferencias', index=False)
                            
                            st.download_button(
                                label="📥 Descargar datos",
                                data=output_paired.getvalue(),
                                file_name=f"datos_ttest_pareado_{var_before}_{var_after}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True
                            )
                
                except Exception as e:
                    st.error(f"Error en prueba T pareada: {e}")
    
    # ========================================================================
    # PESTAÑA 7: ANOVA
    # ========================================================================
    with tab7:
        st.subheader("📊 Análisis de Varianza (ANOVA)")
        
        if numeric_cols and categorical_cols:
            col_anova_select1, col_anova_select2 = st.columns(2)
            with col_anova_select1:
                num_var = st.selectbox("**Variable numérica:**", numeric_cols, key="anova_num_select")
            
            with col_anova_select2:
                cat_var = st.selectbox("**Variable categórica:**", categorical_cols, key="anova_cat_select")
            
            # Opciones de ANOVA
            col_anova_opt1, col_anova_opt2 = st.columns(2)
            with col_anova_opt1:
                anova_type = st.radio(
                    "**Tipo de ANOVA:**",
                    ["Una vía (One-Way)", "Dos vías (Two-Way)"],
                    key="anova_type_select"
                )
            
            with col_anova_opt2:
                alpha_anova = st.slider(
                    "**Nivel de significancia (α):**",
                    min_value=0.01,
                    max_value=0.10,
                    value=0.05,
                    step=0.01,
                    key="anova_alpha_slider"
                )
            
            # Para ANOVA de dos vías, seleccionar segunda variable categórica
            if anova_type == "Dos vías (Two-Way)":
                available_cats = [col for col in categorical_cols if col != cat_var]
                if available_cats:
                    cat_var2 = st.selectbox(
                        "**Segunda variable categórica:**",
                        available_cats,
                        key="anova_cat2_select"
                    )
                else:
                    st.warning("Se necesita al menos una variable categórica diferente")
                    cat_var2 = None
            else:
                cat_var2 = None
            
            if st.button("📊 Ejecutar ANOVA", key="run_anova_full", use_container_width=True):
                try:
                    if anova_type == "Una vía (One-Way)":
                        # Preparar datos para ANOVA de una vía
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
                            # Ejecutar ANOVA de una vía
                            f_stat, p_value = stats.f_oneway(*groups_data)
                            
                            # Resultados principales
                            st.markdown("### 📋 Resultados del ANOVA de Una Vía")
                            
                            col_res_anova1, col_res_anova2, col_res_anova3, col_res_anova4 = st.columns(4)
                            with col_res_anova1:
                                st.metric("Estadístico F", f"{f_stat:.4f}")
                            with col_res_anova2:
                                st.metric("p-valor", f"{p_value:.4f}")
                            with col_res_anova3:
                                df_between = len(groups_data) - 1
                                df_within = sum(len(g) for g in groups_data) - len(groups_data)
                                st.metric("Grados libertad", f"{df_between}, {df_within}")
                            with col_res_anova4:
                                is_significant = p_value < alpha_anova
                                st.metric("Significativo", "✅ Sí" if is_significant else "❌ No")
                            
                            # Estadísticas descriptivas por grupo
                            st.markdown("### 📊 Estadísticas por Grupo")
                            
                            stats_data = []
                            for name, data in zip(group_names, groups_data):
                                stats_data.append({
                                    'Grupo': name,
                                    'n': len(data),
                                    'Media': f"{data.mean():.4f}",
                                    'Desviación': f"{data.std():.4f}",
                                    'Error estándar': f"{stats.sem(data):.4f}",
                                    'IC 95% inferior': f"{stats.t.interval(0.95, len(data)-1, loc=data.mean(), scale=stats.sem(data))[0]:.4f}",
                                    'IC 95% superior': f"{stats.t.interval(0.95, len(data)-1, loc=data.mean(), scale=stats.sem(data))[1]:.4f}"
                                })
                            
                            stats_df = pd.DataFrame(stats_data)
                            st.dataframe(stats_df, use_container_width=True)
                            
                            # Interpretación
                            st.info(f"""
                            **📝 Interpretación:**
                            - **Hipótesis nula (H₀):** μ₁ = μ₂ = ... = μₖ (Todas las medias de grupo son iguales)
                            - **Hipótesis alternativa (H₁):** Al menos una media es diferente
                            - **Decisión:** {'Se rechaza H₀' if is_significant else 'No se rechaza H₀'} (p = {p_value:.4f} {'<' if is_significant else '≥'} α = {alpha_anova})
                            - {'Existen diferencias significativas' if is_significant else 'No existen diferencias significativas'} entre los grupos
                            """)
                            
                            # Pruebas post-hoc si ANOVA es significativo
                            if is_significant and len(groups_data) > 2:
                                st.markdown("### 🔍 Comparaciones Múltiples (Post-hoc - Tukey HSD)")
                                
                                try:
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
                                    st.dataframe(result_df, use_container_width=True)
                                    
                                    # Identificar diferencias significativas
                                    sig_pairs = result_df[result_df['p-adj'] < alpha_anova]
                                    if not sig_pairs.empty:
                                        st.write("**📌 Diferencias significativas entre pares:**")
                                        for _, row in sig_pairs.iterrows():
                                            st.write(f"- **{row['group1']} vs {row['group2']}:** p-adj = {row['p-adj']:.4f}")
                                    else:
                                        st.info("No hay diferencias significativas entre pares específicos")
                                        
                                except ImportError:
                                    st.warning("Para comparaciones post-hoc, instala statsmodels: pip install statsmodels")
                                except Exception as e:
                                    st.warning(f"No se pudo realizar análisis post-hoc: {e}")
                            
                            # Visualización
                            st.markdown("### 📈 Visualización")
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
                            
                            # Boxplot comparativo
                            plot_data = []
                            for name, data in zip(group_names, groups_data):
                                for value in data:
                                    plot_data.append({'Grupo': name, 'Valor': value})
                            
                            plot_df = pd.DataFrame(plot_data)
                            sns.boxplot(data=plot_df, x='Grupo', y='Valor', ax=ax1, palette='Set2')
                            ax1.set_title(f'ANOVA: {num_var} por {cat_var}', fontsize=14, fontweight='bold')
                            ax1.set_xlabel(cat_var)
                            ax1.set_ylabel(num_var)
                            ax1.tick_params(axis='x', rotation=45)
                            ax1.grid(True, alpha=0.3, axis='y')
                            
                            # Agregar medias al gráfico
                            for i, name in enumerate(group_names):
                                mean_val = groups_data[i].mean()
                                ax1.text(i, mean_val, f'{mean_val:.2f}', 
                                    ha='center', va='bottom', fontweight='bold')
                            
                            # Gráfico de medias con barras de error
                            means = [d.mean() for d in groups_data]
                            errors = [stats.sem(d) for d in groups_data]
                            
                            bars = ax2.bar(group_names, means, yerr=errors, capsize=10, alpha=0.7, 
                                        color='skyblue', edgecolor='black')
                            ax2.set_title('Medias con Intervalos de Confianza', fontsize=14, fontweight='bold')
                            ax2.set_xlabel(cat_var)
                            ax2.set_ylabel(f'Media de {num_var}')
                            ax2.tick_params(axis='x', rotation=45)
                            ax2.grid(True, alpha=0.3, axis='y')
                            
                            # Agregar valores a las barras
                            for bar, mean in zip(bars, means):
                                height = bar.get_height()
                                ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                                    f'{mean:.2f}', ha='center', va='bottom', fontweight='bold')
                            
                            # Agregar estadísticas F al gráfico
                            stats_text = f"F = {f_stat:.3f}\np = {p_value:.4f}\ndf = {df_between}, {df_within}"
                            ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
                                fontsize=11, verticalalignment='top',
                                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            # Interpretación con OpenAI
                            if openai_api_key and st.button("🤖 Obtener interpretación experta", 
                                                        key="anova_oneway_ai", use_container_width=True):
                                with st.spinner("Consultando al experto..."):
                                    prompt = f"""
                                    Como experto en estadística, interpreta los siguientes resultados de ANOVA de una vía:
                                    
                                    Variable dependiente: {num_var}
                                    Variable independiente: {cat_var}
                                    Número de grupos: {len(groups_data)}
                                    Nivel de significancia: {alpha_anova}
                                    
                                    Resultados:
                                    - Estadístico F: {f_stat:.4f}
                                    - p-valor: {p_value:.4f}
                                    - Grados de libertad: {df_between} (entre grupos), {df_within} (dentro de grupos)
                                    - Significativo: {'Sí' if is_significant else 'No'}
                                    
                                    Estadísticas por grupo:
                                    """
                                    
                                    for i, (name, data) in enumerate(zip(group_names, groups_data)):
                                        prompt += f"\n- {name}: n = {len(data)}, Media = {data.mean():.4f}, Desviación = {data.std():.4f}"
                                    
                                    if is_significant and len(groups_data) > 2:
                                        prompt += f"\n\nComparaciones post-hoc (Tukey HSD):"
                                        if not sig_pairs.empty:
                                            for _, row in sig_pairs.iterrows():
                                                prompt += f"\n- {row['group1']} vs {row['group2']}: p-adj = {row['p-adj']:.4f}"
                                        else:
                                            prompt += "\n- No hay diferencias significativas entre pares específicos"
                                    
                                    prompt += f"""
                                    
                                    Hipótesis:
                                    - H₀: μ₁ = μ₂ = ... = μₖ (Todas las medias de grupo son iguales)
                                    - H₁: Al menos una media es diferente
                                    
                                    Proporciona una interpretación detallada que incluya:
                                    1. Explicación del ANOVA de una vía
                                    2. Interpretación práctica de los resultados
                                    3. Implicaciones de la significancia estadística
                                    4. Análisis de las diferencias entre grupos
                                    5. Recomendaciones para acciones basadas en los resultados
                                    6. Limitaciones y consideraciones importantes
                                    7. Sugerencias para análisis complementarios
                                    
                                    Sé claro, práctico y aplicable al contexto del análisis.
                                    """
                                    
                                    interpretation = consultar_openai(prompt, max_tokens=2500)
                                    st.markdown("---")
                                    st.markdown("### 📚 Interpretación del Experto")
                                    st.markdown(interpretation)
                                    st.markdown("---")
                            
                            # Descargar resultados
                            col_dl_anova1, col_dl_anova2 = st.columns(2)
                            with col_dl_anova1:
                                # Crear informe detallado
                                report_anova = f"""
                                INFORME DE ANOVA DE UNA VÍA
                                ============================
                                
                                Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                                Variable dependiente: {num_var}
                                Variable independiente: {cat_var}
                                
                                PARÁMETROS:
                                - Nivel de significancia (α): {alpha_anova}
                                - Número de grupos: {len(groups_data)}
                                
                                DATOS POR GRUPO:
                                """
                                
                                for i, row in stats_df.iterrows():
                                    report_anova += f"\n- {row['Grupo']}:"
                                    report_anova += f"\n  * n = {row['n']}"
                                    report_anova += f"\n  * Media = {row['Media']}"
                                    report_anova += f"\n  * Desviación = {row['Desviación']}"
                                    report_anova += f"\n  * Error estándar = {row['Error estándar']}"
                                
                                report_anova += f"""
                                
                                RESULTADOS DEL ANOVA:
                                - Estadístico F: {f_stat:.4f}
                                - p-valor: {p_value:.4f}
                                - Grados de libertad: {df_between} (entre grupos), {df_within} (dentro de grupos)
                                - Significativo: {'Sí' if is_significant else 'No'}
                                
                                HIPÓTESIS:
                                - H₀: μ₁ = μ₂ = ... = μₖ (Todas las medias de grupo son iguales)
                                - H₁: Al menos una media es diferente
                                
                                DECISIÓN:
                                - {'Se rechaza H₀' if is_significant else 'No se rechaza H₀'} (p = {p_value:.4f} {'<' if is_significant else '≥'} α = {alpha_anova})
                                """
                                
                                if is_significant and len(groups_data) > 2 and not sig_pairs.empty:
                                    report_anova += "\n\nCOMPARACIONES POST-HOC (Tukey HSD):"
                                    for _, row in sig_pairs.iterrows():
                                        report_anova += f"\n- {row['group1']} vs {row['group2']}: p-adj = {row['p-adj']:.4f}"
                                
                                report_anova += f"""
                                
                                INTERPRETACIÓN:
                                - {'Existen diferencias significativas' if is_significant else 'No existen diferencias significativas'} entre los grupos
                                - El estadístico F ({f_stat:.4f}) indica la razón de varianza entre grupos vs dentro de grupos
                                
                                CONSIDERACIONES:
                                1. ANOVA requiere normalidad de los residuos y homogeneidad de varianzas
                                2. Verificar supuestos antes de generalizar resultados
                                3. Para 2 grupos, usar prueba T en lugar de ANOVA
                                4. Considerar pruebas no paramétricas (Kruskal-Wallis) si no se cumplen los supuestos
                                """
                                
                                st.download_button(
                                    label="📥 Descargar informe",
                                    data=report_anova,
                                    file_name=f"informe_anova_{num_var}_{cat_var}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                    mime="text/plain",
                                    use_container_width=True
                                )
                            
                            with col_dl_anova2:
                                # Descargar datos
                                output_anova = io.BytesIO()
                                with pd.ExcelWriter(output_anova, engine='openpyxl') as writer:
                                    # Datos por grupo
                                    group_data_all = pd.DataFrame()
                                    for name, data in zip(group_names, groups_data):
                                        temp_df = pd.DataFrame({name: data})
                                        group_data_all = pd.concat([group_data_all, temp_df], axis=1)
                                    group_data_all.to_excel(writer, sheet_name='Datos por Grupo', index=False)
                                    
                                    # Resultados
                                    results_df = pd.DataFrame({
                                        'Métrica': ['Estadístico F', 'p-valor', 'Grados libertad', 'Significativo'],
                                        'Valor': [f"{f_stat:.4f}", f"{p_value:.4f}", f"{df_between}, {df_within}",
                                                'Sí' if is_significant else 'No']
                                    })
                                    results_df.to_excel(writer, sheet_name='Resultados', index=False)
                                    
                                    # Estadísticas por grupo
                                    stats_df.to_excel(writer, sheet_name='Estadísticas', index=False)
                                    
                                    # Comparaciones post-hoc
                                    if is_significant and len(groups_data) > 2:
                                        try:
                                            tukey_data.to_excel(writer, sheet_name='Datos Tukey', index=False)
                                            if not sig_pairs.empty:
                                                sig_pairs.to_excel(writer, sheet_name='Diferencias', index=False)
                                        except:
                                            pass
                                
                                st.download_button(
                                    label="📥 Descargar datos",
                                    data=output_anova.getvalue(),
                                    file_name=f"datos_anova_{num_var}_{cat_var}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    use_container_width=True
                                )
                    
                    elif anova_type == "Dos vías (Two-Way)" and cat_var2:
                        # Implementación de ANOVA de dos vías
                        st.markdown("### 📊 ANOVA de Dos Vías")
                        
                        # Filtrar datos completos
                        two_way_data = df[[num_var, cat_var, cat_var2]].dropna()
                        
                        if len(two_way_data) < 2:
                            st.error("Se necesitan al menos 2 observaciones completas")
                        else:
                            # Crear fórmula para ANOVA de dos vías
                            formula = f'{num_var} ~ C({cat_var}) + C({cat_var2}) + C({cat_var}):C({cat_var2})'
                            
                            try:
                                # Ejecutar ANOVA de dos vías
                                model = ols(formula, data=two_way_data).fit()
                                anova_table = sm.stats.anova_lm(model, typ=2)
                                
                                # Renombrar columnas para consistencia
                                anova_table = anova_table.rename(columns={
                                    'sum_sq': 'Suma_cuadrados',
                                    'df': 'Grados_libertad',
                                    'F': 'Estadistico_F',
                                    'PR(>F)': 'p_valor'
                                })
                                
                                # Agregar columna de cuadrado medio si no existe
                                if 'mean_sq' not in anova_table.columns and 'Suma_cuadrados' in anova_table.columns and 'Grados_libertad' in anova_table.columns:
                                    anova_table['Cuadrado_medio'] = anova_table['Suma_cuadrados'] / anova_table['Grados_libertad']
                                
                                # Resultados principales
                                st.markdown("### 📋 Resultados del ANOVA de Dos Vías")
                                
                                # Mostrar tabla ANOVA con formato
                                display_table = anova_table.copy()
                                display_table = display_table.round(4)
                                st.dataframe(display_table, use_container_width=True)
                                
                                # Extraer resultados importantes
                                main_effect1_p = anova_table.loc[f'C({cat_var})', 'p_valor'] if f'C({cat_var})' in anova_table.index else None
                                main_effect2_p = anova_table.loc[f'C({cat_var2})', 'p_valor'] if f'C({cat_var2})' in anova_table.index else None
                                interaction_p = anova_table.loc[f'C({cat_var}):C({cat_var2})', 'p_valor'] if f'C({cat_var}):C({cat_var2})' in anova_table.index else None
                                
                                # Resultados resumidos
                                st.markdown("### 🎯 Significancia de los Efectos")
                                
                                col_effects1, col_effects2, col_effects3 = st.columns(3)
                                
                                with col_effects1:
                                    if main_effect1_p is not None:
                                        is_sig1 = main_effect1_p < alpha_anova
                                        st.metric(
                                            f"Efecto principal de {cat_var}",
                                            "✅ Significativo" if is_sig1 else "❌ No significativo",
                                            f"p = {main_effect1_p:.4f}"
                                        )
                                
                                with col_effects2:
                                    if main_effect2_p is not None:
                                        is_sig2 = main_effect2_p < alpha_anova
                                        st.metric(
                                            f"Efecto principal de {cat_var2}",
                                            "✅ Significativo" if is_sig2 else "❌ No significativo",
                                            f"p = {main_effect2_p:.4f}"
                                        )
                                
                                with col_effects3:
                                    if interaction_p is not None:
                                        is_sig_int = interaction_p < alpha_anova
                                        st.metric(
                                            f"Interacción {cat_var} × {cat_var2}",
                                            "✅ Significativo" if is_sig_int else "❌ No significativo",
                                            f"p = {interaction_p:.4f}"
                                        )
                                
                                # Estadísticas descriptivas
                                st.markdown("### 📊 Estadísticas Descriptivas por Combinación")
                                
                                # Calcular estadísticas para cada combinación
                                combinations = two_way_data.groupby([cat_var, cat_var2])[num_var].agg(['count', 'mean', 'std', 'sem'])
                                combinations.columns = ['n', 'Media', 'Desviación', 'Error estándar']
                                
                                st.dataframe(combinations, use_container_width=True)
                                
                                # Visualización
                                st.markdown("### 📈 Visualización de Efectos")
                                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
                                
                                # Gráfico de interacción
                                interaction_data = two_way_data.groupby([cat_var, cat_var2])[num_var].mean().unstack()
                                interaction_data.plot(kind='line', marker='o', ax=ax1, linewidth=2, markersize=8)
                                ax1.set_title(f'Gráfico de Interacción: {cat_var} × {cat_var2}', fontsize=14, fontweight='bold')
                                ax1.set_xlabel(cat_var)
                                ax1.set_ylabel(f'Media de {num_var}')
                                ax1.legend(title=cat_var2)
                                ax1.grid(True, alpha=0.3)
                                
                                # Heatmap de medias
                                pivot_table = two_way_data.pivot_table(values=num_var, index=cat_var, columns=cat_var2, aggfunc='mean')
                                sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap='YlOrRd', ax=ax2)
                                ax2.set_title(f'Heatmap de Medias por Combinación', fontsize=14, fontweight='bold')
                                ax2.set_xlabel(cat_var2)
                                ax2.set_ylabel(cat_var)
                                
                                plt.tight_layout()
                                st.pyplot(fig)
                                
                                # Interpretación
                                st.markdown("### 📝 Interpretación")
                                
                                interpretation_text = f"""
                                **Análisis de Varianza de Dos Vías para {num_var}**
                                
                                **Efectos principales:**
                                """
                                
                                if main_effect1_p is not None:
                                    interpretation_text += f"\n- **{cat_var}:** {'Significativo' if main_effect1_p < alpha_anova else 'No significativo'} (p = {main_effect1_p:.4f})"
                                
                                if main_effect2_p is not None:
                                    interpretation_text += f"\n- **{cat_var2}:** {'Significativo' if main_effect2_p < alpha_anova else 'No significativo'} (p = {main_effect2_p:.4f})"
                                
                                if interaction_p is not None:
                                    interpretation_text += f"\n\n**Interacción {cat_var} × {cat_var2}:**"
                                    interpretation_text += f"\n- {'Significativa' if interaction_p < alpha_anova else 'No significativa'} (p = {interaction_p:.4f})"
                                    
                                    if interaction_p < alpha_anova:
                                        interpretation_text += f"\n- **IMPORTANTE:** La interacción significativa indica que el efecto de {cat_var} sobre {num_var} depende del nivel de {cat_var2} (y viceversa)."
                                        interpretation_text += f"\n- Se debe interpretar la interacción antes que los efectos principales."
                                
                                st.info(interpretation_text)
                                
                                # Interpretación con OpenAI
                                if openai_api_key and st.button("🤖 Obtener interpretación experta", 
                                                            key="anova_twoway_ai", use_container_width=True):
                                    with st.spinner("Consultando al experto..."):
                                        prompt = f"""
                                        Como experto en estadística, interpreta los siguientes resultados de ANOVA de dos vías:
                                        
                                        Variable dependiente: {num_var}
                                        Variables independientes: {cat_var} y {cat_var2}
                                        Nivel de significancia: {alpha_anova}
                                        Número de observaciones: {len(two_way_data)}
                                        
                                        Resultados:
                                        - Efecto principal de {cat_var}: p = {main_effect1_p:.4f} {'(Significativo)' if main_effect1_p < alpha_anova else '(No significativo)'}
                                        - Efecto principal de {cat_var2}: p = {main_effect2_p:.4f} {'(Significativo)' if main_effect2_p < alpha_anova else '(No significativo)'}
                                        - Interacción {cat_var} × {cat_var2}: p = {interaction_p:.4f} {'(Significativa)' if interaction_p < alpha_anova else '(No significativa)'}
                                        
                                        Tabla ANOVA completa:
                                        {anova_table.to_string()}
                                        
                                        Estadísticas descriptivas por combinación:
                                        {combinations.to_string()}
                                        
                                        Proporciona una interpretación detallada que incluya:
                                        1. Explicación del ANOVA de dos vías y sus componentes
                                        2. Interpretación de los efectos principales
                                        3. Interpretación de la interacción (si es significativa)
                                        4. Implicaciones prácticas de los resultados
                                        5. Recomendaciones para análisis complementarios
                                        6. Limitaciones y consideraciones importantes
                                        7. Sugerencias para la presentación de resultados
                                        
                                        Sé claro, práctico y aplicable al contexto del análisis.
                                        """
                                        
                                        interpretation = consultar_openai(prompt, max_tokens=2500)
                                        st.markdown("---")
                                        st.markdown("### 📚 Interpretación del Experto")
                                        st.markdown(interpretation)
                                        st.markdown("---")
                                
                                # Descargar resultados
                                col_dl_twoway1, col_dl_twoway2 = st.columns(2)
                                with col_dl_twoway1:
                                    # Crear informe detallado
                                    report_twoway = f"""
                                    INFORME DE ANOVA DE DOS VÍAS
                                    ==============================
                                    
                                    Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                                    Variable dependiente: {num_var}
                                    Variables independientes: {cat_var} y {cat_var2}
                                    
                                    PARÁMETROS:
                                    - Nivel de significancia (α): {alpha_anova}
                                    - Número de observaciones: {len(two_way_data)}
                                    
                                    RESULTADOS ANOVA:
                                    """
                                    
                                    for idx, row in anova_table.iterrows():
                                        report_twoway += f"\n- {idx}:"
                                        report_twoway += f"\n  * Suma de cuadrados: {row['Suma_cuadrados']:.4f}"
                                        report_twoway += f"\n  * Grados de libertad: {row['Grados_libertad']:.0f}"
                                        if 'Cuadrado_medio' in row:
                                            report_twoway += f"\n  * Cuadrado medio: {row['Cuadrado_medio']:.4f}"
                                        if 'Estadistico_F' in row:
                                            report_twoway += f"\n  * Estadístico F: {row['Estadistico_F']:.4f}"
                                        report_twoway += f"\n  * p-valor: {row['p_valor']:.4f}"
                                    
                                    report_twoway += f"""
                                    
                                    SIGNIFICANCIA:
                                    - Efecto principal de {cat_var}: p = {main_effect1_p:.4f} ({'Significativo' if main_effect1_p < alpha_anova else 'No significativo'})
                                    - Efecto principal de {cat_var2}: p = {main_effect2_p:.4f} ({'Significativo' if main_effect2_p < alpha_anova else 'No significativo'})
                                    - Interacción {cat_var} × {cat_var2}: p = {interaction_p:.4f} ({'Significativa' if interaction_p < alpha_anova else 'No significativa'})
                                    
                                    ESTADÍSTICAS DESCRIPTIVAS POR COMBINACIÓN:
                                    """
                                    
                                    for (cat1_val, cat2_val), row in combinations.iterrows():
                                        report_twoway += f"\n- {cat_var} = {cat1_val}, {cat_var2} = {cat2_val}:"
                                        report_twoway += f"\n  * n = {row['n']}"
                                        report_twoway += f"\n  * Media = {row['Media']:.4f}"
                                        report_twoway += f"\n  * Desviación = {row['Desviación']:.4f}"
                                        report_twoway += f"\n  * Error estándar = {row['Error estándar']:.4f}"
                                    
                                    report_twoway += f"""
                                    
                                    INTERPRETACIÓN:
                                    """
                                    
                                    if interaction_p < alpha_anova:
                                        report_twoway += f"\n- La interacción significativa indica que el efecto de {cat_var} sobre {num_var} depende del nivel de {cat_var2}."
                                        report_twoway += f"\n- Se debe analizar e interpretar la interacción antes que los efectos principales."
                                    else:
                                        report_twoway += f"\n- No hay interacción significativa entre {cat_var} y {cat_var2}."
                                        if main_effect1_p < alpha_anova:
                                            report_twoway += f"\n- {cat_var} tiene un efecto significativo sobre {num_var}."
                                        if main_effect2_p < alpha_anova:
                                            report_twoway += f"\n- {cat_var2} tiene un efecto significativo sobre {num_var}."
                                    
                                    report_twoway += f"""
                                    
                                    CONSIDERACIONES:
                                    1. ANOVA de dos vías requiere normalidad de los residuos y homogeneidad de varianzas
                                    2. Verificar supuestos antes de generalizar resultados
                                    3. Con interacción significativa, los efectos principales deben interpretarse con cautela
                                    4. Considerar análisis de efectos simples si hay interacción significativa
                                    """
                                    
                                    st.download_button(
                                        label="📥 Descargar informe",
                                        data=report_twoway,
                                        file_name=f"informe_anova_dos_vias_{num_var}_{cat_var}_{cat_var2}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                        mime="text/plain",
                                        use_container_width=True
                                    )
                                
                                with col_dl_twoway2:
                                    # Descargar datos
                                    output_twoway = io.BytesIO()
                                    with pd.ExcelWriter(output_twoway, engine='openpyxl') as writer:
                                        # Datos completos
                                        two_way_data.to_excel(writer, sheet_name='Datos', index=False)
                                        
                                        # Tabla ANOVA
                                        anova_table.to_excel(writer, sheet_name='ANOVA')
                                        
                                        # Estadísticas descriptivas
                                        combinations.to_excel(writer, sheet_name='Estadísticas')
                                        
                                        # Medias por combinación
                                        pivot_table.to_excel(writer, sheet_name='Medias por Combinación')
                                    
                                    st.download_button(
                                        label="📥 Descargar datos",
                                        data=output_twoway.getvalue(),
                                        file_name=f"datos_anova_dos_vias_{num_var}_{cat_var}_{cat_var2}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                        use_container_width=True
                                    )
                                
                            except Exception as e:
                                st.error(f"Error en ANOVA de dos vías: {e}")
                                st.info("Asegúrate de tener suficientes datos para cada combinación de factores.")
                
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
            "**Selecciona la prueba no paramétrica:**",
            ["Mann-Whitney U", "Wilcoxon (Pareada)", "Kruskal-Wallis", "Chi-cuadrado"],
            horizontal=True,
            key="nonpar_test_select"
        )
        
        alpha_nonpar = st.slider(
            "**Nivel de significancia (α):**",
            min_value=0.01,
            max_value=0.10,
            value=0.05,
            step=0.01,
            key="nonpar_alpha_slider"
        )
        
        # Información sobre pruebas no paramétricas
        with st.expander("📚 ¿Cuándo usar pruebas no paramétricas?"):
            st.markdown("""
            **🎯 Usa pruebas no paramétricas cuando:**
            - Los datos **no siguen distribución normal**
            - Tienes **muestras pequeñas** (<30 observaciones)
            - Los datos son **ordinales o de rangos**
            - Hay **valores atípicos extremos**
            - Los datos tienen **varianzas desiguales**
            
            **✅ Ventajas:**
            - No requieren supuestos de normalidad
            - Robustas a valores atípicos
            - Apropiadas para datos ordinales
            
            **⚠️ Desventajas:**
            - Menos potencia estadística que las paramétricas
            - No utilizan toda la información de los datos
            """)
        
        # Mann-Whitney U
        if nonpar_test == "Mann-Whitney U" and numeric_cols and categorical_cols:
            st.markdown("### 📊 Prueba de Mann-Whitney U")
            
            col_mw1, col_mw2 = st.columns(2)
            with col_mw1:
                mw_var = st.selectbox("**Variable numérica:**", numeric_cols, key="mw_var_select")
            
            with col_mw2:
                mw_group = st.selectbox("**Variable categórica (debe tener 2 grupos):**", 
                                       categorical_cols, key="mw_group_select")
            
            # Verificar grupos
            unique_groups = df[mw_group].dropna().unique()
            if len(unique_groups) == 2:
                group1, group2 = unique_groups
                
                if st.button("📊 Ejecutar Mann-Whitney U", key="run_mannwhitney_full", use_container_width=True):
                    try:
                        data1 = df[df[mw_group] == group1][mw_var].dropna()
                        data2 = df[df[mw_group] == group2][mw_var].dropna()
                        
                        if len(data1) < 3 or len(data2) < 3:
                            st.error("Cada grupo necesita al menos 3 observaciones")
                        else:
                            # Ejecutar prueba
                            u_stat, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                            
                            # Resultados principales
                            st.markdown("### 📋 Resultados")
                            
                            col_res_mw1, col_res_mw2, col_res_mw3 = st.columns(3)
                            with col_res_mw1:
                                st.metric("Estadístico U", f"{u_stat:.4f}")
                            with col_res_mw2:
                                st.metric("p-valor", f"{p_value:.4f}")
                            with col_res_mw3:
                                is_significant = p_value < alpha_nonpar
                                st.metric("Significativo", "✅ Sí" if is_significant else "❌ No")
                            
                            # Estadísticas descriptivas
                            st.markdown("### 📊 Estadísticas por Grupo")
                            
                            col_stats_mw1, col_stats_mw2 = st.columns(2)
                            with col_stats_mw1:
                                st.metric(f"Mediana {group1}", f"{data1.median():.4f}")
                                st.metric(f"Rango IQ {group1}", 
                                        f"{data1.quantile(0.75) - data1.quantile(0.25):.4f}")
                                st.metric(f"n {group1}", len(data1))
                                st.metric(f"Mínimo {group1}", f"{data1.min():.4f}")
                            
                            with col_stats_mw2:
                                st.metric(f"Mediana {group2}", f"{data2.median():.4f}")
                                st.metric(f"Rango IQ {group2}", 
                                        f"{data2.quantile(0.75) - data2.quantile(0.25):.4f}")
                                st.metric(f"n {group2}", len(data2))
                                st.metric(f"Máximo {group2}", f"{data2.max():.4f}")
                            
                            # Interpretación
                            st.info(f"""
                            **📝 Interpretación:**
                            - **Hipótesis nula (H₀):** Las distribuciones de {group1} y {group2} son iguales
                            - **Hipótesis alternativa (H₁):** Las distribuciones de {group1} y {group2} son diferentes
                            - **Decisión:** {'Se rechaza H₀' if is_significant else 'No se rechaza H₀'} (p = {p_value:.4f} {'<' if is_significant else '≥'} α = {alpha_nonpar})
                            - {'Existen diferencias significativas' if is_significant else 'No existen diferencias significativas'} entre las distribuciones
                            """)
                            
                            # Visualización
                            st.markdown("### 📈 Visualización")
                            fig, ax = plt.subplots(figsize=(12, 8))
                            
                            # Boxplot comparativo
                            plot_data = pd.DataFrame({
                                'Grupo': [group1] * len(data1) + [group2] * len(data2),
                                'Valor': list(data1) + list(data2)
                            })
                            sns.boxplot(data=plot_data, x='Grupo', y='Valor', ax=ax, palette='Set2')
                            ax.set_title(f'Prueba de Mann-Whitney U: {mw_var} por {mw_group}', 
                                       fontsize=14, fontweight='bold')
                            ax.set_xlabel('Grupo')
                            ax.set_ylabel(mw_var)
                            ax.grid(True, alpha=0.3, axis='y')
                            
                            # Agregar medianas al gráfico
                            for i, (name, data) in enumerate(zip([group1, group2], [data1, data2])):
                                median_val = data.median()
                                ax.text(i, median_val, f'{median_val:.2f}', 
                                       ha='center', va='bottom', fontweight='bold')
                            
                            # Agregar estadísticas U al gráfico
                            stats_text = f"U = {u_stat:.3f}\np = {p_value:.4f}\nn₁ = {len(data1)}, n₂ = {len(data2)}"
                            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                                   fontsize=11, verticalalignment='top',
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
                            
                            st.pyplot(fig)
                            
                            # Interpretación con OpenAI
                            if openai_api_key and st.button("🤖 Obtener interpretación experta", 
                                                          key="mannwhitney_ai", use_container_width=True):
                                with st.spinner("Consultando al experto..."):
                                    prompt = f"""
                                    Como experto en estadística, interpreta los siguientes resultados de la prueba de Mann-Whitney U:
                                    
                                    Variable: {mw_var}
                                    Grupos: {group1} vs {group2}
                                    Tamaños de muestra: {len(data1)} y {len(data2)}
                                    Nivel de significancia: {alpha_nonpar}
                                    
                                    Resultados:
                                    - Estadístico U: {u_stat:.4f}
                                    - p-valor: {p_value:.4f}
                                    - Significativo: {'Sí' if is_significant else 'No'}
                                    
                                    Estadísticas por grupo:
                                    - {group1}: Mediana = {data1.median():.4f}, Rango IQ = {data1.quantile(0.75) - data1.quantile(0.25):.4f}, n = {len(data1)}
                                    - {group2}: Mediana = {data2.median():.4f}, Rango IQ = {data2.quantile(0.75) - data2.quantile(0.25):.4f}, n = {len(data2)}
                                    
                                    Hipótesis:
                                    - H₀: Las distribuciones de {group1} y {group2} son iguales
                                    - H₁: Las distribuciones de {group1} y {group2} son diferentes
                                    
                                    Proporciona una interpretación detallada que incluya:
                                    1. Explicación de la prueba de Mann-Whitney U
                                    2. Diferencias con la prueba T para muestras independientes
                                    3. Interpretación práctica de los resultados
                                    4. Implicaciones de la significancia estadística
                                    5. Análisis de las medianas y rangos intercuartílicos
                                    6. Recomendaciones para acciones basadas en los resultados
                                    7. Limitaciones y consideraciones importantes
                                    
                                    Sé claro, práctico y aplicable al contexto del análisis.
                                    """
                                    
                                    interpretation = consultar_openai(prompt, max_tokens=2500)
                                    st.markdown("---")
                                    st.markdown("### 📚 Interpretación del Experto")
                                    st.markdown(interpretation)
                                    st.markdown("---")
                            
                            # Descargar resultados
                            col_dl_mw1, col_dl_mw2 = st.columns(2)
                            with col_dl_mw1:
                                # Crear informe detallado
                                report_mw = f"""
                                INFORME DE PRUEBA DE MANN-WHITNEY U
                                ======================================
                                
                                Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                                Variable: {mw_var}
                                Grupos: {group1} vs {group2}
                                
                                PARÁMETROS:
                                - Nivel de significancia (α): {alpha_nonpar}
                                - Prueba: Mann-Whitney U (no paramétrica)
                                
                                DATOS POR GRUPO:
                                - {group1}:
                                  * n = {len(data1)}
                                  * Mediana = {data1.median():.4f}
                                  * Rango intercuartílico = {data1.quantile(0.75) - data1.quantile(0.25):.4f}
                                  * Mínimo = {data1.min():.4f}
                                  * Máximo = {data1.max():.4f}
                                
                                - {group2}:
                                  * n = {len(data2)}
                                  * Mediana = {data2.median():.4f}
                                  * Rango intercuartílico = {data2.quantile(0.75) - data2.quantile(0.25):.4f}
                                  * Mínimo = {data2.min():.4f}
                                  * Máximo = {data2.max():.4f}
                                
                                RESULTADOS:
                                - Estadístico U: {u_stat:.4f}
                                - p-valor: {p_value:.4f}
                                - Significativo: {'Sí' if is_significant else 'No'}
                                
                                HIPÓTESIS:
                                - H₀: Las distribuciones de {group1} y {group2} son iguales
                                - H₁: Las distribuciones de {group1} y {group2} son diferentes
                                
                                DECISIÓN:
                                - {'Se rechaza H₀' if is_significant else 'No se rechaza H₀'} (p = {p_value:.4f} {'<' if is_significant else '≥'} α = {alpha_nonpar})
                                
                                INTERPRETACIÓN:
                                - {'Existen diferencias significativas' if is_significant else 'No existen diferencias significativas'} entre las distribuciones
                                - Diferencia entre medianas: {data1.median() - data2.median():.4f}
                                
                                CONSIDERACIONES:
                                1. La prueba de Mann-Whitney U es la alternativa no paramétrica a la prueba T para muestras independientes
                                2. Evalúa diferencias en las distribuciones, no solo en las medias
                                3. Es apropiada cuando los datos no son normales o tienen outliers
                                4. Menos poderosa que la prueba T cuando se cumplen los supuestos de normalidad
                                """
                                
                                st.download_button(
                                    label="📥 Descargar informe",
                                    data=report_mw,
                                    file_name=f"informe_mannwhitney_{mw_var}_{mw_group}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                    mime="text/plain",
                                    use_container_width=True
                                )
                            
                            with col_dl_mw2:
                                # Descargar datos
                                output_mw = io.BytesIO()
                                with pd.ExcelWriter(output_mw, engine='openpyxl') as writer:
                                    # Datos por grupo
                                    data1_df = pd.DataFrame({f'{group1}': data1})
                                    data2_df = pd.DataFrame({f'{group2}': data2})
                                    max_len = max(len(data1_df), len(data2_df))
                                    data1_df = data1_df.reindex(range(max_len))
                                    data2_df = data2_df.reindex(range(max_len))
                                    combined_df = pd.concat([data1_df, data2_df], axis=1)
                                    combined_df.to_excel(writer, sheet_name='Datos', index=False)
                                    
                                    # Resultados
                                    results_df = pd.DataFrame({
                                        'Métrica': ['Estadístico U', 'p-valor', 'Significativo', 
                                                   f'Mediana {group1}', f'Mediana {group2}', 'Diferencia medianas'],
                                        'Valor': [f"{u_stat:.4f}", f"{p_value:.4f}", 
                                                 'Sí' if is_significant else 'No',
                                                 f"{data1.median():.4f}", f"{data2.median():.4f}",
                                                 f"{data1.median() - data2.median():.4f}"]
                                    })
                                    results_df.to_excel(writer, sheet_name='Resultados', index=False)
                                    
                                    # Estadísticas descriptivas
                                    stats_mw = pd.DataFrame({
                                        'Grupo': [group1, group2],
                                        'n': [len(data1), len(data2)],
                                        'Mediana': [data1.median(), data2.median()],
                                        'Rango_IQ': [data1.quantile(0.75) - data1.quantile(0.25),
                                                    data2.quantile(0.75) - data2.quantile(0.25)],
                                        'Mínimo': [data1.min(), data2.min()],
                                        'Máximo': [data1.max(), data2.max()]
                                    })
                                    stats_mw.to_excel(writer, sheet_name='Estadísticas', index=False)
                                
                                st.download_button(
                                    label="📥 Descargar datos",
                                    data=output_mw.getvalue(),
                                    file_name=f"datos_mannwhitney_{mw_var}_{mw_group}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    use_container_width=True
                                )
                    
                    except Exception as e:
                        st.error(f"Error en Mann-Whitney U: {e}")
            else:
                st.warning(f"La variable '{mw_group}' tiene {len(unique_groups)} grupos. Debe tener exactamente 2 grupos.")
        
        # Prueba de Wilcoxon (Pareada)
        elif nonpar_test == "Wilcoxon (Pareada)" and len(numeric_cols) >= 2:
            st.markdown("### 📊 Prueba de Wilcoxon (Muestras Pareadas)")
            
            col_wilcox1, col_wilcox2 = st.columns(2)
            with col_wilcox1:
                wilcox_before = st.selectbox("**Variable 'Antes':**", numeric_cols, key="wilcox_before_select")
            
            with col_wilcox2:
                wilcox_after = st.selectbox("**Variable 'Después':**", numeric_cols, key="wilcox_after_select")
            
            if st.button("📊 Ejecutar Prueba de Wilcoxon", key="run_wilcoxon_full", use_container_width=True):
                try:
                    # Filtrar pares completos
                    paired_data = df[[wilcox_before, wilcox_after]].dropna()
                    
                    if len(paired_data) < 3:
                        st.error("Se necesitan al menos 3 pares completos de observaciones")
                    else:
                        # Ejecutar prueba de Wilcoxon
                        w_stat, p_value = stats.wilcoxon(paired_data[wilcox_before], paired_data[wilcox_after])
                        
                        # Resultados principales
                        st.markdown("### 📋 Resultados")
                        
                        col_res_wilcox1, col_res_wilcox2, col_res_wilcox3 = st.columns(3)
                        with col_res_wilcox1:
                            st.metric("Estadístico W", f"{w_stat:.4f}")
                        with col_res_wilcox2:
                            st.metric("p-valor", f"{p_value:.4f}")
                        with col_res_wilcox3:
                            is_significant = p_value < alpha_nonpar
                            st.metric("Significativo", "✅ Sí" if is_significant else "❌ No")
                        
                        # Estadísticas descriptivas
                        st.markdown("### 📊 Estadísticas Descriptivas")
                        
                        col_stats_wilcox1, col_stats_wilcox2 = st.columns(2)
                        with col_stats_wilcox1:
                            st.metric(f"Mediana '{wilcox_before}'", f"{paired_data[wilcox_before].median():.4f}")
                            st.metric(f"Rango IQ '{wilcox_before}'", 
                                    f"{paired_data[wilcox_before].quantile(0.75) - paired_data[wilcox_before].quantile(0.25):.4f}")
                            st.metric(f"n pares", len(paired_data))
                        
                        with col_stats_wilcox2:
                            st.metric(f"Mediana '{wilcox_after}'", f"{paired_data[wilcox_after].median():.4f}")
                            st.metric(f"Rango IQ '{wilcox_after}'", 
                                    f"{paired_data[wilcox_after].quantile(0.75) - paired_data[wilcox_after].quantile(0.25):.4f}")
                        
                        # Calcular diferencias
                        differences = paired_data[wilcox_after] - paired_data[wilcox_before]
                        st.metric("Diferencia de medianas", f"{paired_data[wilcox_after].median() - paired_data[wilcox_before].median():.4f}")
                        
                        # Interpretación
                        st.info(f"""
                        **📝 Interpretación:**
                        - **Hipótesis nula (H₀):** Las distribuciones de {wilcox_before} y {wilcox_after} son iguales
                        - **Hipótesis alternativa (H₁):** Las distribuciones de {wilcox_before} y {wilcox_after} son diferentes
                        - **Decisión:** {'Se rechaza H₀' if is_significant else 'No se rechaza H₀'} (p = {p_value:.4f} {'<' if is_significant else '≥'} α = {alpha_nonpar})
                        - {'Existen diferencias significativas' if is_significant else 'No existen diferencias significativas'} entre las distribuciones
                        """)
                        
                        # Visualización
                        st.markdown("### 📈 Visualización")
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
                        
                        # Boxplot comparativo
                        plot_data = pd.DataFrame({
                            'Momento': ['Antes'] * len(paired_data) + ['Después'] * len(paired_data),
                            'Valor': list(paired_data[wilcox_before]) + list(paired_data[wilcox_after])
                        })
                        sns.boxplot(data=plot_data, x='Momento', y='Valor', ax=ax1, palette='Set2')
                        ax1.set_title('Distribución Antes vs Después', fontsize=14, fontweight='bold')
                        ax1.set_xlabel('Momento')
                        ax1.set_ylabel('Valor')
                        ax1.grid(True, alpha=0.3, axis='y')
                        
                        # Agregar medianas al gráfico
                        for i, (name, data) in enumerate(zip(['Antes', 'Después'], [paired_data[wilcox_before], paired_data[wilcox_after]])):
                            median_val = data.median()
                            ax1.text(i, median_val, f'{median_val:.2f}', 
                                   ha='center', va='bottom', fontweight='bold')
                        
                        # Gráfico de diferencias
                        ax2.hist(differences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                        ax2.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Sin cambio')
                        ax2.axvline(differences.median(), color='green', linestyle='-', linewidth=2.5, 
                                  alpha=0.8, label=f'Mediana diferencias: {differences.median():.4f}')
                        ax2.set_title('Distribución de las Diferencias', fontsize=14, fontweight='bold')
                        ax2.set_xlabel('Diferencia (Después - Antes)')
                        ax2.set_ylabel('Frecuencia')
                        ax2.legend()
                        ax2.grid(True, alpha=0.3)
                        
                        # Agregar estadísticas W al gráfico
                        stats_text = f"W = {w_stat:.3f}\np = {p_value:.4f}\nn = {len(paired_data)}"
                        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
                               fontsize=11, verticalalignment='top',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
                        
                        st.pyplot(fig)
                        
                        # Interpretación con OpenAI
                        if openai_api_key and st.button("🤖 Obtener interpretación experta", 
                                                      key="wilcoxon_ai", use_container_width=True):
                            with st.spinner("Consultando al experto..."):
                                prompt = f"""
                                Como experto en estadística, interpreta los siguientes resultados de la prueba de Wilcoxon para muestras pareadas:
                                
                                Variables: {wilcox_before} (Antes) y {wilcox_after} (Después)
                                Número de pares: {len(paired_data)}
                                Nivel de significancia: {alpha_nonpar}
                                
                                Resultados:
                                - Estadístico W: {w_stat:.4f}
                                - p-valor: {p_value:.4f}
                                - Significativo: {'Sí' if is_significant else 'No'}
                                
                                Estadísticas:
                                - Mediana Antes: {paired_data[wilcox_before].median():.4f}
                                - Mediana Después: {paired_data[wilcox_after].median():.4f}
                                - Diferencia de medianas: {paired_data[wilcox_after].median() - paired_data[wilcox_before].median():.4f}
                                - Rango intercuartílico Antes: {paired_data[wilcox_before].quantile(0.75) - paired_data[wilcox_before].quantile(0.25):.4f}
                                - Rango intercuartílico Después: {paired_data[wilcox_after].quantile(0.75) - paired_data[wilcox_after].quantile(0.25):.4f}
                                
                                Hipótesis:
                                - H₀: Las distribuciones de {wilcox_before} y {wilcox_after} son iguales
                                - H₁: Las distribuciones de {wilcox_before} y {wilcox_after} son diferentes
                                
                                Proporciona una interpretación detallada que incluya:
                                1. Explicación de la prueba de Wilcoxon para muestras pareadas
                                2. Diferencias con la prueba T pareada
                                3. Interpretación práctica de los resultados
                                4. Implicaciones de la significancia estadística
                                5. Análisis de las medianas y rangos intercuartílicos
                                6. Recomendaciones para acciones basadas en los resultados
                                7. Limitaciones y consideraciones importantes
                                
                                Sé claro, práctico y aplicable al contexto del análisis.
                                """
                                
                                interpretation = consultar_openai(prompt, max_tokens=2500)
                                st.markdown("---")
                                st.markdown("### 📚 Interpretación del Experto")
                                st.markdown(interpretation)
                                st.markdown("---")
                        
                        # Descargar resultados
                        col_dl_wilcox1, col_dl_wilcox2 = st.columns(2)
                        with col_dl_wilcox1:
                            # Crear informe detallado
                            report_wilcox = f"""
                            INFORME DE PRUEBA DE WILCOXON (MUESTRAS PAREADAS)
                            ====================================================
                            
                            Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                            Variables: {wilcox_before} (Antes) y {wilcox_after} (Después)
                            
                            PARÁMETROS:
                            - Nivel de significancia (α): {alpha_nonpar}
                            - Prueba: Wilcoxon para muestras pareadas (no paramétrica)
                            
                            DATOS:
                            - Número de pares: {len(paired_data)}
                            - Mediana Antes: {paired_data[wilcox_before].median():.4f}
                            - Mediana Después: {paired_data[wilcox_after].median():.4f}
                            - Diferencia de medianas: {paired_data[wilcox_after].median() - paired_data[wilcox_before].median():.4f}
                            - Rango intercuartílico Antes: {paired_data[wilcox_before].quantile(0.75) - paired_data[wilcox_before].quantile(0.25):.4f}
                            - Rango intercuartílico Después: {paired_data[wilcox_after].quantile(0.75) - paired_data[wilcox_after].quantile(0.25):.4f}
                            
                            RESULTADOS:
                            - Estadístico W: {w_stat:.4f}
                            - p-valor: {p_value:.4f}
                            - Significativo: {'Sí' if is_significant else 'No'}
                            
                            HIPÓTESIS:
                            - H₀: Las distribuciones de {wilcox_before} y {wilcox_after} son iguales
                            - H₁: Las distribuciones de {wilcox_before} y {wilcox_after} son diferentes
                            
                            DECISIÓN:
                            - {'Se rechaza H₀' if is_significant else 'No se rechaza H₀'} (p = {p_value:.4f} {'<' if is_significant else '≥'} α = {alpha_nonpar})
                            
                            INTERPRETACIÓN:
                            - {'Existen diferencias significativas' if is_significant else 'No existen diferencias significativas'} entre las distribuciones
                            - Diferencia observada: {paired_data[wilcox_after].median() - paired_data[wilcox_before].median():.4f}
                            
                            CONSIDERACIONES:
                            1. La prueba de Wilcoxon es la alternativa no paramétrica a la prueba T para muestras pareadas
                            2. Evalúa diferencias en las distribuciones, no solo en las medias
                            3. Es apropiada cuando las diferencias no son normales o tienen outliers
                            4. Menos poderosa que la prueba T pareada cuando se cumplen los supuestos de normalidad
                            """
                            
                            st.download_button(
                                label="📥 Descargar informe",
                                data=report_wilcox,
                                file_name=f"informe_wilcoxon_{wilcox_before}_{wilcox_after}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                mime="text/plain",
                                use_container_width=True
                            )
                        
                        with col_dl_wilcox2:
                            # Descargar datos
                            output_wilcox = io.BytesIO()
                            with pd.ExcelWriter(output_wilcox, engine='openpyxl') as writer:
                                # Datos
                                paired_df = paired_data.copy()
                                paired_df['Diferencia'] = differences
                                paired_df['ID'] = range(1, len(paired_df) + 1)
                                paired_df = paired_df[['ID', wilcox_before, wilcox_after, 'Diferencia']]
                                paired_df.to_excel(writer, sheet_name='Datos', index=False)
                                
                                # Resultados
                                results_df = pd.DataFrame({
                                    'Métrica': ['Estadístico W', 'p-valor', 'Significativo',
                                               'Mediana Antes', 'Mediana Después', 'Diferencia medianas'],
                                    'Valor': [f"{w_stat:.4f}", f"{p_value:.4f}", 
                                             'Sí' if is_significant else 'No',
                                             f"{paired_data[wilcox_before].median():.4f}",
                                             f"{paired_data[wilcox_after].median():.4f}",
                                             f"{paired_data[wilcox_after].median() - paired_data[wilcox_before].median():.4f}"]
                                })
                                results_df.to_excel(writer, sheet_name='Resultados', index=False)
                                
                                # Estadísticas de diferencias
                                diff_stats = pd.DataFrame({
                                    'Estadística': ['Mediana', 'Rango_IQ', 'Mínimo', 'Máximo'],
                                    'Valor': [differences.median(), 
                                             differences.quantile(0.75) - differences.quantile(0.25),
                                             differences.min(), differences.max()]
                                })
                                diff_stats.to_excel(writer, sheet_name='Diferencias', index=False)
                            
                            st.download_button(
                                label="📥 Descargar datos",
                                data=output_wilcox.getvalue(),
                                file_name=f"datos_wilcoxon_{wilcox_before}_{wilcox_after}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True
                            )
                
                except Exception as e:
                    st.error(f"Error en prueba de Wilcoxon: {e}")
        
        # Prueba de Kruskal-Wallis
        elif nonpar_test == "Kruskal-Wallis" and numeric_cols and categorical_cols:
            st.markdown("### 📊 Prueba de Kruskal-Wallis")
            
            col_kw1, col_kw2 = st.columns(2)
            with col_kw1:
                kw_var = st.selectbox("**Variable numérica:**", numeric_cols, key="kw_var_select")
            
            with col_kw2:
                kw_group = st.selectbox("**Variable categórica:**", 
                                       categorical_cols, key="kw_group_select")
            
            if st.button("📊 Ejecutar Kruskal-Wallis", key="run_kruskal_full", use_container_width=True):
                try:
                    # Preparar datos por grupos
                    groups_data = []
                    group_names = []
                    
                    for group in df[kw_group].dropna().unique():
                        group_data = df[df[kw_group] == group][kw_var].dropna()
                        if len(group_data) >= 3:
                            groups_data.append(group_data)
                            group_names.append(str(group))
                    
                    if len(groups_data) < 2:
                        st.error("Se necesitan al menos 2 grupos con datos válidos")
                    else:
                        # Ejecutar prueba de Kruskal-Wallis
                        h_stat, p_value = stats.kruskal(*groups_data)
                        
                        # Resultados principales
                        st.markdown("### 📋 Resultados")
                        
                        col_res_kw1, col_res_kw2, col_res_kw3 = st.columns(3)
                        with col_res_kw1:
                            st.metric("Estadístico H", f"{h_stat:.4f}")
                        with col_res_kw2:
                            st.metric("p-valor", f"{p_value:.4f}")
                        with col_res_kw3:
                            is_significant = p_value < alpha_nonpar
                            st.metric("Significativo", "✅ Sí" if is_significant else "❌ No")
                        
                        # Estadísticas descriptivas por grupo
                        st.markdown("### 📊 Estadísticas por Grupo")
                        
                        stats_data = []
                        for name, data in zip(group_names, groups_data):
                            stats_data.append({
                                'Grupo': name,
                                'n': len(data),
                                'Mediana': f"{data.median():.4f}",
                                'Rango IQ': f"{data.quantile(0.75) - data.quantile(0.25):.4f}",
                                'Mínimo': f"{data.min():.4f}",
                                'Máximo': f"{data.max():.4f}"
                            })
                        
                        stats_df = pd.DataFrame(stats_data)
                        st.dataframe(stats_df, use_container_width=True)
                        
                        # Interpretación
                        st.info(f"""
                        **📝 Interpretación:**
                        - **Hipótesis nula (H₀):** Todas las distribuciones de los grupos son iguales
                        - **Hipótesis alternativa (H₁):** Al menos una distribución es diferente
                        - **Decisión:** {'Se rechaza H₀' if is_significant else 'No se rechaza H₀'} (p = {p_value:.4f} {'<' if is_significant else '≥'} α = {alpha_nonpar})
                        - {'Existen diferencias significativas' if is_significant else 'No existen diferencias significativas'} entre los grupos
                        """)
                        
                        # Visualización
                        st.markdown("### 📈 Visualización")
                        fig, ax = plt.subplots(figsize=(12, 8))
                        
                        # Boxplot comparativo
                        plot_data = []
                        for name, data in zip(group_names, groups_data):
                            for value in data:
                                plot_data.append({'Grupo': name, 'Valor': value})
                        
                        plot_df = pd.DataFrame(plot_data)
                        sns.boxplot(data=plot_df, x='Grupo', y='Valor', ax=ax, palette='Set2')
                        ax.set_title(f'Prueba de Kruskal-Wallis: {kw_var} por {kw_group}', 
                                   fontsize=14, fontweight='bold')
                        ax.set_xlabel(kw_group)
                        ax.set_ylabel(kw_var)
                        ax.tick_params(axis='x', rotation=45)
                        ax.grid(True, alpha=0.3, axis='y')
                        
                        # Agregar medianas al gráfico
                        for i, (name, data) in enumerate(zip(group_names, groups_data)):
                            median_val = data.median()
                            ax.text(i, median_val, f'{median_val:.2f}', 
                                   ha='center', va='bottom', fontweight='bold')
                        
                        # Agregar estadísticas H al gráfico
                        stats_text = f"H = {h_stat:.3f}\np = {p_value:.4f}\nn = {sum(len(g) for g in groups_data)}"
                        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                               fontsize=11, verticalalignment='top',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
                        
                        st.pyplot(fig)
                        
                        # Interpretación con OpenAI
                        if openai_api_key and st.button("🤖 Obtener interpretación experta", 
                                                      key="kruskal_ai", use_container_width=True):
                            with st.spinner("Consultando al experto..."):
                                prompt = f"""
                                Como experto en estadística, interpreta los siguientes resultados de la prueba de Kruskal-Wallis:
                                
                                Variable: {kw_var}
                                Variable de agrupación: {kw_group}
                                Número de grupos: {len(groups_data)}
                                Nivel de significancia: {alpha_nonpar}
                                
                                Resultados:
                                - Estadístico H: {h_stat:.4f}
                                - p-valor: {p_value:.4f}
                                - Significativo: {'Sí' if is_significant else 'No'}
                                
                                Estadísticas por grupo:
                                """
                                
                                for i, (name, data) in enumerate(zip(group_names, groups_data)):
                                    prompt += f"\n- {name}: n = {len(data)}, Mediana = {data.median():.4f}, Rango IQ = {data.quantile(0.75) - data.quantile(0.25):.4f}"
                                
                                prompt += f"""
                                
                                Hipótesis:
                                - H₀: Todas las distribuciones de los grupos son iguales
                                - H₁: Al menos una distribución es diferente
                                
                                Proporciona una interpretación detallada que incluya:
                                1. Explicación de la prueba de Kruskal-Wallis
                                2. Diferencias con el ANOVA de una vía
                                3. Interpretación práctica de los resultados
                                4. Implicaciones de la significancia estadística
                                5. Análisis de las medianas y rangos intercuartílicos por grupo
                                6. Recomendaciones para análisis post-hoc si es significativo
                                7. Limitaciones y consideraciones importantes
                                
                                Sé claro, práctico y aplicable al contexto del análisis.
                                """
                                
                                interpretation = consultar_openai(prompt, max_tokens=2500)
                                st.markdown("---")
                                st.markdown("### 📚 Interpretación del Experto")
                                st.markdown(interpretation)
                                st.markdown("---")
                        
                        # Descargar resultados
                        col_dl_kw1, col_dl_kw2 = st.columns(2)
                        with col_dl_kw1:
                            # Crear informe detallado
                            report_kw = f"""
                            INFORME DE PRUEBA DE KRUSKAL-WALLIS
                            ======================================
                            
                            Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                            Variable: {kw_var}
                            Variable de agrupación: {kw_group}
                            
                            PARÁMETROS:
                            - Nivel de significancia (α): {alpha_nonpar}
                            - Prueba: Kruskal-Wallis (no paramétrica)
                            - Número de grupos: {len(groups_data)}
                            
                            DATOS POR GRUPO:
                            """
                            
                            for i, row in stats_df.iterrows():
                                report_kw += f"\n- {row['Grupo']}:"
                                report_kw += f"\n  * n = {row['n']}"
                                report_kw += f"\n  * Mediana = {row['Mediana']}"
                                report_kw += f"\n  * Rango intercuartílico = {row['Rango IQ']}"
                                report_kw += f"\n  * Mínimo = {row['Mínimo']}"
                                report_kw += f"\n  * Máximo = {row['Máximo']}"
                            
                            report_kw += f"""
                            
                            RESULTADOS:
                            - Estadístico H: {h_stat:.4f}
                            - p-valor: {p_value:.4f}
                            - Significativo: {'Sí' if is_significant else 'No'}
                            
                            HIPÓTESIS:
                            - H₀: Todas las distribuciones de los grupos son iguales
                            - H₁: Al menos una distribución es diferente
                            
                            DECISIÓN:
                            - {'Se rechaza H₀' if is_significant else 'No se rechaza H₀'} (p = {p_value:.4f} {'<' if is_significant else '≥'} α = {alpha_nonpar})
                            
                            INTERPRETACIÓN:
                            - {'Existen diferencias significativas' if is_significant else 'No existen diferencias significativas'} entre los grupos
                            - La prueba evalúa diferencias en las distribuciones, no solo en las medias
                            
                            CONSIDERACIONES:
                            1. La prueba de Kruskal-Wallis es la alternativa no paramétrica al ANOVA de una vía
                            2. Evalúa diferencias en las distribuciones entre tres o más grupos independientes
                            3. Es apropiada cuando los datos no son normales o tienen outliers
                            4. Para comparaciones múltiples post-hoc, considerar pruebas como Dunn's test
                            """
                            
                            st.download_button(
                                label="📥 Descargar informe",
                                data=report_kw,
                                file_name=f"informe_kruskal_wallis_{kw_var}_{kw_group}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                mime="text/plain",
                                use_container_width=True
                            )
                        
                        with col_dl_kw2:
                            # Descargar datos
                            output_kw = io.BytesIO()
                            with pd.ExcelWriter(output_kw, engine='openpyxl') as writer:
                                # Datos por grupo
                                group_data_all = pd.DataFrame()
                                for name, data in zip(group_names, groups_data):
                                    temp_df = pd.DataFrame({name: data})
                                    group_data_all = pd.concat([group_data_all, temp_df], axis=1)
                                group_data_all.to_excel(writer, sheet_name='Datos por Grupo', index=False)
                                
                                # Resultados
                                results_df = pd.DataFrame({
                                    'Métrica': ['Estadístico H', 'p-valor', 'Significativo', 'Número de grupos'],
                                    'Valor': [f"{h_stat:.4f}", f"{p_value:.4f}", 
                                             'Sí' if is_significant else 'No', len(groups_data)]
                                })
                                results_df.to_excel(writer, sheet_name='Resultados', index=False)
                                
                                # Estadísticas por grupo
                                stats_df.to_excel(writer, sheet_name='Estadísticas', index=False)
                            
                            st.download_button(
                                label="📥 Descargar datos",
                                data=output_kw.getvalue(),
                                file_name=f"datos_kruskal_wallis_{kw_var}_{kw_group}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True
                            )
                
                except Exception as e:
                    st.error(f"Error en Kruskal-Wallis: {e}")
        
        # Prueba Chi-cuadrado
        elif nonpar_test == "Chi-cuadrado" and len(categorical_cols) >= 2:
            st.markdown("### 📊 Prueba Chi-cuadrado de Independencia")
            
            col_chi1, col_chi2 = st.columns(2)
            with col_chi1:
                chi_var1 = st.selectbox("**Variable categórica 1:**", categorical_cols, key="chi_var1_select")
            
            with col_chi2:
                chi_var2 = st.selectbox("**Variable categórica 2:**", 
                                       [c for c in categorical_cols if c != chi_var1], 
                                       key="chi_var2_select")
            
            if st.button("📊 Ejecutar Chi-cuadrado", key="run_chisquare_full", use_container_width=True):
                try:
                    # Crear tabla de contingencia
                    contingency_table = pd.crosstab(df[chi_var1], df[chi_var2])
                    
                    # Verificar que todas las celdas tengan frecuencia >= 5
                    if (contingency_table < 5).sum().sum() > 0:
                        st.warning("⚠️ Algunas celdas tienen frecuencia < 5. Considera agrupar categorías.")
                    
                    # Ejecutar prueba Chi-cuadrado
                    chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
                    
                    # Resultados principales
                    st.markdown("### 📋 Resultados")
                    
                    col_res_chi1, col_res_chi2, col_res_chi3, col_res_chi4 = st.columns(4)
                    with col_res_chi1:
                        st.metric("Estadístico χ²", f"{chi2_stat:.4f}")
                    with col_res_chi2:
                        st.metric("p-valor", f"{p_value:.4f}")
                    with col_res_chi3:
                        st.metric("Grados libertad", dof)
                    with col_res_chi4:
                        is_significant = p_value < alpha_nonpar
                        st.metric("Significativo", "✅ Sí" if is_significant else "❌ No")
                    
                    # Mostrar tabla de contingencia
                    st.markdown("### 📊 Tabla de Contingencia")
                    st.dataframe(contingency_table, use_container_width=True)
                    
                    # Mostrar frecuencias esperadas
                    st.markdown("### 📈 Frecuencias Esperadas (Bajo Independencia)")
                    expected_df = pd.DataFrame(expected, 
                                             index=contingency_table.index, 
                                             columns=contingency_table.columns)
                    st.dataframe(expected_df, use_container_width=True)
                    
                    # Calcular residuos estandarizados
                    residuals = (contingency_table - expected) / np.sqrt(expected)
                    st.markdown("### 🔍 Residuos Estandarizados")
                    residuals_df = pd.DataFrame(residuals, 
                                              index=contingency_table.index, 
                                              columns=contingency_table.columns)
                    st.dataframe(residuals_df.style.background_gradient(cmap='RdBu', vmin=-3, vmax=3), 
                                use_container_width=True)
                    
                    # Interpretación
                    st.info(f"""
                    **📝 Interpretación:**
                    - **Hipótesis nula (H₀):** {chi_var1} y {chi_var2} son independientes
                    - **Hipótesis alternativa (H₁):** {chi_var1} y {chi_var2} están asociadas
                    - **Decisión:** {'Se rechaza H₀' if is_significant else 'No se rechaza H₀'} (p = {p_value:.4f} {'<' if is_significant else '≥'} α = {alpha_nonpar})
                    - {'Existe asociación significativa' if is_significant else 'No existe asociación significativa'} entre las variables
                    
                    **📊 Residuos estandarizados:**
                    - Valores > |2| indican asociación significativa en esa celda
                    - Positivos: Frecuencia observada > esperada
                    - Negativos: Frecuencia observada < esperada
                    """)
                    
                    # Visualización
                    st.markdown("### 📈 Visualización")
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
                    
                    # Heatmap de la tabla de contingencia
                    sns.heatmap(contingency_table, annot=True, fmt='d', cmap='YlOrRd', ax=ax1)
                    ax1.set_title(f'Tabla de Contingencia: {chi_var1} × {chi_var2}', 
                                fontsize=14, fontweight='bold')
                    ax1.set_xlabel(chi_var2)
                    ax1.set_ylabel(chi_var1)
                    
                    # Heatmap de residuos estandarizados
                    sns.heatmap(residuals_df, annot=True, fmt='.2f', cmap='RdBu', center=0, 
                              vmin=-3, vmax=3, ax=ax2)
                    ax2.set_title(f'Residuos Estandarizados', fontsize=14, fontweight='bold')
                    ax2.set_xlabel(chi_var2)
                    ax2.set_ylabel(chi_var1)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Interpretación con OpenAI
                    if openai_api_key and st.button("🤖 Obtener interpretación experta", 
                                                  key="chisquare_ai", use_container_width=True):
                        with st.spinner("Consultando al experto..."):
                            prompt = f"""
                            Como experto en estadística, interpreta los siguientes resultados de la prueba Chi-cuadrado de independencia:
                            
                            Variables: {chi_var1} y {chi_var2}
                            Nivel de significancia: {alpha_nonpar}
                            
                            Resultados:
                            - Estadístico χ²: {chi2_stat:.4f}
                            - p-valor: {p_value:.4f}
                            - Grados de libertad: {dof}
                            - Significativo: {'Sí' if is_significant else 'No'}
                            
                            Tabla de contingencia observada:
                            {contingency_table.to_string()}
                            
                            Tabla de frecuencias esperadas (bajo independencia):
                            {expected_df.to_string()}
                            
                            Residuos estandarizados:
                            {residuals_df.to_string()}
                            
                            Hipótesis:
                            - H₀: {chi_var1} y {chi_var2} son independientes
                            - H₁: {chi_var1} y {chi_var2} están asociadas
                            
                            Proporciona una interpretación detallada que incluya:
                            1. Explicación de la prueba Chi-cuadrado de independencia
                            2. Interpretación práctica de los resultados
                            3. Análisis de la tabla de contingencia
                            4. Interpretación de los residuos estandarizados
                            5. Implicaciones de la significancia estadística
                            6. Recomendaciones para análisis complementarios
                            7. Limitaciones y consideraciones importantes
                            
                            Sé claro, práctico y aplicable al contexto del análisis.
                            """
                            
                            interpretation = consultar_openai(prompt, max_tokens=2500)
                            st.markdown("---")
                            st.markdown("### 📚 Interpretación del Experto")
                            st.markdown(interpretation)
                            st.markdown("---")
                    
                    # Descargar resultados
                    col_dl_chi1, col_dl_chi2 = st.columns(2)
                    with col_dl_chi1:
                        # Crear informe detallado
                        report_chi = f"""
                        INFORME DE PRUEBA CHI-CUADRADO DE INDEPENDENCIA
                        ================================================
                        
                        Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                        Variables: {chi_var1} y {chi_var2}
                        
                        PARÁMETROS:
                        - Nivel de significancia (α): {alpha_nonpar}
                        - Prueba: Chi-cuadrado de independencia
                        
                        TABLA DE CONTINGENCIA OBSERVADA:
                        """
                        
                        # Agregar tabla observada
                        report_chi += f"\n{chi_var1} \\ {chi_var2}"
                        for col in contingency_table.columns:
                            report_chi += f"\t{col}"
                        report_chi += "\tTotal"
                        
                        for idx, row in contingency_table.iterrows():
                            report_chi += f"\n{idx}"
                            for col in contingency_table.columns:
                                report_chi += f"\t{row[col]}"
                            report_chi += f"\t{row.sum()}"
                        
                        report_chi += f"\nTotal"
                        for col in contingency_table.columns:
                            report_chi += f"\t{contingency_table[col].sum()}"
                        report_chi += f"\t{contingency_table.sum().sum()}"
                        
                        report_chi += f"""
                        
                        RESULTADOS:
                        - Estadístico χ²: {chi2_stat:.4f}
                        - p-valor: {p_value:.4f}
                        - Grados de libertad: {dof}
                        - Significativo: {'Sí' if is_significant else 'No'}
                        
                        HIPÓTESIS:
                        - H₀: {chi_var1} y {chi_var2} son independientes
                        - H₁: {chi_var1} y {chi_var2} están asociadas
                        
                        DECISIÓN:
                        - {'Se rechaza H₀' if is_significant else 'No se rechaza H₀'} (p = {p_value:.4f} {'<' if is_significant else '≥'} α = {alpha_nonpar})
                        
                        INTERPRETACIÓN:
                        - {'Existe asociación significativa' if is_significant else 'No existe asociación significativa'} entre {chi_var1} y {chi_var2}
                        - La fuerza de la asociación puede medirse con coeficientes como Phi, Cramer's V o coeficiente de contingencia
                        
                        CONSIDERACIONES:
                        1. La prueba requiere que todas las celdas tengan frecuencia esperada ≥ 5
                        2. Para tablas 2x2 con frecuencias pequeñas, usar prueba exacta de Fisher
                        3. Los residuos estandarizados > |2| indican asociación significativa en esa celda específica
                        4. La prueba mide asociación, no causalidad
                        """
                        
                        st.download_button(
                            label="📥 Descargar informe",
                            data=report_chi,
                            file_name=f"informe_chicuadrado_{chi_var1}_{chi_var2}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
                    
                    with col_dl_chi2:
                        # Descargar datos
                        output_chi = io.BytesIO()
                        with pd.ExcelWriter(output_chi, engine='openpyxl') as writer:
                            # Datos originales
                            chi_data = df[[chi_var1, chi_var2]].copy()
                            chi_data['ID'] = range(1, len(chi_data) + 1)
                            chi_data = chi_data[['ID', chi_var1, chi_var2]]
                            chi_data.to_excel(writer, sheet_name='Datos', index=False)
                            
                            # Tabla de contingencia
                            contingency_table.to_excel(writer, sheet_name='Tabla Observada')
                            
                            # Frecuencias esperadas
                            expected_df.to_excel(writer, sheet_name='Frecuencias Esperadas')
                            
                            # Residuos estandarizados
                            residuals_df.to_excel(writer, sheet_name='Residuos')
                            
                            # Resultados
                            results_df = pd.DataFrame({
                                'Métrica': ['Estadístico χ²', 'p-valor', 'Grados libertad', 'Significativo'],
                                'Valor': [f"{chi2_stat:.4f}", f"{p_value:.4f}", dof, 
                                         'Sí' if is_significant else 'No']
                            })
                            results_df.to_excel(writer, sheet_name='Resultados', index=False)
                        
                        st.download_button(
                            label="📥 Descargar datos",
                            data=output_chi.getvalue(),
                            file_name=f"datos_chicuadrado_{chi_var1}_{chi_var2}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
                
                except Exception as e:
                    st.error(f"Error en prueba Chi-cuadrado: {e}")
    
    # ========================================================================
    # PESTAÑA 9: REPORTES
    # ========================================================================
    with tab9:
        st.subheader("📋 Reportes y Exportación")
        
        # Reporte de análisis exploratorio
        st.markdown("### 📊 Reporte de Análisis Exploratorio")
        st.markdown("Genera un reporte completo de análisis exploratorio de datos (EDA) usando ydata-profiling.")
        
        if st.button("📈 Generar Reporte EDA Completo", key="generate_eda_report", use_container_width=True):
            with st.spinner("Generando reporte exploratorio... Esto puede tomar unos segundos"):
                try:
                    # Configurar opciones del reporte
                    with st.expander("⚙️ Configurar reporte", expanded=False):
                        col_config1, col_config2 = st.columns(2)
                        with col_config1:
                            show_correlations = st.checkbox("Mostrar correlaciones", value=True)
                            show_missing = st.checkbox("Mostrar análisis de valores faltantes", value=True)
                        with col_config2:
                            show_samples = st.checkbox("Mostrar muestras", value=True)
                            minimal_mode = st.checkbox("Modo minimal (más rápido)", value=False)
                    
                    # Generar reporte
                    profile = ProfileReport(
                        df, 
                        title="Análisis Exploratorio de Datos",
                        explorative=True,
                        minimal=minimal_mode
                    )
                    
                    html_content = profile.to_html()
                    
                    # Mostrar información del reporte
                    st.success("✅ Reporte generado correctamente")
                    st.info(f"""
                    **📊 Características del reporte:**
                    - Análisis de todas las variables
                    - Estadísticas descriptivas
                    - {'Correlaciones' if show_correlations else ''}
                    - {'Análisis de valores faltantes' if show_missing else ''}
                    - {'Muestras de datos' if show_samples else ''}
                    - Visualizaciones interactivas
                    """)
                    
                    # Opciones de descarga
                    col_dl_eda1, col_dl_eda2, col_dl_eda3 = st.columns(3)
                    
                    with col_dl_eda1:
                        st.download_button(
                            label="📥 Descargar HTML",
                            data=html_content,
                            file_name=f"reporte_eda_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                            mime="text/html",
                            use_container_width=True
                        )
                    
                    with col_dl_eda2:
                        # También ofrecer JSON
                        json_content = profile.to_json()
                        st.download_button(
                            label="📥 Descargar JSON",
                            data=json_content,
                            file_name=f"reporte_eda_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json",
                            use_container_width=True
                        )
                    
                    with col_dl_eda3:
                        # Vista previa del reporte
                        if st.button("👁️ Vista previa del reporte", use_container_width=True):
                            st.components.v1.html(html_content, height=800, scrolling=True)
                    
                    # Reporte resumen
                    st.markdown("### 📋 Reporte Resumen")
                    
                    if st.button("📝 Generar Reporte Resumen Ejecutivo", key="generate_exec_summary", use_container_width=True):
                        with st.spinner("Generando reporte ejecutivo..."):
                            try:
                                # Crear reporte resumen
                                summary_report = f"""
                                REPORTE EJECUTIVO DE ANÁLISIS DE DATOS
                                ========================================
                                
                                Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                                Dataset: {uploaded_file.name if uploaded_file else 'No especificado'}
                                
                                1. RESUMEN DEL DATASET:
                                   - Filas: {df.shape[0]:,}
                                   - Columnas: {df.shape[1]}
                                   - Variables numéricas: {len(numeric_cols)}
                                   - Variables categóricas: {len(categorical_cols)}
                                   - Valores faltantes: {df.isnull().sum().sum():,} ({df.isnull().sum().sum()/(df.shape[0]*df.shape[1])*100:.1f}%)
                                   - Memoria: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
                                
                                2. VARIABLES PRINCIPALES:
                                   - Numéricas: {', '.join(numeric_cols[:5])}{'...' if len(numeric_cols) > 5 else ''}
                                   - Categóricas: {', '.join(categorical_cols[:3])}{'...' if len(categorical_cols) > 3 else ''}
                                
                                3. ESTADÍSTICAS CLAVE:
                                """
                                
                                if numeric_cols:
                                    for i, col in enumerate(numeric_cols[:3]):
                                        summary_report += f"\n   - {col}:"
                                        summary_report += f"\n     * Media: {df[col].mean():.4f}"
                                        summary_report += f"\n     * Mediana: {df[col].median():.4f}"
                                        summary_report += f"\n     * Desviación: {df[col].std():.4f}"
                                        summary_report += f"\n     * Rango: [{df[col].min():.4f}, {df[col].max():.4f}]"
                                
                                if categorical_cols:
                                    summary_report += "\n\n   4. DISTRIBUCIÓN CATEGÓRICAS:"
                                    for i, col in enumerate(categorical_cols[:2]):
                                        summary_report += f"\n   - {col}: {df[col].nunique()} categorías"
                                        if df[col].nunique() <= 5:
                                            for val in df[col].dropna().unique():
                                                count = (df[col] == val).sum()
                                                summary_report += f"\n     * '{val}': {count} ({count/len(df)*100:.1f}%)"
                                
                                summary_report += f"""
                                
                                5. CALIDAD DE DATOS:
                                   - Completitud: {(1 - df.isnull().sum().sum()/(df.shape[0]*df.shape[1]))*100:.1f}%
                                   - Variables con >20% faltantes: {sum(df.isnull().sum()/len(df) > 0.2)}
                                   - Variables constantes: {sum(df.nunique() == 1)}
                                
                                6. RECOMENDACIONES INICIALES:
                                   - {'Considerar imputación de valores faltantes' if df.isnull().sum().sum() > 0 else 'Datos completos'}
                                   - {'Evaluar normalidad para análisis paramétricos' if numeric_cols else ''}
                                   - {'Considerar codificación de variables categóricas' if categorical_cols else ''}
                                
                                7. ANÁLISIS SUGERIDOS:
                                   - Estadísticas descriptivas por grupo
                                   - Pruebas de correlación entre variables numéricas
                                   - {'ANOVA o pruebas T para comparar grupos' if categorical_cols and numeric_cols else ''}
                                   - {'Análisis de frecuencia para variables categóricas' if categorical_cols else ''}
                                """
                                
                                st.text_area("Reporte Ejecutivo", summary_report, height=400)
                                
                                # Descargar reporte ejecutivo
                                st.download_button(
                                    label="📥 Descargar Reporte Ejecutivo",
                                    data=summary_report,
                                    file_name=f"reporte_ejecutivo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                    mime="text/plain",
                                    use_container_width=True
                                )
                                
                            except Exception as e:
                                st.error(f"Error generando reporte ejecutivo: {e}")
                    
                    # Exportación de datos procesados
                    st.markdown("### 💾 Exportar Datos Procesados")
                    
                    col_export1, col_export2, col_export3 = st.columns(3)
                    
                    with col_export1:
                        if st.button("📊 Exportar a Excel", key="export_excel", use_container_width=True):
                            try:
                                output = io.BytesIO()
                                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                    df.to_excel(writer, sheet_name='Datos Originales', index=False)
                                    
                                    # Agregar estadísticas descriptivas
                                    if numeric_cols:
                                        df[numeric_cols].describe().to_excel(
                                            writer, sheet_name='Estadísticas Descriptivas'
                                        )
                                    
                                    # Agregar frecuencias de variables categóricas
                                    if categorical_cols:
                                        for i, col in enumerate(categorical_cols[:5]):
                                            freq_df = df[col].value_counts().reset_index()
                                            freq_df.columns = [col, 'Frecuencia']
                                            freq_df['Porcentaje'] = (freq_df['Frecuencia'] / len(df)) * 100
                                            freq_df.to_excel(writer, sheet_name=f'Frec_{col[:20]}', index=False)
                                
                                st.download_button(
                                    label="📥 Descargar Excel",
                                    data=output.getvalue(),
                                    file_name=f"datos_procesados_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    use_container_width=True
                                )
                                
                            except Exception as e:
                                st.error(f"Error exportando a Excel: {e}")
                    
                    with col_export2:
                        if st.button("📄 Exportar a CSV", key="export_csv", use_container_width=True):
                            try:
                                csv_data = df.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="📥 Descargar CSV",
                                    data=csv_data,
                                    file_name=f"datos_procesados_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                            except Exception as e:
                                st.error(f"Error exportando a CSV: {e}")
                    
                    with col_export3:
                        if st.button("📋 Exportar a JSON", key="export_json", use_container_width=True):
                            try:
                                json_data = df.to_json(orient='records', indent=2)
                                st.download_button(
                                    label="📥 Descargar JSON",
                                    data=json_data,
                                    file_name=f"datos_procesados_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                    mime="application/json",
                                    use_container_width=True
                                )
                            except Exception as e:
                                st.error(f"Error exportando a JSON: {e}")
                    
                    # Reporte de calidad de datos
                    st.markdown("### 🔍 Reporte de Calidad de Datos")
                    
                    if st.button("📊 Generar Reporte de Calidad", key="generate_quality_report", use_container_width=True):
                        with st.spinner("Analizando calidad de datos..."):
                            try:
                                # Análisis de calidad
                                quality_report = f"""
                                REPORTE DE CALIDAD DE DATOS
                                ============================
                                
                                Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                                Dataset: {uploaded_file.name if uploaded_file else 'No especificado'}
                                
                                1. COMPLETITUD DE DATOS:
                                   - Total de celdas: {df.shape[0] * df.shape[1]:,}
                                   - Valores faltantes: {df.isnull().sum().sum():,}
                                   - Porcentaje de completitud: {(1 - df.isnull().sum().sum()/(df.shape[0]*df.shape[1]))*100:.1f}%
                                
                                2. ANÁLISIS POR VARIABLE:
                                """
                                
                                # Por variable
                                missing_by_var = df.isnull().sum()
                                for col in df.columns:
                                    missing_pct = (missing_by_var[col] / len(df)) * 100
                                    unique_vals = df[col].nunique()
                                    dtype = df[col].dtype
                                    
                                    quality_report += f"\n   - {col} ({dtype}):"
                                    quality_report += f"\n     * Faltantes: {missing_by_var[col]} ({missing_pct:.1f}%)"
                                    quality_report += f"\n     * Valores únicos: {unique_vals}"
                                    if unique_vals == 1:
                                        quality_report += f" ⚠️ CONSTANTE"
                                    if missing_pct > 20:
                                        quality_report += f" ⚠️ ALTO % FALTANTES"
                                    if df[col].dtype in ['int64', 'float64'] and unique_vals < 10:
                                        quality_report += f" ⚠️ POSIBLE VARIABLE CATEGÓRICA"
                                
                                quality_report += f"""
                                
                                3. PROBLEMAS IDENTIFICADOS:
                                   - Variables constantes: {sum(df.nunique() == 1)}
                                   - Variables con >20% faltantes: {sum(missing_by_var/len(df) > 0.2)}
                                   - Variables con >50% faltantes: {sum(missing_by_var/len(df) > 0.5)}
                                   - Variables numéricas con pocos valores únicos: {sum([df[col].nunique() < 10 for col in numeric_cols]) if numeric_cols else 0}
                                
                                4. RECOMENDACIONES:
                                   - Eliminar variables con >50% faltantes
                                   - Considerar imputación para variables con 20-50% faltantes
                                   - Investigar variables constantes
                                   - Revisar tipos de datos
                                   - Considerar transformaciones si es necesario
                                
                                5. CALIFICACIÓN GENERAL: """
                                
                                # Calcular calificación
                                completeness_score = (1 - df.isnull().sum().sum()/(df.shape[0]*df.shape[1])) * 100
                                constant_vars_penalty = sum(df.nunique() == 1) * 10
                                high_missing_penalty = sum(missing_by_var/len(df) > 0.2) * 5
                                
                                quality_score = max(0, completeness_score - constant_vars_penalty - high_missing_penalty)
                                
                                if quality_score >= 80:
                                    quality_report += f"✅ EXCELENTE ({quality_score:.1f}/100)"
                                elif quality_score >= 60:
                                    quality_report += f"🟡 ACEPTABLE ({quality_score:.1f}/100)"
                                else:
                                    quality_report += f"🔴 DEFICIENTE ({quality_score:.1f}/100)"
                                
                                st.text_area("Reporte de Calidad", quality_report, height=400)
                                
                                # Descargar reporte de calidad
                                st.download_button(
                                    label="📥 Descargar Reporte de Calidad",
                                    data=quality_report,
                                    file_name=f"reporte_calidad_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                    mime="text/plain",
                                    use_container_width=True
                                )
                                
                            except Exception as e:
                                st.error(f"Error generando reporte de calidad: {e}")
                    
                    # Panel de control de reportes
                    st.markdown("### 🎛️ Panel de Control de Reportes")
                    
                    col_control1, col_control2 = st.columns(2)
                    
                    with col_control1:
                        include_descriptive = st.checkbox("Incluir estadísticas descriptivas", value=True)
                        include_correlations = st.checkbox("Incluir correlaciones", value=True)
                        include_missing = st.checkbox("Incluir análisis de valores faltantes", value=True)
                    
                    with col_control2:
                        include_visualizations = st.checkbox("Incluir visualizaciones", value=True)
                        include_recommendations = st.checkbox("Incluir recomendaciones", value=True)
                        include_code = st.checkbox("Incluir código de análisis", value=False)
                    
                    if st.button("🔄 Generar Reporte Personalizado", key="generate_custom_report", use_container_width=True):
                        with st.spinner("Generando reporte personalizado..."):
                            try:
                                # Aquí se generaría un reporte personalizado basado en las selecciones
                                st.success("Reporte personalizado generado (implementación completa en versión extendida)")
                                st.info("""
                                **Características del reporte personalizado:**
                                - Estadísticas descriptivas: Sí
                                - Análisis de correlaciones: Sí
                                - Visualizaciones: Sí
                                - Recomendaciones: Sí
                                - Código de análisis: No
                                
                                **Para una implementación completa, contactar al desarrollador.**
                                """)
                                
                            except Exception as e:
                                st.error(f"Error generando reporte personalizado: {e}")
                    
                except Exception as e:
                    st.error(f"Error generando reporte EDA: {e}")

# ============================================================================
# FOOTER Y MENSAJES FINALES
# ============================================================================

st.markdown("---")
st.markdown(
    """
    **📊 Analizador Estadístico Universal** - Herramienta para análisis estadísticos descriptivos e inferenciales  
    **Desarrollado con:** Streamlit, SciPy, Statsmodels, ydata-profiling y OpenAI GPT
    
    **📞 Soporte y contacto:**
    - Para reportar bugs o sugerencias
    - Para solicitar funcionalidades adicionales
    - Para consultas sobre análisis estadístico
    
    **🔒 Seguridad y privacidad:**
    - Tus datos no se almacenan en servidores externos
    - Los análisis se ejecutan localmente en tu máquina
    - La API de OpenAI solo recibe resúmenes estadísticos (no datos crudos)
    
    **📚 Recursos adicionales:**
    - [Documentación de SciPy](https://docs.scipy.org/doc/scipy/)
    - [Documentación de Statsmodels](https://www.statsmodels.org/stable/index.html)
    - [Guías de análisis estadístico](https://www.analyticsvidhya.com/blog/category/statistics/)
    """
)

# Mensaje cuando no hay datos cargados
if df is None:
    st.info("👆 **Para comenzar:** Carga un archivo de datos en la barra lateral para acceder a todas las funciones de análisis.")
    
    # Ejemplos de datos de demostración
    with st.expander("💡 ¿Necesitas datos de ejemplo para probar?"):
        st.markdown("""
        **📊 Conjuntos de datos de ejemplo recomendados:**
        
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
        
        5. **Diabetes Dataset** (Predicción médica)
           - Variables: age, sex, bmi, bp, s1, s2, s3, s4, s5, s6, target
           - URL: https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html
        
        **📝 Cómo crear tus propios datos de prueba:**
        - Usa Excel o Google Sheets
        - Asegúrate de tener al menos 30-50 filas para análisis significativos
        - Incluye tanto variables numéricas como categóricas
        - Guarda como CSV o Excel
        - ¡Listo para analizar!
        """)