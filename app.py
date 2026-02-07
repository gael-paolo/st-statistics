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
from ydata_profiling import ProfileReport
import io
import os

# Intentar importar librerías de AI opcionales
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Configuración de la página
st.set_page_config(
    page_title="Stats Bot - Análisis Estadístico",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🤖 Stats Bot")
st.markdown("""
Esta aplicación te ayuda a realizar análisis estadísticos descriptivos e inferenciales.
Carga tus datos y consulta a la IA qué análisis realizar, luego ejecuta las funciones disponibles.
""")

# Sidebar para configuración
st.sidebar.header("🔧 Configuración")

# Configuración de API de IA
st.sidebar.subheader("🤖 Configuración de Asistente IA")

# Selector de proveedor
ai_provider = st.sidebar.radio(
    "Selecciona el proveedor de IA:",
    ["Ninguno", "Gemini", "OpenAI"],
    help="Elige qué modelo de IA usar para las recomendaciones"
)

ai_configured = False
ai_client = None

if ai_provider == "Gemini":
    if not GENAI_AVAILABLE:
        st.sidebar.error("❌ Biblioteca google-generativeai no instalada. Ejecuta: pip install google-generativeai")
    else:
        gemini_api_key = st.sidebar.text_input("API Key de Gemini:", type="password")
        if gemini_api_key:
            try:
                genai.configure(api_key=gemini_api_key)
                ai_client = genai.GenerativeModel('gemini-2.5-flash')
                ai_configured = True
                st.sidebar.success("✅ Gemini configurado correctamente")
            except Exception as e:
                st.sidebar.error(f"Error configurando Gemini: {e}")
        else:
            st.sidebar.warning("⚠️ Ingresa tu API Key de Gemini")

elif ai_provider == "OpenAI":
    if not OPENAI_AVAILABLE:
        st.sidebar.error("❌ Biblioteca openai no instalada. Ejecuta: pip install openai")
    else:
        openai_api_key = st.sidebar.text_input("API Key de OpenAI:", type="password")
        openai_model = st.sidebar.selectbox(
            "Modelo:",
            ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
            help="Selecciona el modelo de OpenAI a utilizar"
        )
        if openai_api_key:
            try:
                # Nueva API de OpenAI >= 1.0.0
                ai_client = OpenAI(api_key=openai_api_key)
                ai_configured = True
                st.sidebar.success(f"✅ OpenAI ({openai_model}) configurado correctamente")
            except Exception as e:
                st.sidebar.error(f"Error configurando OpenAI: {e}")
        else:
            st.sidebar.warning("⚠️ Ingresa tu API Key de OpenAI")

else:
    st.sidebar.info("ℹ️ Selecciona un proveedor de IA para usar el asistente")

# Función para consultar IA
def consultar_ia(prompt, max_tokens=800):
    """
    Consulta al modelo de IA configurado con respuesta limitada y concisa.
    
    Args:
        prompt: El prompt completo a enviar
        max_tokens: Límite de tokens para respuesta concisa
    
    Returns:
        Texto de respuesta o None si hay error
    """
    if not ai_configured or ai_client is None:
        return None
    
    try:
        if ai_provider == "Gemini":
            response = ai_client.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=0.3
                )
            )
            return response.text
            
        elif ai_provider == "OpenAI":
            # Nueva API de OpenAI >= 1.0.0
            response = ai_client.chat.completions.create(
                model=openai_model,
                messages=[
                    {"role": "system", "content": "Eres un experto en estadística. Responde de manera concisa, directa y práctica. Máximo 3-4 puntos clave."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.3
            )
            return response.choices[0].message.content
            
    except Exception as e:
        st.error(f"Error consultando IA: {e}")
        return None

# ASISTENTE TEÓRICO EN ESTADÍSTICA (se muestra siempre, sin necesidad de datos)
st.subheader("📚 Asistente Teórico en Estadística")
st.markdown("""
Consulta conceptos teóricos sobre métodos estadísticos, interpretación de resultados y mejores prácticas.
*No requiere que tengas datos cargados.*
""")

theory_question = st.text_area(
    "Haz tu pregunta sobre conceptos estadísticos:",
    placeholder="Ej: ¿Cuándo usar T-test vs ANOVA? ¿Qué es un p-valor? ¿Cómo interpretar correlaciones?",
    height=100,
    key="theory_question_main"
)

if st.button("Consultar teoría estadística", key="theory_consultation_main") and theory_question:
    if ai_configured:
        with st.spinner("Consultando al experto..."):
            prompt = f"""Responde breve y directamente esta pregunta de estadística:

Pregunta: {theory_question}

Instrucciones:
- Máximo 3-4 puntos clave
- Sé práctico y directo
- Incluye cuándo usarlo y un ejemplo rápido si aplica
- Máximo 300 palabras

Respuesta:"""
            
            respuesta = consultar_ia(prompt, max_tokens=600)
            if respuesta:
                st.success("📚 Respuesta:")
                st.markdown(respuesta)
    else:
        st.error("🔑 Configura una API Key en la barra lateral para usar el asistente")

# Información sobre el asistente teórico
with st.expander("💡 Ejemplos de preguntas"):
    st.markdown("""
    - ¿Cuándo usar pruebas paramétricas vs no paramétricas?
    - ¿Cómo interpretar un intervalo de confianza?
    - ¿Qué tamaño de muestra necesito?
    - ¿Cómo detectar outliers?
    - ¿Qué es la potencia estadística?
    """)

# Línea separadora
st.markdown("---")

# Carga de datos
st.sidebar.subheader("📂 Carga de Datos")
uploaded_file = st.sidebar.file_uploader("Sube tu archivo Excel o CSV", type=['xlsx', 'csv'])

@st.cache_data
def load_data(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)

# Funciones de muestreo
def generate_sample(df, sample_size, method="simple", stratify_col=None, random_state=None):
    """
    Genera un muestreo a partir de un DataFrame.
    """
    if isinstance(sample_size, float):
        if sample_size <= 0 or sample_size > 1:
            raise ValueError("Si 'sample_size' es porcentaje, debe estar entre 0 y 1.")
        sample_size = int(len(df) * sample_size)

    if sample_size <= 0 or sample_size > len(df):
        raise ValueError("Tamaño de muestra inválido.")

    if method == "simple":
        sample_df = df.sample(n=sample_size, random_state=random_state)
    elif method == "stratified":
        if stratify_col is None:
            raise ValueError("Se requiere 'stratify_col' para muestreo estratificado.")
        if stratify_col not in df.columns:
            raise ValueError(f"La columna '{stratify_col}' no existe.")
        sample_df = df.groupby(stratify_col, group_keys=False).apply(
            lambda x: x.sample(n=int(sample_size * len(x) / len(df)), random_state=random_state)
        )
    else:
        raise ValueError("Método debe ser 'simple' o 'stratified'.")

    return sample_df.reset_index(drop=True)

def calculate_sample_size(population_size, margin_of_error=0.05, confidence_level=0.95, proportion=0.5):
    """
    Calcula tamaño de muestra requerido.
    """
    if not (0 < margin_of_error < 1):
        raise ValueError("Margen de error debe estar entre 0 y 1.")
    if not (0 < confidence_level < 1):
        raise ValueError("Nivel de confianza debe estar entre 0 y 1.")
    if population_size <= 0:
        raise ValueError("Población debe ser mayor que 0.")

    z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)
    sample_size = (z_score**2 * proportion * (1 - proportion)) / (margin_of_error**2)
    adjusted_sample_size = sample_size / (1 + (sample_size - 1) / population_size)
    
    return int(np.ceil(adjusted_sample_size))

df = None
if uploaded_file is not None:
    try:
        df = load_data(uploaded_file)
        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)
        st.sidebar.success(f"✅ Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
    except Exception as e:
        st.sidebar.error(f"Error cargando archivo: {e}")

# Mostrar datos si están cargados
if df is not None:
    st.subheader("📋 Vista previa de los datos")
    st.dataframe(df.head())
    
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
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    col1, col2 = st.columns(2)
    with col1:
        if numeric_cols:
            selected_numeric = st.multiselect("Variables numéricas:", numeric_cols, 
                                            default=numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols)
        else:
            st.warning("No se encontraron variables numéricas")
    
    with col2:
        if categorical_cols:
            selected_categorical = st.multiselect("Variables categóricas:", categorical_cols, 
                                                default=categorical_cols[0] if categorical_cols else None)
        else:
            st.warning("No se encontraron variables categóricas")

    # Sección de consulta a IA para datos específicos
    st.subheader("🤖 Asistente de Análisis para tus Datos")
    st.markdown("Consulta recomendaciones específicas basadas en los datos cargados.")
    
    user_question = st.text_area(
        "Describe tu caso o pregunta qué análisis realizar:",
        placeholder="Ej: Quiero comparar grupos, analizar relaciones entre variables, identificar patrones...",
        height=80,
        key="business_question_main"
    )
    
    if st.button("Obtener recomendaciones", key="business_recommendations_main") and user_question:
        if ai_configured:
            with st.spinner("Analizando datos..."):
                prompt = f"""Análisis estadístico para dataset con {df.shape[0]} filas y {df.shape[1]} columnas.

Variables numéricas: {numeric_cols}
Variables categóricas: {categorical_cols}

Consulta del usuario: {user_question}

Recomienda MÁXIMO 3 análisis específicos de esta lista:
- Muestreo
- Descriptivos
- Normalidad
- Correlaciones
- Pruebas T
- ANOVA
- No paramétricas
- Chi-cuadrado

Para cada uno indica:
1. Qué variables usar (máx 10 palabras)
2. Qué pregunta responde (máx 15 palabras)
3. Interpretación clave esperada (máx 15 palabras)

Sé directo y práctico. Máximo 200 palabras totales."""
                
                respuesta = consultar_ia(prompt, max_tokens=500)
                if respuesta:
                    st.success("🎯 Recomendaciones:")
                    st.markdown(respuesta)
        else:
            st.error("🔑 Configura una API Key en la barra lateral")

# Sección de análisis estadísticos
if df is not None:
    st.header("📊 Análisis Estadísticos")
    
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
    
    with tab1:  # Muestreo
        st.subheader("🎯 Análisis de Muestreo")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Generar Muestra")
            
            sample_method = st.radio(
                "Método de muestreo:",
                ["simple", "stratified"],
                format_func=lambda x: "Aleatorio Simple" if x == "simple" else "Estratificado"
            )
            
            sample_size_type = st.radio(
                "Tipo de tamaño:",
                ["percentage", "absolute"],
                format_func=lambda x: "Porcentaje" if x == "percentage" else "Número absoluto"
            )
            
            if sample_size_type == "percentage":
                sample_size_input = st.slider("Porcentaje:", 1, 50, 20) / 100.0
            else:
                sample_size_input = st.number_input("Tamaño:", 1, len(df), min(100, len(df)))
            
            stratify_column = None
            if sample_method == "stratified" and categorical_cols:
                stratify_column = st.selectbox("Variable estratificación:", categorical_cols)
            
            if st.button("🎲 Generar Muestra"):
                try:
                    with st.spinner("Generando..."):
                        sample_df = generate_sample(df, sample_size_input, sample_method, 
                                                  stratify_column, random_state=42)
                    
                    st.success(f"✅ Muestra: {len(sample_df)} registros")
                    
                    col1, col2 = st.columns(2)
                    col1.metric("Tamaño", len(sample_df))
                    col2.metric("Porcentaje", f"{(len(sample_df)/len(df))*100:.1f}%")
                    
                    st.dataframe(sample_df.head())
                    
                    if sample_method == "stratified" and stratify_column:
                        st.subheader("Distribución")
                        sample_dist = sample_df[stratify_column].value_counts()
                        original_dist = df[stratify_column].value_counts()
                        
                        dist_comparison = pd.DataFrame({
                            'Original': original_dist,
                            'Muestra': sample_dist,
                            '% Original': (original_dist / len(df)) * 100,
                            '% Muestra': (sample_dist / len(sample_df)) * 100
                        })
                        st.dataframe(dist_comparison)
                    
                    # Descargar muestra
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        sample_df.to_excel(writer, index=False, sheet_name='Muestra')
                    
                    st.download_button(
                        label="📥 Descargar Excel",
                        data=output.getvalue(),
                        file_name="muestra.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    
                except Exception as e:
                    st.error(f"Error: {e}")
        
        with col2:
            st.subheader("🧮 Calcular Tamaño de Muestra")
            
            population_size = st.number_input("Población:", 1, value=len(df))
            margin_error = st.slider("Margen de error (%):", 1, 10, 5) / 100.0
            confidence_level = st.slider("Confianza (%):", 80, 99, 95) / 100.0
            proportion = st.slider("Proporción esperada (%):", 1, 99, 50) / 100.0
            
            if st.button("📐 Calcular"):
                try:
                    sample_size = calculate_sample_size(population_size, margin_error, 
                                                      confidence_level, proportion)
                    
                    st.success(f"🎯 Tamaño recomendado: **{sample_size}**")
                    
                    st.info(f"""
                    **Parámetros:**
                    - Población: {population_size:,}
                    - Margen: ±{margin_error*100:.1f}%
                    - Confianza: {confidence_level*100:.1f}%
                    """)
                    
                    if population_size == len(df):
                        coverage = (sample_size / len(df)) * 100
                        st.metric("Cobertura actual", f"{coverage:.1f}%")
                        
                        if sample_size > len(df):
                            st.warning("⚠️ Dataset actual insuficiente")
                        else:
                            st.success("✅ Dataset suficiente")
                            
                except Exception as e:
                    st.error(f"Error: {e}")
    
    with tab2:  # Descriptivos
        st.subheader("📈 Análisis Descriptivo")
        
        if numeric_cols:
            selected_var = st.selectbox("Variable:", numeric_cols, key="desc_var")
            if selected_var:
                desc_stats = df[selected_var].describe()
                st.write(desc_stats)
                
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
        
        st.subheader("Reporte Completo")
        if st.button("📊 Generar Reporte"):
            with st.spinner("Generando..."):
                try:
                    profile = ProfileReport(df, title="Profiling Report", minimal=True)
                    html_content = profile.to_html()
                    
                    st.download_button(
                        label="📥 Descargar HTML",
                        data=html_content,
                        file_name="reporte.html",
                        mime="text/html"
                    )
                    st.success("✅ Reporte listo")
                except Exception as e:
                    st.error(f"Error: {e}")
    
    with tab3:  # Normalidad
        st.subheader("🔍 Pruebas de Normalidad")
        
        if numeric_cols:
            selected_normal_var = st.selectbox("Variable:", numeric_cols, key="normal_var")
            alpha_normal = st.slider("Nivel α:", 0.01, 0.10, 0.05, key="normal_alpha")
            
            if st.button("📊 Ejecutar Pruebas"):
                try:
                    data = df[selected_normal_var].dropna()
                    n = len(data)
                    
                    if n < 3:
                        st.error("Mínimo 3 observaciones")
                    else:
                        st.info(f"**n = {n:,}**")
                        
                        # Shapiro-Wilk
                        st.markdown("#### Shapiro-Wilk")
                        if n > 5000:
                            st.warning("Muestra grande - interpretar con precaución")
                        shapiro_stat, shapiro_p = shapiro(data)
                        shapiro_normal = shapiro_p > alpha_normal
                        
                        col1, col2 = st.columns(2)
                        col1.metric("W", f"{shapiro_stat:.4f}")
                        col2.metric("p-valor", f"{shapiro_p:.4f}")
                        
                        if shapiro_normal:
                            st.success("✅ Normal")
                        else:
                            st.error("❌ No normal")
                        
                        # Anderson-Darling
                        st.markdown("#### Anderson-Darling")
                        ad_test = anderson(data, dist='norm')
                        ad_statistic = ad_test.statistic
                        
                        alpha_to_idx = {0.15: 0, 0.10: 1, 0.05: 2, 0.025: 3, 0.01: 4}
                        closest_alpha = min(alpha_to_idx.keys(), key=lambda x: abs(x - alpha_normal))
                        idx = alpha_to_idx[closest_alpha]
                        critical_value = ad_test.critical_values[idx]
                        
                        ad_normal = ad_statistic < critical_value
                        
                        col1, col2 = st.columns(2)
                        col1.metric("A-D", f"{ad_statistic:.4f}")
                        col2.metric(f"Crítico (α={closest_alpha})", f"{critical_value:.3f}")
                        
                        if ad_normal:
                            st.success("✅ Normal")
                        else:
                            st.error("❌ No normal")
                        
                        # Lilliefors
                        st.markdown("#### Lilliefors (K-S)")
                        lilliefors_stat, lilliefors_p = lilliefors(data)
                        lilliefors_normal = lilliefors_p > alpha_normal
                        
                        col1, col2 = st.columns(2)
                        col1.metric("D", f"{lilliefors_stat:.4f}")
                        col2.metric("p-valor", f"{lilliefors_p:.4f}")
                        
                        if lilliefors_normal:
                            st.success("✅ Normal")
                        else:
                            st.error("❌ No normal")
                        
                        # Conclusión
                        st.markdown("---")
                        st.subheader("🎯 Conclusión")
                        
                        results = [shapiro_normal, ad_normal, lilliefors_normal]
                        passed = sum(results)
                        
                        if passed >= 2:
                            st.success(f"✅ **NORMAL** ({passed}/3 pruebas)")
                            st.write("Usa pruebas paramétricas")
                        else:
                            st.error(f"❌ **NO NORMAL** ({passed}/3 pruebas)")
                            st.write("Considera transformaciones o pruebas no paramétricas")
                        
                        # Visualizaciones
                        st.markdown("---")
                        st.subheader("📊 Gráficos")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            fig, ax = plt.subplots()
                            sns.histplot(data, kde=True, ax=ax, stat='density')
                            mu, sigma = data.mean(), data.std()
                            x = np.linspace(data.min(), data.max(), 100)
                            ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', label='Normal teórica')
                            ax.legend()
                            st.pyplot(fig)
                        
                        with col2:
                            fig, ax = plt.subplots()
                            stats.probplot(data, dist="norm", plot=ax)
                            st.pyplot(fig)
                        
                        # Outliers
                        fig, ax = plt.subplots(figsize=(10, 2))
                        sns.boxplot(x=data, ax=ax)
                        st.pyplot(fig)
                        
                        Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
                        IQR = Q3 - Q1
                        outliers = data[(data < Q1 - 1.5*IQR) | (data > Q3 + 1.5*IQR)]
                        
                        if len(outliers) > 0:
                            st.warning(f"⚠️ {len(outliers)} outliers detectados ({len(outliers)/len(data)*100:.1f}%)")
                        
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("No hay variables numéricas")
    
    with tab4:  # Correlaciones
        st.subheader("📉 Análisis de Correlación")
        
        if len(numeric_cols) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                var1 = st.selectbox("Variable 1:", numeric_cols, key="corr_var1")
            with col2:
                var2 = st.selectbox("Variable 2:", numeric_cols, key="corr_var2")
            
            alpha_corr = st.slider("Nivel α:", 0.01, 0.10, 0.05, key="corr_alpha")
            
            if st.button("🔍 Analizar Correlación"):
                try:
                    clean_data = df[[var1, var2]].dropna()
                    
                    if len(clean_data) < 3:
                        st.error("Mínimo 3 observaciones")
                    else:
                        # Verificar normalidad
                        _, p1 = shapiro(clean_data[var1])
                        _, p2 = shapiro(clean_data[var2])
                        
                        if p1 > alpha_corr and p2 > alpha_corr:
                            corr, p_value = stats.pearsonr(clean_data[var1], clean_data[var2])
                            method = "Pearson"
                        else:
                            corr, p_value = stats.spearmanr(clean_data[var1], clean_data[var2])
                            method = "Spearman"
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Método", method)
                        col2.metric("Coeficiente", f"{corr:.4f}")
                        col3.metric("p-valor", f"{p_value:.4f}")
                        
                        # Interpretación
                        abs_corr = abs(corr)
                        if abs_corr < 0.3:
                            strength = "débil"
                        elif abs_corr < 0.7:
                            strength = "moderada"
                        else:
                            strength = "fuerte"
                        
                        direction = "positiva" if corr > 0 else "negativa"
                        
                        st.write(f"**Correlación {strength} y {direction}** entre {var1} y {var2}")
                        
                        if p_value < alpha_corr:
                            st.success("✅ Significativa")
                        else:
                            st.warning("❌ No significativa")
                        
                        # Gráfico
                        fig, ax = plt.subplots()
                        sns.scatterplot(data=clean_data, x=var1, y=var2, alpha=0.6, ax=ax)
                        
                        z = np.polyfit(clean_data[var1], clean_data[var2], 1)
                        p = np.poly1d(z)
                        ax.plot(clean_data[var1], p(clean_data[var1]), "r--", alpha=0.8)
                        
                        ax.set_title(f'{method}: r = {corr:.3f}, p = {p_value:.4f}')
                        st.pyplot(fig)
                        
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Se necesitan 2+ variables numéricas")
    
    with tab5:  # Homogeneidad
        st.subheader("⚖️ Homogeneidad de Varianzas")
        
        if numeric_cols and categorical_cols:
            col1, col2 = st.columns(2)
            with col1:
                homo_var = st.selectbox("Variable numérica:", numeric_cols, key="homo_var")
            with col2:
                homo_group = st.selectbox("Variable categórica:", categorical_cols, key="homo_group")
            
            alpha_homo = st.slider("Nivel α:", 0.01, 0.10, 0.05, key="homo_alpha")
            
            if st.button("📊 Ejecutar Pruebas"):
                try:
                    groups_data = []
                    group_names = []
                    
                    for group in df[homo_group].dropna().unique():
                        group_data = df[df[homo_group] == group][homo_var].dropna()
                        if len(group_data) >= 2:
                            groups_data.append(group_data)
                            group_names.append(group)
                    
                    if len(groups_data) < 2:
                        st.error("Mínimo 2 grupos con datos")
                    else:
                        # Levene
                        st.markdown("#### Levene (robusta)")
                        levene_stat, levene_p = stats.levene(*groups_data)
                        levene_homo = levene_p > alpha_homo
                        
                        col1, col2 = st.columns(2)
                        col1.metric("Estadístico", f"{levene_stat:.4f}")
                        col2.metric("p-valor", f"{levene_p:.4f}")
                        
                        if levene_homo:
                            st.success("✅ Varianzas homogéneas")
                        else:
                            st.error("❌ Varianzas no homogéneas")
                        
                        # Bartlett
                        st.markdown("#### Bartlett (sensible)")
                        bartlett_stat, bartlett_p = stats.bartlett(*groups_data)
                        bartlett_homo = bartlett_p > alpha_homo
                        
                        col1, col2 = st.columns(2)
                        col1.metric("Estadístico", f"{bartlett_stat:.4f}")
                        col2.metric("p-valor", f"{bartlett_p:.4f}")
                        
                        if bartlett_homo:
                            st.success("✅ Varianzas homogéneas")
                        else:
                            st.error("❌ Varianzas no homogéneas")
                        
                        # Conclusión
                        st.markdown("---")
                        if levene_homo and bartlett_homo:
                            st.success("✅ **HOMOGÉNEAS** - Usa pruebas estándar")
                        elif levene_homo:
                            st.warning("⚠️ **DUDOSAS** - Confía en Levene, usa Welch si hay dudas")
                        else:
                            st.error("❌ **NO HOMOGÉNEAS** - Usa Welch o no paramétricas")
                        
                        # Tabla de varianzas
                        variance_data = []
                        for name, data in zip(group_names, groups_data):
                            variance_data.append({
                                'Grupo': name,
                                'n': len(data),
                                'Media': f"{data.mean():.4f}",
                                'DE': f"{data.std():.4f}",
                                'Varianza': f"{data.var():.4f}"
                            })
                        
                        st.dataframe(pd.DataFrame(variance_data), hide_index=True)
                        
                        # Visualización
                        fig, ax = plt.subplots()
                        plot_data = []
                        for name, data in zip(group_names, groups_data):
                            for value in data:
                                plot_data.append({'Grupo': name, 'Valor': value})
                        
                        sns.boxplot(data=pd.DataFrame(plot_data), x='Grupo', y='Valor', ax=ax)
                        ax.set_title(f'Distribución por grupo')
                        plt.xticks(rotation=45)
                        st.pyplot(fig)
                        
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Se necesitan variables numéricas y categóricas")
    
    with tab6:  # Pruebas T
        st.subheader("✅ Pruebas T")
        
        test_type = st.radio("Tipo:", ["Una muestra", "Independientes", "Pareadas"], key="ttest_type")
        
        col_alpha, col_alt = st.columns(2)
        with col_alpha:
            alpha_ttest = st.slider("Nivel α:", 0.01, 0.10, 0.05, key="ttest_alpha")
        with col_alt:
            alternative = st.selectbox("Hipótesis:", 
                                     ["two-sided", "less", "greater"],
                                     format_func=lambda x: {"two-sided": "Bilateral", 
                                                           "less": "Unilateral <", 
                                                           "greater": "Unilateral >"}[x])
        
        def calculate_effect_size_ttest(data1, data2=None, pop_mean=0, paired=False):
            try:
                if data2 is None:  # Una muestra
                    d = (data1.mean() - pop_mean) / data1.std()
                elif paired:  # Pareada
                    diff = data1 - data2
                    d = diff.mean() / diff.std()
                else:  # Independientes
                    n1, n2 = len(data1), len(data2)
                    pooled_std = np.sqrt(((n1-1)*data1.std()**2 + (n2-1)*data2.std()**2) / (n1 + n2 - 2))
                    d = (data1.mean() - data2.mean()) / pooled_std
                return abs(d)
            except:
                return 0
        
        def interpret_effect_size(d):
            abs_d = abs(d)
            if abs_d < 0.2:
                return "Muy pequeño", "gray"
            elif abs_d < 0.5:
                return "Pequeño", "orange"
            elif abs_d < 0.8:
                return "Mediano", "blue"
            else:
                return "Grande", "green"
        
        if test_type == "Una muestra" and numeric_cols:
            var = st.selectbox("Variable:", numeric_cols)
            pop_mean = st.number_input("Media poblacional:", value=0.0)
            
            if st.button("📊 Ejecutar"):
                try:
                    data = df[var].dropna()
                    if len(data) < 2:
                        st.error("Mínimo 2 observaciones")
                    else:
                        t_stat, p_value = stats.ttest_1samp(data, pop_mean)
                        
                        # Ajuste unilateral
                        if alternative != "two-sided":
                            if (alternative == "less" and data.mean() < pop_mean) or \
                               (alternative == "greater" and data.mean() > pop_mean):
                                p_value = p_value / 2
                            else:
                                p_value = 1 - p_value / 2
                        
                        effect_size = calculate_effect_size_ttest(data, pop_mean=pop_mean)
                        magnitude, color = interpret_effect_size(effect_size)
                        
                        col1, col2 = st.columns(2)
                        col1.metric("t", f"{t_stat:.4f}")
                        col2.metric("p-valor", f"{p_value:.4f}")
                        
                        col1, col2 = st.columns(2)
                        col1.metric("Media muestral", f"{data.mean():.4f}")
                        col2.metric("Media poblacional", f"{pop_mean:.4f}")
                        
                        st.metric("Tamaño del efecto (d)", f"{effect_size:.4f}")
                        st.markdown(f"**Magnitud:** <span style='color:{color}'>{magnitude}</span>", 
                                  unsafe_allow_html=True)
                        
                        if p_value < alpha_ttest:
                            st.success("✅ Diferencia significativa")
                        else:
                            st.warning("❌ Sin diferencia significativa")
                        
                        # Gráfico
                        fig, ax = plt.subplots()
                        sns.histplot(data, kde=True, ax=ax)
                        ax.axvline(data.mean(), color='red', label=f'Media: {data.mean():.2f}')
                        ax.axvline(pop_mean, color='blue', linestyle='--', label=f'Referencia: {pop_mean}')
                        ax.legend()
                        st.pyplot(fig)
                        
                except Exception as e:
                    st.error(f"Error: {e}")
        
        elif test_type == "Independientes" and numeric_cols and categorical_cols:
            var = st.selectbox("Variable numérica:", numeric_cols)
            group_var = st.selectbox("Variable categórica (2 grupos):", categorical_cols)
            
            unique_groups = df[group_var].dropna().unique()
            if len(unique_groups) == 2:
                g1, g2 = unique_groups
                
                if st.button("📊 Ejecutar"):
                    try:
                        data1 = df[df[group_var] == g1][var].dropna()
                        data2 = df[df[group_var] == g2][var].dropna()
                        
                        if len(data1) < 2 or len(data2) < 2:
                            st.error("Mínimo 2 observaciones por grupo")
                        else:
                            # Levene para decidir
                            _, levene_p = stats.levene(data1, data2)
                            equal_var = levene_p > 0.05
                            
                            t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=equal_var)
                            
                            # Ajuste unilateral
                            if alternative != "two-sided":
                                if (alternative == "less" and data1.mean() < data2.mean()) or \
                                   (alternative == "greater" and data1.mean() > data2.mean()):
                                    p_value = p_value / 2
                                else:
                                    p_value = 1 - p_value / 2
                            
                            effect_size = calculate_effect_size_ttest(data1, data2)
                            magnitude, color = interpret_effect_size(effect_size)
                            
                            col1, col2, col3 = st.columns(3)
                            col1.metric("t", f"{t_stat:.4f}")
                            col2.metric("p-valor", f"{p_value:.4f}")
                            col3.metric("Varianzas", "Iguales" if equal_var else "Diferentes")
                            
                            col1, col2 = st.columns(2)
                            col1.metric(f"Media {g1}", f"{data1.mean():.4f}")
                            col2.metric(f"Media {g2}", f"{data2.mean():.4f}")
                            
                            st.metric("Diferencia", f"{data1.mean() - data2.mean():.4f}")
                            st.metric("Tamaño del efecto (d)", f"{effect_size:.4f}")
                            st.markdown(f"**Magnitud:** <span style='color:{color}'>{magnitude}</span>", 
                                      unsafe_allow_html=True)
                            
                            if p_value < alpha_ttest:
                                st.success("✅ Diferencia significativa entre grupos")
                            else:
                                st.warning("❌ Sin diferencia significativa")
                            
                            # Gráfico
                            fig, ax = plt.subplots()
                            plot_df = pd.DataFrame({
                                'Grupo': [g1]*len(data1) + [g2]*len(data2),
                                'Valor': list(data1) + list(data2)
                            })
                            sns.boxplot(data=plot_df, x='Grupo', y='Valor', ax=ax)
                            st.pyplot(fig)
                            
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.warning("La variable categórica debe tener exactamente 2 grupos")
        
        elif test_type == "Pareadas" and len(numeric_cols) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                var1 = st.selectbox("Antes:", numeric_cols)
            with col2:
                var2 = st.selectbox("Después:", [c for c in numeric_cols if c != var1])
            
            if st.button("📊 Ejecutar"):
                try:
                    paired_data = df[[var1, var2]].dropna()
                    
                    if len(paired_data) < 2:
                        st.error("Mínimo 2 pares completos")
                    else:
                        t_stat, p_value = stats.ttest_rel(paired_data[var1], paired_data[var2])
                        
                        # Ajuste unilateral
                        diff = paired_data[var2] - paired_data[var1]
                        if alternative != "two-sided":
                            if (alternative == "less" and diff.mean() < 0) or \
                               (alternative == "greater" and diff.mean() > 0):
                                p_value = p_value / 2
                            else:
                                p_value = 1 - p_value / 2
                        
                        effect_size = calculate_effect_size_ttest(paired_data[var1], 
                                                                 paired_data[var2], 
                                                                 paired=True)
                        magnitude, color = interpret_effect_size(effect_size)
                        
                        col1, col2 = st.columns(2)
                        col1.metric("t", f"{t_stat:.4f}")
                        col2.metric("p-valor", f"{p_value:.4f}")
                        
                        col1, col2 = st.columns(2)
                        col1.metric(f"Media {var1}", f"{paired_data[var1].mean():.4f}")
                        col2.metric(f"Media {var2}", f"{paired_data[var2].mean():.4f}")
                        
                        st.metric("Diferencia media", f"{diff.mean():.4f}")
                        st.metric("Tamaño del efecto (d)", f"{effect_size:.4f}")
                        st.markdown(f"**Magnitud:** <span style='color:{color}'>{magnitude}</span>", 
                                  unsafe_allow_html=True)
                        
                        if p_value < alpha_ttest:
                            st.success("✅ Diferencia significativa")
                        else:
                            st.warning("❌ Sin diferencia significativa")
                        
                        # Gráficos
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                        
                        plot_df = pd.DataFrame({
                            'Momento': ['Antes']*len(paired_data) + ['Después']*len(paired_data),
                            'Valor': list(paired_data[var1]) + list(paired_data[var2])
                        })
                        sns.boxplot(data=plot_df, x='Momento', y='Valor', ax=ax1)
                        
                        sns.histplot(diff, kde=True, ax=ax2)
                        ax2.axvline(0, color='red', linestyle='--')
                        
                        st.pyplot(fig)
                        
                except Exception as e:
                    st.error(f"Error: {e}")
    
    with tab7:  # ANOVA
        st.subheader("📊 ANOVA")
        
        if numeric_cols and categorical_cols:
            anova_type = st.radio("Tipo:", ["Una vía", "Dos vías"])
            anova_var = st.selectbox("Variable numérica:", numeric_cols)
            alpha_anova = st.slider("Nivel α:", 0.01, 0.10, 0.05)
            
            if anova_type == "Una vía":
                anova_group = st.selectbox("Variable categórica:", categorical_cols)
            else:
                col1, col2 = st.columns(2)
                with col1:
                    group1 = st.selectbox("Factor 1:", categorical_cols)
                with col2:
                    group2 = st.selectbox("Factor 2:", [c for c in categorical_cols if c != group1])
            
            if st.button("📊 Ejecutar ANOVA"):
                try:
                    if anova_type == "Una vía":
                        groups_data = []
                        group_names = []
                        
                        for group in df[anova_group].dropna().unique():
                            data = df[df[anova_group] == group][anova_var].dropna()
                            if len(data) >= 2:
                                groups_data.append(data)
                                group_names.append(str(group))
                        
                        if len(groups_data) < 2:
                            st.error("Mínimo 2 grupos")
                        else:
                            f_stat, p_value = stats.f_oneway(*groups_data)
                            
                            col1, col2 = st.columns(2)
                            col1.metric("F", f"{f_stat:.4f}")
                            col2.metric("p-valor", f"{p_value:.4f}")
                            
                            # Tabla descriptiva
                            desc_data = []
                            for name, data in zip(group_names, groups_data):
                                desc_data.append({
                                    'Grupo': name,
                                    'n': len(data),
                                    'Media': f"{data.mean():.4f}",
                                    'DE': f"{data.std():.4f}"
                                })
                            st.dataframe(pd.DataFrame(desc_data), hide_index=True)
                            
                            if p_value < alpha_anova:
                                st.success("✅ Diferencias significativas entre grupos")
                                
                                # Tukey HSD
                                try:
                                    tukey_data = df[[anova_var, anova_group]].dropna()
                                    tukey = pairwise_tukeyhsd(tukey_data[anova_var], 
                                                             tukey_data[anova_group], 
                                                             alpha=alpha_anova)
                                    
                                    result_df = pd.DataFrame(data=tukey._results_table.data[1:], 
                                                           columns=tukey._results_table.data[0])
                                    st.subheader("Comparaciones post-hoc (Tukey)")
                                    st.dataframe(result_df)
                                    
                                    sig_pairs = result_df[result_df['p-adj'] < alpha_anova]
                                    if len(sig_pairs) > 0:
                                        st.write("**Diferencias significativas:**")
                                        for _, row in sig_pairs.iterrows():
                                            st.write(f"- {row['group1']} vs {row['group2']}")
                                except Exception as e:
                                    st.warning(f"No se pudo calcular Tukey: {e}")
                            else:
                                st.warning("❌ Sin diferencias significativas")
                            
                            # Gráfico
                            fig, ax = plt.subplots()
                            plot_data = []
                            for name, data in zip(group_names, groups_data):
                                for val in data:
                                    plot_data.append({'Grupo': name, 'Valor': val})
                            
                            sns.boxplot(data=pd.DataFrame(plot_data), x='Grupo', y='Valor', ax=ax)
                            plt.xticks(rotation=45)
                            st.pyplot(fig)
                    
                    else:  # Dos vías
                        anova_data = df[[anova_var, group1, group2]].dropna()
                        
                        if len(anova_data) == 0:
                            st.error("Sin datos válidos")
                        else:
                            formula = f'{anova_var} ~ C({group1}) * C({group2})'
                            model = ols(formula, data=anova_data).fit()
                            anova_table = sm.stats.anova_lm(model, typ=2)
                            
                            st.dataframe(anova_table)
                            
                            # Resumen de significancia
                            st.subheader("Resultados")
                            for idx in anova_table.index:
                                if idx != 'Residual':
                                    p_val = anova_table.loc[idx, 'PR(>F)']
                                    sig = "✅ Significativo" if p_val < alpha_anova else "❌ No significativo"
                                    st.write(f"**{idx}:** p = {p_val:.4f} - {sig}")
                            
                            # Gráficos
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                            sns.boxplot(data=anova_data, x=group1, y=anova_var, ax=ax1)
                            sns.boxplot(data=anova_data, x=group2, y=anova_var, ax=ax2)
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Se necesitan variables numéricas y categóricas")
    
    with tab8:  # No paramétricas
        st.subheader("🔄 Pruebas No Paramétricas")
        
        test = st.radio("Prueba:", 
                       ["Mann-Whitney U", "Wilcoxon", "Kruskal-Wallis", "Chi-cuadrado", "Welch"])
        
        alpha_np = st.slider("Nivel α:", 0.01, 0.10, 0.05)
        
        if test == "Mann-Whitney U" and numeric_cols and categorical_cols:
            var = st.selectbox("Variable:", numeric_cols)
            group_var = st.selectbox("Grupo (2 categorías):", categorical_cols)
            
            groups = df[group_var].dropna().unique()
            if len(groups) == 2:
                g1, g2 = groups
                
                if st.button("📊 Ejecutar"):
                    try:
                        d1 = df[df[group_var] == g1][var].dropna()
                        d2 = df[df[group_var] == g2][var].dropna()
                        
                        if len(d1) < 3 or len(d2) < 3:
                            st.error("Mínimo 3 por grupo")
                        else:
                            u_stat, p_value = stats.mannwhitneyu(d1, d2, alternative='two-sided')
                            
                            col1, col2 = st.columns(2)
                            col1.metric("U", f"{u_stat:.4f}")
                            col2.metric("p-valor", f"{p_value:.4f}")
                            
                            col1, col2 = st.columns(2)
                            col1.metric(f"Mediana {g1}", f"{d1.median():.4f}")
                            col2.metric(f"Mediana {g2}", f"{d2.median():.4f}")
                            
                            if p_value < alpha_np:
                                st.success("✅ Distribuciones diferentes")
                            else:
                                st.warning("❌ Sin diferencias significativas")
                            
                            fig, ax = plt.subplots()
                            plot_df = pd.DataFrame({
                                'Grupo': [g1]*len(d1) + [g2]*len(d2),
                                'Valor': list(d1) + list(d2)
                            })
                            sns.boxplot(data=plot_df, x='Grupo', y='Valor', ax=ax)
                            st.pyplot(fig)
                            
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.warning("Se necesitan exactamente 2 grupos")
        
        elif test == "Wilcoxon" and len(numeric_cols) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                v1 = st.selectbox("Antes:", numeric_cols)
            with col2:
                v2 = st.selectbox("Después:", [c for c in numeric_cols if c != v1])
            
            if st.button("📊 Ejecutar"):
                try:
                    paired = df[[v1, v2]].dropna()
                    
                    if len(paired) < 3:
                        st.error("Mínimo 3 pares")
                    else:
                        diff = paired[v2] - paired[v1]
                        w_stat, p_value = stats.wilcoxon(diff)
                        
                        col1, col2 = st.columns(2)
                        col1.metric("W", f"{w_stat:.4f}")
                        col2.metric("p-valor", f"{p_value:.4f}")
                        
                        col1, col2 = st.columns(2)
                        col1.metric(f"Mediana {v1}", f"{paired[v1].median():.4f}")
                        col2.metric(f"Mediana {v2}", f"{paired[v2].median():.4f}")
                        
                        if p_value < alpha_np:
                            st.success("✅ Diferencia significativa")
                        else:
                            st.warning("❌ Sin diferencia significativa")
                        
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                        
                        plot_df = pd.DataFrame({
                            'Momento': ['Antes']*len(paired) + ['Después']*len(paired),
                            'Valor': list(paired[v1]) + list(paired[v2])
                        })
                        sns.boxplot(data=plot_df, x='Momento', y='Valor', ax=ax1)
                        
                        sns.histplot(diff, kde=True, ax=ax2)
                        ax2.axvline(0, color='red', linestyle='--')
                        
                        st.pyplot(fig)
                        
                except Exception as e:
                    st.error(f"Error: {e}")
        
        elif test == "Kruskal-Wallis" and numeric_cols and categorical_cols:
            var = st.selectbox("Variable:", numeric_cols)
            group_var = st.selectbox("Grupo:", categorical_cols)
            
            if st.button("📊 Ejecutar"):
                try:
                    groups_data = []
                    group_names = []
                    
                    for group in df[group_var].dropna().unique():
                        data = df[df[group_var] == group][var].dropna()
                        if len(data) >= 3:
                            groups_data.append(data)
                            group_names.append(group)
                    
                    if len(groups_data) < 2:
                        st.error("Mínimo 2 grupos con 3+ observaciones")
                    else:
                        h_stat, p_value = stats.kruskal(*groups_data)
                        
                        col1, col2 = st.columns(2)
                        col1.metric("H", f"{h_stat:.4f}")
                        col2.metric("p-valor", f"{p_value:.4f}")
                        
                        desc_data = []
                        for name, data in zip(group_names, groups_data):
                            desc_data.append({
                                'Grupo': name,
                                'n': len(data),
                                'Mediana': f"{data.median():.4f}",
                                'RIQ': f"{data.quantile(0.75) - data.quantile(0.25):.4f}"
                            })
                        st.dataframe(pd.DataFrame(desc_data), hide_index=True)
                        
                        if p_value < alpha_np:
                            st.success("✅ Al menos un grupo diferente")
                        else:
                            st.warning("❌ Sin diferencias significativas")
                        
                        fig, ax = plt.subplots()
                        plot_data = []
                        for name, data in zip(group_names, groups_data):
                            for val in data:
                                plot_data.append({'Grupo': name, 'Valor': val})
                        
                        sns.boxplot(data=pd.DataFrame(plot_data), x='Grupo', y='Valor', ax=ax)
                        plt.xticks(rotation=45)
                        st.pyplot(fig)
                        
                except Exception as e:
                    st.error(f"Error: {e}")
        
        elif test == "Chi-cuadrado" and len(categorical_cols) >= 2:
            c1 = st.selectbox("Variable 1:", categorical_cols)
            c2 = st.selectbox("Variable 2:", [c for c in categorical_cols if c != c1])
            
            if st.button("📊 Ejecutar"):
                try:
                    contingency = pd.crosstab(df[c1], df[c2])
                    
                    if contingency.sum().sum() < 10:
                        st.error("Datos insuficientes")
                    else:
                        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
                        
                        # Verificar supuestos
                        expected_lt_5 = (expected < 5).sum()
                        pct_lt_5 = (expected_lt_5 / expected.size) * 100
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("χ²", f"{chi2:.4f}")
                        col2.metric("p-valor", f"{p_value:.4f}")
                        col3.metric("gl", dof)
                        
                        if pct_lt_5 > 20:
                            st.warning(f"⚠️ {pct_lt_5:.1f}% celdas < 5. Considera agrupar.")
                        
                        st.subheader("Tabla de contingencia")
                        st.dataframe(contingency)
                        
                        st.subheader("% por fila")
                        st.dataframe((contingency.div(contingency.sum(axis=1), axis=0) * 100).round(2))
                        
                        if p_value < alpha_np:
                            st.success("✅ Asociación significativa")
                        else:
                            st.warning("❌ Sin asociación significativa")
                            
                except Exception as e:
                    st.error(f"Error: {e}")
        
        elif test == "Welch" and numeric_cols and categorical_cols:
            var = st.selectbox("Variable:", numeric_cols)
            group_var = st.selectbox("Grupo (2 categorías):", categorical_cols)
            
            groups = df[group_var].dropna().unique()
            if len(groups) == 2:
                g1, g2 = groups
                
                if st.button("📊 Ejecutar"):
                    try:
                        d1 = df[df[group_var] == g1][var].dropna()
                        d2 = df[df[group_var] == g2][var].dropna()
                        
                        if len(d1) < 2 or len(d2) < 2:
                            st.error("Mínimo 2 por grupo")
                        else:
                            # Levene
                            _, levene_p = stats.levene(d1, d2)
                            
                            # Welch (equal_var=False)
                            t_stat, p_value = stats.ttest_ind(d1, d2, equal_var=False)
                            
                            col1, col2, col3 = st.columns(3)
                            col1.metric("t", f"{t_stat:.4f}")
                            col2.metric("p-valor", f"{p_value:.4f}")
                            col3.metric("Levene p", f"{levene_p:.4f}")
                            
                            col1, col2 = st.columns(2)
                            col1.metric(f"Media {g1}", f"{d1.mean():.4f}")
                            col2.metric(f"Media {g2}", f"{d2.mean():.4f}")
                            
                            st.metric("Diferencia", f"{d1.mean() - d2.mean():.4f}")
                            
                            if levene_p < 0.05:
                                st.info("Varianzas desiguales - Welch apropiado")
                            else:
                                st.info("Varianzas similares - Podría usarse T estándar")
                            
                            if p_value < alpha_np:
                                st.success("✅ Medias diferentes")
                            else:
                                st.warning("❌ Sin diferencia en medias")
                            
                            fig, ax = plt.subplots()
                            plot_df = pd.DataFrame({
                                'Grupo': [g1]*len(d1) + [g2]*len(d2),
                                'Valor': list(d1) + list(d2)
                            })
                            sns.boxplot(data=plot_df, x='Grupo', y='Valor', ax=ax)
                            st.pyplot(fig)
                            
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.warning("Se necesitan exactamente 2 grupos")

else:
    st.info("👆 Carga un archivo en la barra lateral para comenzar")

# Footer
st.markdown("---")
st.markdown("**Stats Bot** - Herramienta de análisis estadístico")