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
    page_title="Analytics Statistics Bot",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal - CAMBIO 2: Generalizar de People Analytics a análisis general
st.title("🤖 Analytics Stats Bot")
st.markdown("""
Esta aplicación te ayuda a realizar análisis estadísticos descriptivos e inferenciales para diversos tipos de datos.
Carga tus datos y consulta a OpenAI qué análisis realizar, luego ejecuta las funciones disponibles.
""")

# Sidebar para configuración
st.sidebar.header("🔧 Configuración")

# Configuración de OpenAI API - CAMBIO 1: Cambiar de Gemini a OpenAI
st.sidebar.subheader("Configuración de OpenAI")
openai_api_key = st.sidebar.text_input("Ingresa tu API Key de OpenAI:", type="password")

# Inicializar cliente OpenAI
openai_client = None
if openai_api_key:
    try:
        openai_client = openai.OpenAI(api_key=openai_api_key)
        st.sidebar.success("✅ OpenAI configurado correctamente")
    except Exception as e:
        st.sidebar.error(f"Error configurando OpenAI: {e}")
else:
    st.sidebar.warning("⚠️ Ingresa tu API Key de OpenAI para usar las recomendaciones")

# Función para consultar a OpenAI - CAMBIO 1
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

# Carga de datos
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
        placeholder="Ej: Quiero analizar si hay diferencias en las métricas entre diferentes categorías, y cómo se relacionan con el rendimiento...",
        height=100,
        key="business_question_main"
    )
    
    if st.button("Obtener recomendaciones de análisis", key="business_recommendations_main") and user_question:
        if openai_api_key:
            with st.spinner("OpenAI está analizando tu caso y datos..."):
                try:
                    # Preparar contexto para OpenAI
                    context = f"""
                    Tengo un dataset con {df.shape[0]} filas y {df.shape[1]} columnas.
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
        
        # Reporte descriptivo básico (sin ydata_profiling) - CAMBIO 3: Quitar profiling
        st.subheader("Reporte Descriptivo Básico")
        st.markdown("Genera un reporte básico de análisis exploratorio de datos.")
        
        if st.button("📊 Generar Reporte Descriptivo", key="descriptive_report_button"):
            with st.spinner("Generando reporte descriptivo..."):
                try:
                    # Reporte descriptivo básico
                    report_content = []
                    
                    # 1. Resumen general
                    report_content.append("## 📊 RESUMEN GENERAL DEL DATASET")
                    report_content.append(f"- **Total de registros:** {df.shape[0]}")
                    report_content.append(f"- **Total de variables:** {df.shape[1]}")
                    report_content.append(f"- **Valores faltantes totales:** {df.isnull().sum().sum()}")
                    
                    # 2. Variables numéricas
                    if numeric_cols:
                        report_content.append("\n## 🔢 VARIABLES NUMÉRICAS")
                        report_content.append(f"Número de variables numéricas: {len(numeric_cols)}")
                        
                        for var in numeric_cols[:10]:  # Mostrar solo las primeras 10
                            stats = df[var].describe()
                            report_content.append(f"\n### {var}")
                            report_content.append(f"- Media: {stats['mean']:.4f}")
                            report_content.append(f"- Desviación estándar: {stats['std']:.4f}")
                            report_content.append(f"- Mínimo: {stats['min']:.4f}")
                            report_content.append(f"- Percentil 25%: {stats['25%']:.4f}")
                            report_content.append(f"- Mediana: {stats['50%']:.4f}")
                            report_content.append(f"- Percentil 75%: {stats['75%']:.4f}")
                            report_content.append(f"- Máximo: {stats['max']:.4f}")
                            report_content.append(f"- Valores faltantes: {df[var].isnull().sum()}")
                    
                    # 3. Variables categóricas
                    if categorical_cols:
                        report_content.append("\n## 📝 VARIABLES CATEGÓRICAS")
                        report_content.append(f"Número de variables categóricas: {len(categorical_cols)}")
                        
                        for var in categorical_cols[:10]:  # Mostrar solo las primeras 10
                            report_content.append(f"\n### {var}")
                            report_content.append(f"- Valores únicos: {df[var].nunique()}")
                            report_content.append(f"- Valores faltantes: {df[var].isnull().sum()}")
                            if df[var].nunique() <= 20:  # Solo mostrar distribución si hay pocas categorías
                                report_content.append("- Distribución:")
                                for value, count in df[var].value_counts().head(10).items():
                                    percentage = (count / len(df)) * 100
                                    report_content.append(f"  - {value}: {count} ({percentage:.2f}%)")
                    
                    # 4. Matriz de correlaciones (solo si hay suficientes variables numéricas)
                    if len(numeric_cols) >= 2:
                        report_content.append("\n## 📈 MATRIZ DE CORRELACIONES")
                        corr_matrix = df[numeric_cols].corr()
                        # Mostrar solo las correlaciones más fuertes
                        strong_corrs = []
                        for i in range(len(corr_matrix.columns)):
                            for j in range(i+1, len(corr_matrix.columns)):
                                corr_value = corr_matrix.iloc[i, j]
                                if abs(corr_value) > 0.5:  # Mostrar solo correlaciones fuertes
                                    strong_corrs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_value))
                        
                        if strong_correls:
                            report_content.append("Correlaciones fuertes encontradas (|r| > 0.5):")
                            for var1, var2, corr in strong_corrs:
                                report_content.append(f"- {var1} ↔ {var2}: {corr:.4f}")
                        else:
                            report_content.append("No se encontraron correlaciones fuertes (|r| > 0.5)")
                    
                    # Unir todo el contenido
                    full_report = "\n".join(report_content)
                    
                    # Crear botón de descarga
                    st.download_button(
                        label="📥 Descargar Reporte Descriptivo (TXT)",
                        data=full_report,
                        file_name="reporte_descriptivo.txt",
                        mime="text/plain"
                    )
                    
                    st.success("✅ Reporte generado correctamente. Haz clic en el botón de descarga.")
                    
                    # Mostrar vista previa del reporte
                    with st.expander("📋 Vista previa del reporte"):
                        st.text(full_report[:2000] + "..." if len(full_report) > 2000 else full_report)
                    
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
                        
                        # Calcular rango de p-valor
                        if ad_statistic < ad_test.critical_values[0]:
                            p_value_range = f"> {ad_test.significance_level[0]/100:.2f}"
                            p_value_numeric = 0.20
                            p_value_interpretation = f"El p-valor es mayor a {ad_test.significance_level[0]/100:.2f}"
                        elif ad_statistic >= ad_test.critical_values[-1]:
                            p_value_range = f"< {ad_test.significance_level[-1]/100:.2f}"
                            p_value_numeric = 0.005
                            p_value_interpretation = f"El p-valor es menor a {ad_test.significance_level[-1]/100:.2f}"
                        else:
                            for i in range(len(ad_test.critical_values) - 1):
                                if ad_test.critical_values[i] <= ad_statistic < ad_test.critical_values[i+1]:
                                    lower_sig = ad_test.significance_level[i+1] / 100
                                    upper_sig = ad_test.significance_level[i] / 100
                                    p_value_range = f"{lower_sig:.3f} < p < {upper_sig:.3f}"
                                    p_value_numeric = (lower_sig + upper_sig) / 2
                                    p_value_interpretation = f"El p-valor está entre {lower_sig:.3f} y {upper_sig:.3f}"
                                    break
                        
                        ad_normal = ad_statistic < critical_value
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Estadístico A-D", f"{ad_statistic:.4f}")
                            st.caption(f"Valor crítico (α={closest_alpha:.3f}): {critical_value:.3f}")
                        with col2:
                            st.metric("p-valor (rango aproximado)", p_value_range)
                            st.caption("⚠️ A-D proporciona rangos, no p-valores exactos")
                        
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
                        else:
                            st.success("✅ No se detectaron valores atípicos significativos")
                        
                except Exception as e:
                    st.error(f"Error en pruebas de normalidad: {e}")
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

    with tab5:  # Homogeneidad de Varianzas
        st.subheader("⚖️ Pruebas de Homogeneidad de Varianzas")
        st.markdown("""
        Evalúa si dos o más grupos tienen varianzas similares. 
        Este es un supuesto importante para pruebas paramétricas como la Prueba T y ANOVA.
        """)
        
        if numeric_cols and categorical_cols:
            st.info("💡 **¿Por qué es importante?** Muchas pruebas paramétricas asumen que los grupos comparados tienen varianzas homogéneas.")
            
            # Selección de variables
            col1, col2 = st.columns(2)
            with col1:
                homo_var = st.selectbox("Variable numérica:", numeric_cols, key="homo_var")
            with col2:
                homo_group = st.selectbox("Variable categórica (grupos):", categorical_cols, key="homo_group")
            
            alpha_homo = st.slider("Nivel de significancia (α):", 0.01, 0.10, 0.05, key="homo_alpha")
            
            if st.button("📊 Ejecutar Pruebas de Homogeneidad"):
                try:
                    # Preparar datos por grupos
                    groups_data = []
                    group_names = []
                    
                    for group in df[homo_group].dropna().unique():
                        group_data = df[df[homo_group] == group][homo_var].dropna()
                        if len(group_data) >= 2:
                            groups_data.append(group_data)
                            group_names.append(group)
                    
                    if len(groups_data) < 2:
                        st.error("Se necesitan al menos 2 grupos con datos válidos")
                    else:
                        st.subheader("📋 Resultados de las Pruebas")
                        
                        # ==========================================
                        # 1. PRUEBA DE LEVENE
                        # ==========================================
                        st.markdown("#### 1. Prueba de Levene")
                        st.caption("Más robusta ante desviaciones de la normalidad")
                        
                        levene_stat, levene_p = stats.levene(*groups_data)
                        levene_homo = levene_p > alpha_homo
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Estadístico de Levene", f"{levene_stat:.4f}")
                        with col2:
                            st.metric("p-valor", f"{levene_p:.4f}")
                        
                        if levene_homo:
                            st.success("✅ Las varianzas son homogéneas según Levene")
                        else:
                            st.error("❌ Las varianzas NO son homogéneas según Levene")
                        
                        # ==========================================
                        # 2. PRUEBA DE BARTLETT
                        # ==========================================
                        st.markdown("#### 2. Prueba de Bartlett")
                        st.caption("Más sensible, pero requiere normalidad estricta")
                        
                        bartlett_stat, bartlett_p = stats.bartlett(*groups_data)
                        bartlett_homo = bartlett_p > alpha_homo
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Estadístico de Bartlett", f"{bartlett_stat:.4f}")
                        with col2:
                            st.metric("p-valor", f"{bartlett_p:.4f}")
                        
                        if bartlett_homo:
                            st.success("✅ Las varianzas son homogéneas según Bartlett")
                        else:
                            st.error("❌ Las varianzas NO son homogéneas según Bartlett")
                        
                        # ==========================================
                        # COMPARACIÓN DE RESULTADOS
                        # ==========================================
                        st.markdown("---")
                        st.subheader("🎯 COMPARACIÓN Y CONCLUSIÓN")
                        
                        # Tabla comparativa
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
                        st.dataframe(results_df, hide_index=True, use_container_width=True)
                        
                        # Consenso
                        if levene_homo and bartlett_homo:
                            st.success("""
                            ✅ **CONCLUSIÓN: LAS VARIANZAS SON HOMOGÉNEAS**
                            
                            Ambas pruebas confirman homogeneidad de varianzas.
                            
                            **✓ Puedes usar con confianza:**
                            - Prueba T independiente (con equal_var=True)
                            - ANOVA clásica
                            - Otras pruebas que asumen varianzas iguales
                            """)
                        elif levene_homo and not bartlett_homo:
                            st.warning("""
                            ⚠️ **CONCLUSIÓN: RESULTADOS MIXTOS (FAVORECEN HOMOGENEIDAD)**
                            
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
                            ⚠️ **CONCLUSIÓN: RESULTADOS MIXTOS (FAVORECEN NO HOMOGENEIDAD)**
                            
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
                            ❌ **CONCLUSIÓN: LAS VARIANZAS NO SON HOMOGÉNEAS**
                            
                            Ambas pruebas rechazan la homogeneidad de varianzas.
                            
                            **Opciones recomendadas:**
                            
                            **1. Usar versiones robustas de las pruebas:**
                            - Welch's t-test (en lugar de t-test estándar)
                            - Welch's ANOVA (en lugar de ANOVA estándar)
                            
                            **2. Transformar los datos:**
                            - Logaritmo: log(x) - reduce varianzas grandes
                            - Raíz cuadrada: sqrt(x) - estabiliza varianzas
                            - Box-Cox: transformación óptima automática
                            
                            **3. Usar pruebas no paramétricas:**
                            - Mann-Whitney U (en lugar de t-test)
                            - Kruskal-Wallis (en lugar de ANOVA)
                            """)
                        
                except Exception as e:
                    st.error(f"Error en pruebas de homogeneidad: {e}")
        else:
            st.warning("Se necesitan variables numéricas y categóricas para evaluar homogeneidad de varianzas")

    with tab6:  # Pruebas T
        st.subheader("✅ Pruebas T")
        st.markdown("Compara las medias entre grupos o con un valor de referencia.")
        
        test_type = st.radio(
            "Selecciona el tipo de prueba T:",
            ["Una muestra", "Muestras independientes", "Muestras pareadas"],
            key="ttest_type")
        
        col_alpha, col_alt = st.columns(2)
        with col_alpha:
            alpha_ttest = st.slider("Nivel de significancia (α):", 0.01, 0.10, 0.05, key="ttest_alpha")
        with col_alt:
            alternative = st.selectbox(
                "Hipótesis alternativa:",
                ["two-sided", "less", "greater"],
                format_func=lambda x: {
                    "two-sided": "Bilateral (≠)",
                    "less": "Unilateral izquierda (<)",
                    "greater": "Unilateral derecha (>)"
                }[x],
                key="alternative"
            )
        
        # Función para calcular tamaño del efecto
        def calculate_effect_size(test_type, data1=None, data2=None, paired_data=None, var_before=None, var_after=None, pop_mean=0):
            try:
                if test_type == "Una muestra":
                    d = (data1.mean() - pop_mean) / data1.std()
                    return abs(d)
                elif test_type == "Muestras independientes":
                    n1, n2 = len(data1), len(data2)
                    pooled_std = np.sqrt(((n1-1)*data1.std()**2 + (n2-1)*data2.std()**2) / (n1 + n2 - 2))
                    d = (data1.mean() - data2.mean()) / pooled_std
                    return abs(d)
                elif test_type == "Muestras pareadas":
                    differences = paired_data[var_after] - paired_data[var_before]
                    d = differences.mean() / differences.std()
                    return abs(d)
            except:
                return 0
        
        # Función para interpretar tamaño del efecto
        def interpret_effect_size(d):
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
        
        # Función para obtener hipótesis
        def get_hypotheses(test_type, alternative, pop_mean=0, group1=None, group2=None, var_before=None, var_after=None):
            if test_type == "Una muestra":
                h0 = f"μ = {pop_mean} (La media poblacional es igual a {pop_mean})"
                if alternative == "two-sided":
                    h1 = f"μ ≠ {pop_mean} (La media poblacional es diferente de {pop_mean})"
                elif alternative == "less":
                    h1 = f"μ < {pop_mean} (La media poblacional es menor que {pop_mean})"
                else:
                    h1 = f"μ > {pop_mean} (La media poblacional es mayor que {pop_mean})"
                    
            elif test_type == "Muestras independientes":
                h0 = f"μ₁ = μ₂ (Las medias de {group1} y {group2} son iguales)"
                if alternative == "two-sided":
                    h1 = f"μ₁ ≠ μ₂ (Las medias de {group1} y {group2} son diferentes)"
                elif alternative == "less":
                    h1 = f"μ₁ < μ₂ (La media de {group1} es menor que la de {group2})"
                else:
                    h1 = f"μ₁ > μ₂ (La media de {group1} es mayor que la de {group2})"
                    
            else:
                h0 = f"μ_antes = μ_después (No hay diferencia entre {var_before} y {var_after})"
                if alternative == "two-sided":
                    h1 = f"μ_antes ≠ μ_después (Hay diferencia entre {var_before} y {var_after})"
                elif alternative == "less":
                    h1 = f"μ_antes < μ_después ({var_before} es menor que {var_after})"
                else:
                    h1 = f"μ_antes > μ_después ({var_before} es mayor que {var_after})"
                    
            return h0, h1
        
        # Función para cálculo de p-valor unilateral
        def calculate_ttest_pvalue(test_type, alternative, data1=None, data2=None, pop_mean=0, paired_data=None, var_before=None, var_after=None):
            if test_type == "Una muestra":
                t_stat, p_value = stats.ttest_1samp(data1, pop_mean)
                
                if alternative == "less":
                    if data1.mean() < pop_mean:
                        p_value = p_value / 2
                    else:
                        p_value = 1 - p_value / 2
                elif alternative == "greater":
                    if data1.mean() > pop_mean:
                        p_value = p_value / 2
                    else:
                        p_value = 1 - p_value / 2
                        
                return t_stat, p_value
                
            elif test_type == "Muestras independientes":
                levene_stat, levene_p = stats.levene(data1, data2)
                equal_var = levene_p > 0.05
                
                t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=equal_var)
                
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
                        
                return t_stat, p_value, equal_var, levene_p
                
            elif test_type == "Muestras pareadas":
                t_stat, p_value = stats.ttest_rel(paired_data[var_before], paired_data[var_after])
                
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
                        
                return t_stat, p_value

        if test_type == "Una muestra" and numeric_cols:
            st.subheader("Prueba T para Una Muestra")
            st.markdown("Compara la media de tu variable con un valor poblacional de referencia.")
            
            var_onesample = st.selectbox("Variable numérica:", numeric_cols, key="onesample_var")
            pop_mean = st.number_input("Media poblacional de referencia:", value=0.0, key="pop_mean")
            
            if st.button("📊 Ejecutar Prueba T Una Muestra"):
                try:
                    data = df[var_onesample].dropna()
                    
                    if len(data) < 2:
                        st.error("Se necesitan al menos 2 observaciones para la prueba T")
                    else:
                        t_stat, p_value = calculate_ttest_pvalue("Una muestra", alternative, data1=data, pop_mean=pop_mean)
                        
                        from scipy.stats import t
                        dof = len(data) - 1
                        sem = stats.sem(data)
                        ci_low, ci_high = t.interval(1-alpha_ttest, dof, loc=data.mean(), scale=sem)
                        
                        effect_size = calculate_effect_size("Una muestra", data1=data, pop_mean=pop_mean)
                        effect_magnitude, effect_color = interpret_effect_size(effect_size)
                        
                        h0, h1 = get_hypotheses("Una muestra", alternative, pop_mean)
                        
                        st.subheader("📋 Resultados Principales")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Estadístico t", f"{t_stat:.4f}")
                        with col2:
                            st.metric("p-valor", f"{p_value:.4f}")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Media muestral", f"{data.mean():.4f}")
                        with col2:
                            st.metric("Media poblacional referencia", f"{pop_mean:.4f}")
                        
                        st.metric("Desviación estándar", f"{data.std():.4f}")
                        st.metric(f"Intervalo de confianza del {(1-alpha_ttest)*100:.1f}%", 
                                f"[{ci_low:.4f}, {ci_high:.4f}]")
                        st.metric("Tamaño del efecto (Cohen's d)", f"{effect_size:.4f}")
                        
                        with st.expander("🎯 **Interpretación Detallada**", expanded=True):
                            st.write("**Hipótesis:**")
                            st.write(f"- **H₀ (Nula):** {h0}")
                            st.write(f"- **H₁ (Alternativa):** {h1}")
                            st.write("")
                            
                            st.write("**Decisión estadística:**")
                            if p_value < alpha_ttest:
                                st.success(f"✅ **Se rechaza la hipótesis nula** (p = {p_value:.4f} < α = {alpha_ttest})")
                            else:
                                st.warning(f"✅ **No se rechaza la hipótesis nula** (p = {p_value:.4f} ≥ α = {alpha_ttest})")
                            
                            st.write("")
                            st.write("**Tamaño del efecto:**")
                            st.markdown(f"<span style='color:{effect_color}; font-weight:bold;'>{effect_magnitude}</span> (d = {effect_size:.4f})", 
                                    unsafe_allow_html=True)
                        
                        # Gráfico
                        st.subheader("📊 Visualización")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        sns.histplot(data, kde=True, alpha=0.7, ax=ax, label='Distribución muestral')
                        ax.axvline(data.mean(), color='red', linestyle='-', linewidth=2, label=f'Media muestral: {data.mean():.4f}')
                        ax.axvline(pop_mean, color='blue', linestyle='--', linewidth=2, label=f'Media referencia: {pop_mean}')
                        
                        ax.set_title(f'Distribución de {var_onesample} vs Referencia')
                        ax.set_xlabel(var_onesample)
                        ax.legend()
                        st.pyplot(fig)
                        
                except Exception as e:
                    st.error(f"Error en prueba T una muestra: {e}")
        
        elif test_type == "Muestras independientes" and numeric_cols and categorical_cols:
            st.subheader("Prueba T para Muestras Independientes")
            st.markdown("Compara las medias de dos grupos diferentes.")
            
            var_independent = st.selectbox("Variable numérica:", numeric_cols, key="indep_var")
            group_var = st.selectbox("Variable categórica (debe tener 2 grupos):", categorical_cols, key="group_var")
            
            unique_groups = df[group_var].dropna().unique()
            if len(unique_groups) == 2:
                group1, group2 = unique_groups
                
                if st.button("📊 Ejecutar Prueba T Independiente"):
                    try:
                        data1 = df[df[group_var] == group1][var_independent].dropna()
                        data2 = df[df[group_var] == group2][var_independent].dropna()
                        
                        if len(data1) < 2 or len(data2) < 2:
                            st.error("Cada grupo necesita al menos 2 observaciones")
                        else:
                            t_stat, p_value, equal_var, levene_p = calculate_ttest_pvalue(
                                "Muestras independientes", alternative, data1=data1, data2=data2)
                            
                            effect_size = calculate_effect_size("Muestras independientes", data1=data1, data2=data2)
                            effect_magnitude, effect_color = interpret_effect_size(effect_size)
                            
                            h0, h1 = get_hypotheses("Muestras independientes", alternative, group1=group1, group2=group2)
                            
                            st.subheader("📋 Resultados Principales")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Estadístico t", f"{t_stat:.4f}")
                            with col2:
                                st.metric("p-valor", f"{p_value:.4f}")
                            with col3:
                                st.metric("Varianzas", "Iguales" if equal_var else "Diferentes")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric(f"Media {group1}", f"{data1.mean():.4f}")
                                st.metric(f"Desviación {group1}", f"{data1.std():.4f}")
                                st.metric(f"Tamaño {group1}", len(data1))
                            with col2:
                                st.metric(f"Media {group2}", f"{data2.mean():.4f}")
                                st.metric(f"Desviación {group2}", f"{data2.std():.4f}")
                                st.metric(f"Tamaño {group2}", len(data2))
                            
                            st.metric("Diferencia de medias", f"{(data1.mean() - data2.mean()):.4f}")
                            st.metric("Tamaño del efecto (d)", f"{effect_size:.4f}")
                            
                            with st.expander("🎯 **Interpretación Detallada**", expanded=True):
                                st.write("**Hipótesis:**")
                                st.write(f"- **H₀ (Nula):** {h0}")
                                st.write(f"- **H₁ (Alternativa):** {h1}")
                                st.write("")
                                
                                st.write("**Decisión estadística:**")
                                if p_value < alpha_ttest:
                                    st.success(f"✅ **Se rechaza la hipótesis nula** (p = {p_value:.4f} < α = {alpha_ttest})")
                                else:
                                    st.warning(f"✅ **No se rechaza la hipótesis nula** (p = {p_value:.4f} ≥ α = {alpha_ttest})")
                            
                            # Boxplot comparativo
                            st.subheader("📊 Comparación Visual")
                            fig, ax = plt.subplots(figsize=(10, 6))
                            plot_data = pd.DataFrame({
                                'Grupo': [group1] * len(data1) + [group2] * len(data2),
                                'Valor': list(data1) + list(data2)
                            })
                            sns.boxplot(data=plot_data, x='Grupo', y='Valor', ax=ax)
                            ax.set_title(f'Comparación de {var_independent} entre {group1} y {group2}')
                            
                            st.pyplot(fig)
                            
                    except Exception as e:
                        st.error(f"Error en prueba T independiente: {e}")
            else:
                st.warning(f"La variable '{group_var}' tiene {len(unique_groups)} grupos. Debe tener exactamente 2 grupos.")
        
        elif test_type == "Muestras pareadas" and numeric_cols:
            st.subheader("Prueba T para Muestras Pareadas")
            st.markdown("Compara mediciones del mismo grupo en dos momentos diferentes (antes/después).")
            
            st.info("💡 Para esta prueba necesitas dos variables que representen mediciones pareadas (ej: pre-test y post-test)")
            
            if len(numeric_cols) >= 2:
                col1, col2 = st.columns(2)
                with col1:
                    var_before = st.selectbox("Variable 'Antes':", numeric_cols, key="before_var")
                with col2:
                    var_after = st.selectbox("Variable 'Después':", numeric_cols, key="after_var")
                
                if st.button("📊 Ejecutar Prueba T Pareada"):
                    try:
                        paired_data = df[[var_before, var_after]].dropna()
                        
                        if len(paired_data) < 2:
                            st.error("Se necesitan al menos 2 pares completos de observaciones")
                        else:
                            t_stat, p_value = calculate_ttest_pvalue(
                                "Muestras pareadas", alternative, 
                                paired_data=paired_data, 
                                var_before=var_before, 
                                var_after=var_after
                            )
                            
                            effect_size = calculate_effect_size(
                                "Muestras pareadas", 
                                paired_data=paired_data, 
                                var_before=var_before, 
                                var_after=var_after
                            )
                            effect_magnitude, effect_color = interpret_effect_size(effect_size)
                            
                            h0, h1 = get_hypotheses(
                                "Muestras pareadas", alternative, 
                                var_before=var_before, 
                                var_after=var_after
                            )
                            
                            st.subheader("📋 Resultados Principales")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Estadístico t", f"{t_stat:.4f}")
                            with col2:
                                st.metric("p-valor", f"{p_value:.4f}")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric(f"Media '{var_before}'", f"{paired_data[var_before].mean():.4f}")
                                st.metric(f"Desviación '{var_before}'", f"{paired_data[var_before].std():.4f}")
                            with col2:
                                st.metric(f"Media '{var_after}'", f"{paired_data[var_after].mean():.4f}")
                                st.metric(f"Desviación '{var_after}'", f"{paired_data[var_after].std():.4f}")
                            
                            st.metric("Número de pares", len(paired_data))
                            
                            difference = paired_data[var_after].mean() - paired_data[var_before].mean()
                            st.metric("Diferencia media", f"{difference:.4f}")
                            st.metric("Tamaño del efecto (d)", f"{effect_size:.4f}")
                            
                            with st.expander("🎯 **Interpretación Detallada**", expanded=True):
                                st.write("**Hipótesis:**")
                                st.write(f"- **H₀ (Nula):** {h0}")
                                st.write(f"- **H₁ (Alternativa):** {h1}")
                                st.write("")
                                
                                st.write("**Decisión estadística:**")
                                if p_value < alpha_ttest:
                                    st.success(f"✅ **Se rechaza la hipótesis nula** (p = {p_value:.4f} < α = {alpha_ttest})")
                                else:
                                    st.warning(f"✅ **No se rechaza la hipótesis nula** (p = {p_value:.4f} ≥ α = {alpha_ttest})")
                            
                            # Gráfico de comparación
                            st.subheader("📊 Comparación Visual")
                            fig, ax = plt.subplots(figsize=(10, 6))
                            
                            plot_data = pd.DataFrame({
                                'Momento': ['Antes'] * len(paired_data) + ['Después'] * len(paired_data),
                                'Valor': list(paired_data[var_before]) + list(paired_data[var_after])
                            })
                            sns.boxplot(data=plot_data, x='Momento', y='Valor', ax=ax)
                            ax.set_title('Distribución Antes vs Después')
                            
                            st.pyplot(fig)
                            
                    except Exception as e:
                        st.error(f"Error en prueba T pareada: {e}")
            else:
                st.warning("Se necesitan al menos 2 variables numéricas para la prueba pareada")

    with tab7:  # ANOVA
        st.subheader("📊 Análisis de Varianza (ANOVA)")
        st.markdown("Compara las medias de tres o más grupos.")
        
        if numeric_cols and categorical_cols:
            anova_type = st.radio(
                "Tipo de ANOVA:",
                ["Una vía (One-Way)", "Dos vías (Two-Way)"],
                key="anova_type_tab7"
            )
            
            anova_var = st.selectbox("Variable numérica:", numeric_cols, key="anova_var_tab7")
            alpha_anova = st.slider("Nivel de significancia (α):", 0.01, 0.10, 0.05, key="anova_alpha_tab7")
            
            if anova_type == "Una vía (One-Way)":
                anova_group = st.selectbox("Variable categórica:", categorical_cols, key="anova_group_tab7")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    anova_group1 = st.selectbox("Primera variable categórica:", 
                                            categorical_cols, 
                                            key="anova_group1_tab7")
                with col2:
                    available_groups = [col for col in categorical_cols if col != anova_group1]
                    if available_groups:
                        anova_group2 = st.selectbox("Segunda variable categórica:", 
                                                available_groups, 
                                                key="anova_group2_tab7")
                    else:
                        st.warning("Se necesita al menos una variable categórica diferente")
                        anova_group2 = None
            
            if st.button("📊 Ejecutar ANOVA", key="anova_button_tab7"):
                try:
                    if anova_type == "Una vía (One-Way)":
                        groups_data = []
                        group_names = []
                        
                        for group in df[anova_group].dropna().unique():
                            group_data = df[df[anova_group] == group][anova_var].dropna()
                            if len(group_data) >= 2:
                                groups_data.append(group_data)
                                group_names.append(str(group))
                        
                        if len(groups_data) < 2:
                            st.error("Se necesitan al menos 2 grupos con datos válidos")
                        else:
                            f_stat, p_value = stats.f_oneway(*groups_data)
                            
                            st.subheader("📋 Resultados del ANOVA de Una Vía")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Estadístico F", f"{f_stat:.4f}")
                            with col2:
                                st.metric("p-valor", f"{p_value:.4f}")
                            
                            # Estadísticas descriptivas
                            st.subheader("📊 Estadísticas por Grupo")
                            stats_by_group = []
                            for name, data in zip(group_names, groups_data):
                                stats_by_group.append({
                                    'Grupo': name,
                                    'n': len(data),
                                    'Media': f"{data.mean():.4f}",
                                    'Desviación': f"{data.std():.4f}",
                                    'Mínimo': f"{data.min():.4f}",
                                    'Máximo': f"{data.max():.4f}"
                                })
                            
                            stats_df = pd.DataFrame(stats_by_group)
                            st.dataframe(stats_df, use_container_width=True)
                            
                            if p_value < alpha_anova:
                                st.success("""
                                ✅ **Se rechaza la hipótesis nula**
                                Hay al menos una diferencia significativa entre los grupos.
                                """)
                                
                                # Prueba post-hoc Tukey HSD
                                st.subheader("🔍 Comparaciones Múltiples (Tukey HSD)")
                                st.markdown("Identifica qué grupos específicamente son diferentes:")
                                
                                try:
                                    tukey_data = df[[anova_var, anova_group]].dropna()
                                    tukey = pairwise_tukeyhsd(tukey_data[anova_var], tukey_data[anova_group], alpha=alpha_anova)
                                    
                                    result_df = pd.DataFrame(data=tukey._results_table.data[1:], 
                                                        columns=tukey._results_table.data[0])
                                    st.dataframe(result_df, use_container_width=True)
                                    
                                    significant_pairs = result_df[result_df['p-adj'] < alpha_anova]
                                    if len(significant_pairs) > 0:
                                        st.write("**Pares significativamente diferentes:**")
                                        for _, row in significant_pairs.iterrows():
                                            st.write(f"- {row['group1']} vs {row['group2']} (p-adj = {row['p-adj']:.4f})")
                                    else:
                                        st.write("No se encontraron diferencias significativas entre pares específicos")
                                        
                                except Exception as e:
                                    st.error(f"Error en prueba post-hoc: {e}")
                                    
                            else:
                                st.warning("""
                                ✅ **No se rechaza la hipótesis nula**
                                No hay evidencia suficiente de diferencias entre los grupos.
                                """)
                            
                            # Visualización
                            st.subheader("📈 Visualización")
                            fig, ax = plt.subplots(figsize=(12, 6))
                            plot_data = []
                            for name, data in zip(group_names, groups_data):
                                for value in data:
                                    plot_data.append({'Grupo': name, 'Valor': value})
                            
                            plot_df = pd.DataFrame(plot_data)
                            sns.boxplot(data=plot_df, x='Grupo', y='Valor', ax=ax)
                            ax.set_title(f'ANOVA Una Vía: {anova_var} por {anova_group}\n(F = {f_stat:.3f}, p = {p_value:.4f})')
                            ax.tick_params(axis='x', rotation=45)
                            plt.tight_layout()
                            st.pyplot(fig)
                    
                    else:
                        if not anova_group2:
                            st.error("Se necesitan dos variables categóricas diferentes")
                        else:
                            anova_data = df[[anova_var, anova_group1, anova_group2]].dropna()
                            
                            if len(anova_data) == 0:
                                st.error("No hay datos suficientes después de eliminar valores faltantes")
                            else:
                                try:
                                    formula = f'{anova_var} ~ C({anova_group1}) + C({anova_group2}) + C({anova_group1}):C({anova_group2})'
                                    model = ols(formula, data=anova_data).fit()
                                    anova_table = sm.stats.anova_lm(model, typ=2)
                                    
                                    st.subheader("📋 Resultados del ANOVA de Dos Vías")
                                    
                                    st.dataframe(anova_table, use_container_width=True)
                                    
                                    f_stat_group1 = anova_table.loc[f'C({anova_group1})', 'F']
                                    p_value_group1 = anova_table.loc[f'C({anova_group1})', 'PR(>F)']
                                    f_stat_group2 = anova_table.loc[f'C({anova_group2})', 'F']
                                    p_value_group2 = anova_table.loc[f'C({anova_group2})', 'PR(>F)']
                                    
                                    try:
                                        f_stat_interaction = anova_table.loc[f'C({anova_group1}):C({anova_group2})', 'F']
                                        p_value_interaction = anova_table.loc[f'C({anova_group1}):C({anova_group2})', 'PR(>F)']
                                        has_interaction = True
                                    except KeyError:
                                        has_interaction = False
                                    
                                    st.subheader("🎯 Interpretación")
                                    if has_interaction:
                                        col1, col2, col3 = st.columns(3)
                                    else:
                                        col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.write(f"**{anova_group1}:**")
                                        st.metric("F", f"{f_stat_group1:.4f}")
                                        st.metric("p-valor", f"{p_value_group1:.4f}")
                                        if p_value_group1 < alpha_anova:
                                            st.success("✅ Significativo")
                                        else:
                                            st.warning("❌ No significativo")
                                    
                                    with col2:
                                        st.write(f"**{anova_group2}:**")
                                        st.metric("F", f"{f_stat_group2:.4f}")
                                        st.metric("p-valor", f"{p_value_group2:.4f}")
                                        if p_value_group2 < alpha_anova:
                                            st.success("✅ Significativo")
                                        else:
                                            st.warning("❌ No significativo")
                                    
                                    if has_interaction:
                                        with col3:
                                            st.write("**Interacción:**")
                                            st.metric("F", f"{f_stat_interaction:.4f}")
                                            st.metric("p-valor", f"{p_value_interaction:.4f}")
                                            if p_value_interaction < alpha_anova:
                                                st.success("✅ Significativa")
                                            else:
                                                st.warning("❌ No significativa")
                                    
                                    # Visualización
                                    st.subheader("📈 Visualización")
                                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                                    
                                    sns.boxplot(data=anova_data, x=anova_group1, y=anova_var, ax=ax1)
                                    ax1.set_title(f'Efecto de {anova_group1}')
                                    ax1.tick_params(axis='x', rotation=45)
                                    
                                    sns.boxplot(data=anova_data, x=anova_group2, y=anova_var, ax=ax2)
                                    ax2.set_title(f'Efecto de {anova_group2}')
                                    ax2.tick_params(axis='x', rotation=45)
                                    
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                    
                                except Exception as e:
                                    st.error(f"Error en ANOVA de dos vías: {e}")
                    
                except Exception as e:
                    st.error(f"Error en ANOVA: {e}")
        else:
            st.warning("Se necesitan variables numéricas y categóricas para realizar ANOVA")

    with tab8:  # Pruebas no paramétricas
        st.subheader("🔄 Pruebas No Paramétricas")
        st.markdown("Alternativas a las pruebas paramétricas cuando no se cumplen los supuestos de normalidad.")
        
        nonpar_test = st.radio(
            "Selecciona la prueba no paramétrica:",
            ["Mann-Whitney U", "Wilcoxon (Pareada)", "Wilcoxon (Una muestra)", "Kruskal-Wallis", "Chi-cuadrado", "Welch (varianzas desiguales)"],
            key="nonpar_test_tab8"
        )
        
        alpha_nonpar = st.slider("Nivel de significancia (α):", 0.01, 0.10, 0.05, key="nonpar_alpha_tab8")
        
        if nonpar_test == "Mann-Whitney U" and numeric_cols and categorical_cols:
            st.subheader("Prueba de Mann-Whitney U")
            
            mw_var = st.selectbox("Variable numérica:", numeric_cols, key="mw_var_tab8")
            mw_group = st.selectbox("Variable categórica (debe tener 2 grupos):", categorical_cols, key="mw_group_tab8")
            
            unique_groups = df[mw_group].dropna().unique()
            if len(unique_groups) == 2:
                group1, group2 = unique_groups
                
                if st.button("📊 Ejecutar Mann-Whitney U", key="mw_button_tab8"):
                    try:
                        data1 = df[df[mw_group] == group1][mw_var].dropna()
                        data2 = df[df[mw_group] == group2][mw_var].dropna()
                        
                        if len(data1) < 3 or len(data2) < 3:
                            st.error("Cada grupo necesita al menos 3 observaciones")
                        else:
                            u_stat, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                            
                            st.subheader("📋 Resultados")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Estadístico U", f"{u_stat:.4f}")
                            with col2:
                                st.metric("p-valor", f"{p_value:.4f}")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric(f"Mediana {group1}", f"{data1.median():.4f}")
                                st.metric(f"Rango intercuartílico {group1}", f"{data1.quantile(0.75) - data1.quantile(0.25):.4f}")
                                st.metric(f"Tamaño {group1}", len(data1))
                            with col2:
                                st.metric(f"Mediana {group2}", f"{data2.median():.4f}")
                                st.metric(f"Rango intercuartílico {group2}", f"{data2.quantile(0.75) - data2.quantile(0.25):.4f}")
                                st.metric(f"Tamaño {group2}", len(data2))
                            
                            if p_value < alpha_nonpar:
                                st.success(f"✅ **Se rechaza la hipótesis nula** (p = {p_value:.4f} < α = {alpha_nonpar})")
                            else:
                                st.warning(f"✅ **No se rechaza la hipótesis nula** (p = {p_value:.4f} ≥ α = {alpha_nonpar})")
                            
                            # Visualización
                            st.subheader("📊 Comparación Visual")
                            fig, ax = plt.subplots(figsize=(10, 6))
                            plot_data = pd.DataFrame({
                                'Grupo': [group1] * len(data1) + [group2] * len(data2),
                                'Valor': list(data1) + list(data2)
                            })
                            sns.boxplot(data=plot_data, x='Grupo', y='Valor', ax=ax)
                            ax.set_title(f'Comparación de {mw_var} entre {group1} y {group2}\n(Prueba U de Mann-Whitney)')
                            st.pyplot(fig)
                                    
                    except Exception as e:
                        st.error(f"Error en Mann-Whitney U: {e}")
            else:
                st.warning("La variable categórica debe tener exactamente 2 grupos")
        
        elif nonpar_test == "Wilcoxon (Pareada)" and numeric_cols:
            st.subheader("Prueba de Wilcoxon para Muestras Pareadas")
            
            if len(numeric_cols) >= 2:
                col1, col2 = st.columns(2)
                with col1:
                    wilcoxon_before = st.selectbox("Variable 'Antes':", numeric_cols, key="wilcoxon_before_tab8")
                with col2:
                    wilcoxon_after = st.selectbox("Variable 'Después':", numeric_cols, key="wilcoxon_after_tab8")
                
                if st.button("📊 Ejecutar Prueba de Wilcoxon", key="wilcoxon_button_tab8"):
                    try:
                        paired_data = df[[wilcoxon_before, wilcoxon_after]].dropna()
                        
                        if len(paired_data) < 3:
                            st.error("Se necesitan al menos 3 pares completos de observaciones")
                        else:
                            differences = paired_data[wilcoxon_after] - paired_data[wilcoxon_before]
                            w_stat, p_value = stats.wilcoxon(differences)
                            
                            st.subheader("📋 Resultados")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Estadístico W", f"{w_stat:.4f}")
                            with col2:
                                st.metric("p-valor", f"{p_value:.4f}")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric(f"Mediana '{wilcoxon_before}'", f"{paired_data[wilcoxon_before].median():.4f}")
                                st.metric(f"Rango IQ '{wilcoxon_before}'", 
                                        f"{paired_data[wilcoxon_before].quantile(0.75) - paired_data[wilcoxon_before].quantile(0.25):.4f}")
                            with col2:
                                st.metric(f"Mediana '{wilcoxon_after}'", f"{paired_data[wilcoxon_after].median():.4f}")
                                st.metric(f"Rango IQ '{wilcoxon_after}'", 
                                        f"{paired_data[wilcoxon_after].quantile(0.75) - paired_data[wilcoxon_after].quantile(0.25):.4f}")
                            
                            st.metric("Mediana de diferencias", f"{differences.median():.4f}")
                            st.metric("Número de pares", len(paired_data))
                            
                            if p_value < alpha_nonpar:
                                st.success(f"✅ **Se rechaza la hipótesis nula** (p = {p_value:.4f} < α = {alpha_nonpar})")
                            else:
                                st.warning(f"✅ **No se rechaza la hipótesis nula** (p = {p_value:.4f} ≥ α = {alpha_nonpar})")
                            
                            # Visualización
                            st.subheader("📊 Comparación Visual")
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                            
                            plot_data = pd.DataFrame({
                                'Momento': ['Antes'] * len(paired_data) + ['Después'] * len(paired_data),
                                'Valor': list(paired_data[wilcoxon_before]) + list(paired_data[wilcoxon_after])
                            })
                            sns.boxplot(data=plot_data, x='Momento', y='Valor', ax=ax1)
                            ax1.set_title('Distribución Antes vs Después')
                            
                            sns.histplot(differences, kde=True, ax=ax2)
                            ax2.axvline(0, color='red', linestyle='--', alpha=0.7, label='Sin cambio')
                            ax2.axvline(differences.median(), color='green', linestyle='-', alpha=0.8, 
                                        label=f'Mediana diferencias: {differences.median():.4f}')
                            ax2.set_title('Distribución de las Diferencias')
                            ax2.set_xlabel('Diferencia (Después - Antes)')
                            ax2.legend()
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                    except Exception as e:
                        st.error(f"Error en prueba de Wilcoxon: {e}")
            else:
                st.warning("Se necesitan al menos 2 variables numéricas para la prueba pareada")
        
        elif nonpar_test == "Wilcoxon (Una muestra)" and numeric_cols:
            st.subheader("Prueba de Wilcoxon para Una Muestra")
            
            wilcoxon_var = st.selectbox("Variable numérica:", numeric_cols, key="wilcoxon_onesample_var_tab8")
            wilcoxon_reference = st.number_input("Valor de referencia (mediana poblacional):", value=0.0, key="wilcoxon_reference_tab8")
            
            if st.button("📊 Ejecutar Wilcoxon Una Muestra", key="wilcoxon_onesample_button_tab8"):
                try:
                    data = df[wilcoxon_var].dropna()
                    
                    if len(data) < 3:
                        st.error("Se necesitan al menos 3 observaciones")
                    else:
                        w_stat, p_value = stats.wilcoxon(data - wilcoxon_reference)
                        
                        st.subheader("📋 Resultados")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Estadístico W", f"{w_stat:.4f}")
                        with col2:
                            st.metric("p-valor", f"{p_value:.4f}")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Mediana muestral", f"{data.median():.4f}")
                            st.metric("Rango intercuartílico", f"{data.quantile(0.75) - data.quantile(0.25):.4f}")
                        with col2:
                            st.metric("Valor de referencia", f"{wilcoxon_reference:.4f}")
                            st.metric("Tamaño de muestra", len(data))
                        
                        if p_value < alpha_nonpar:
                            st.success(f"✅ **Se rechaza la hipótesis nula** (p = {p_value:.4f} < α = {alpha_nonpar})")
                        else:
                            st.warning(f"✅ **No se rechaza la hipótesis nula** (p = {p_value:.4f} ≥ α = {alpha_nonpar})")
                        
                        # Visualización
                        st.subheader("📊 Visualización")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        sns.histplot(data, kde=True, alpha=0.7, ax=ax, label='Distribución muestral')
                        ax.axvline(data.median(), color='red', linestyle='-', linewidth=2, label=f'Mediana muestral: {data.median():.4f}')
                        ax.axvline(wilcoxon_reference, color='blue', linestyle='--', linewidth=2, label=f'Valor referencia: {wilcoxon_reference}')
                        
                        ax.set_title(f'Distribución de {wilcoxon_var} vs Referencia\n(Prueba de Wilcoxon Una Muestra)')
                        ax.set_xlabel(wilcoxon_var)
                        ax.legend()
                        st.pyplot(fig)
                        
                except Exception as e:
                    st.error(f"Error en Wilcoxon una muestra: {e}")
        
        elif nonpar_test == "Welch (varianzas desiguales)" and numeric_cols and categorical_cols:
            st.subheader("Prueba T de Welch")
            
            welch_var = st.selectbox("Variable numérica:", numeric_cols, key="welch_var_tab8")
            welch_group = st.selectbox("Variable categórica (debe tener 2 grupos):", categorical_cols, key="welch_group_tab8")
            
            unique_groups = df[welch_group].dropna().unique()
            if len(unique_groups) == 2:
                group1, group2 = unique_groups
                
                if st.button("📊 Ejecutar Prueba T de Welch", key="welch_button_tab8"):
                    try:
                        data1 = df[df[welch_group] == group1][welch_var].dropna()
                        data2 = df[df[welch_group] == group2][welch_var].dropna()
                        
                        if len(data1) < 2 or len(data2) < 2:
                            st.error("Cada grupo necesita al menos 2 observaciones")
                        else:
                            t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=False)
                            
                            levene_stat, levene_p = stats.levene(data1, data2)
                            
                            st.subheader("📋 Resultados")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Estadístico t", f"{t_stat:.4f}")
                            with col2:
                                st.metric("p-valor", f"{p_value:.4f}")
                            with col3:
                                st.metric("Prueba", "Welch")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric(f"Media {group1}", f"{data1.mean():.4f}")
                                st.metric(f"Desviación {group1}", f"{data1.std():.4f}")
                                st.metric(f"Varianza {group1}", f"{data1.var():.4f}")
                                st.metric(f"Tamaño {group1}", len(data1))
                            with col2:
                                st.metric(f"Media {group2}", f"{data2.mean():.4f}")
                                st.metric(f"Desviación {group2}", f"{data2.std():.4f}")
                                st.metric(f"Varianza {group2}", f"{data2.var():.4f}")
                                st.metric(f"Tamaño {group2}", len(data2))
                            
                            st.metric("Diferencia de medias", f"{(data1.mean() - data2.mean()):.4f}")
                            
                            if p_value < alpha_nonpar:
                                st.success(f"✅ **Se rechaza la hipótesis nula** (p = {p_value:.4f} < α = {alpha_nonpar})")
                            else:
                                st.warning(f"✅ **No se rechaza la hipótesis nula** (p = {p_value:.4f} ≥ α = {alpha_nonpar})")
                            
                            # Visualización
                            st.subheader("📊 Comparación Visual")
                            fig, ax = plt.subplots(figsize=(10, 6))
                            plot_data = pd.DataFrame({
                                'Grupo': [group1] * len(data1) + [group2] * len(data2),
                                'Valor': list(data1) + list(data2)
                            })
                            sns.boxplot(data=plot_data, x='Grupo', y='Valor', ax=ax)
                            ax.set_title(f'Comparación de {welch_var} entre {group1} y {group2}\n(Prueba T de Welch - Varianzas desiguales)')
                            st.pyplot(fig)
                                    
                    except Exception as e:
                        st.error(f"Error en prueba T de Welch: {e}")
            else:
                st.warning("La variable categórica debe tener exactamente 2 grupos")
        
        elif nonpar_test == "Kruskal-Wallis" and numeric_cols and categorical_cols:
            st.subheader("Prueba de Kruskal-Wallis")
            
            kw_var = st.selectbox("Variable numérica:", numeric_cols, key="kw_var_tab8")
            kw_group = st.selectbox("Variable categórica:", categorical_cols, key="kw_group_tab8")
            
            if st.button("📊 Ejecutar Kruskal-Wallis", key="kw_button_tab8"):
                try:
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
                        h_stat, p_value = stats.kruskal(*groups_data)
                        
                        st.subheader("📋 Resultados")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Estadístico H", f"{h_stat:.4f}")
                        with col2:
                            st.metric("p-valor", f"{p_value:.4f}")
                        
                        # Estadísticas descriptivas
                        st.subheader("📊 Estadísticas por Grupo")
                        stats_by_group = []
                        for name, data in zip(group_names, groups_data):
                            stats_by_group.append({
                                'Grupo': name,
                                'n': len(data),
                                'Mediana': f"{data.median():.4f}",
                                'Rango intercuartílico': f"{data.quantile(0.75) - data.quantile(0.25):.4f}",
                                'Mínimo': f"{data.min():.4f}",
                                'Máximo': f"{data.max():.4f}"
                            })
                        
                        stats_df = pd.DataFrame(stats_by_group)
                        st.dataframe(stats_df, use_container_width=True)
                        
                        if p_value < alpha_nonpar:
                            st.success(f"✅ **Se rechaza la hipótesis nula** (p = {p_value:.4f} < α = {alpha_nonpar})")
                        else:
                            st.warning(f"✅ **No se rechaza la hipótesis nula** (p = {p_value:.4f} ≥ α = {alpha_nonpar})")
                        
                        # Visualización
                        st.subheader("📈 Visualización")
                        fig, ax = plt.subplots(figsize=(12, 6))
                        plot_data = []
                        for name, data in zip(group_names, groups_data):
                            for value in data:
                                plot_data.append({'Grupo': name, 'Valor': value})
                        
                        plot_df = pd.DataFrame(plot_data)
                        sns.boxplot(data=plot_df, x='Grupo', y='Valor', ax=ax)
                        ax.set_title(f'Kruskal-Wallis: {kw_var} por {kw_group}\n(H = {h_stat:.3f}, p = {p_value:.4f})')
                        ax.tick_params(axis='x', rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig)
                                
                except Exception as e:
                    st.error(f"Error en Kruskal-Wallis: {e}")
        
        elif nonpar_test == "Chi-cuadrado" and categorical_cols:
            st.subheader("Prueba de Chi-cuadrado")
            
            if len(categorical_cols) >= 2:
                chi_var1 = st.selectbox("Variable categórica 1:", categorical_cols, key="chi_var1_tab8")
                chi_var2 = st.selectbox("Variable categórica 2:", categorical_cols, key="chi_var2_tab8")
                
                if st.button("📊 Ejecutar Chi-cuadrado", key="chi_button_tab8"):
                    try:
                        contingency_table = pd.crosstab(df[chi_var1], df[chi_var2])
                        
                        if contingency_table.size == 0 or contingency_table.sum().sum() < 10:
                            st.error("No hay suficientes datos para realizar la prueba")
                        else:
                            chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
                            
                            expected_lt_5 = (expected < 5).sum()
                            total_cells = expected.size
                            percent_lt_5 = (expected_lt_5 / total_cells) * 100
                            
                            st.subheader("📋 Resultados")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Estadístico χ²", f"{chi2_stat:.4f}")
                            with col2:
                                st.metric("p-valor", f"{p_value:.4f}")
                            with col3:
                                st.metric("Grados de libertad", dof)
                            
                            if percent_lt_5 > 20:
                                st.warning(f"⚠️ **Advertencia:** {percent_lt_5:.1f}% de las celdas tienen frecuencia esperada < 5. Considera agrupar categorías.")
                            
                            st.subheader("📊 Tabla de Contingencia (Frecuencias Observadas)")
                            st.dataframe(contingency_table, use_container_width=True)
                            
                            st.subheader("📈 Porcentajes por Fila")
                            row_pct = contingency_table.div(contingency_table.sum(axis=1), axis=0) * 100
                            st.dataframe(row_pct.round(2), use_container_width=True)
                            
                            if p_value < alpha_nonpar:
                                st.success(f"✅ **Se rechaza la hipótesis nula** (p = {p_value:.4f} < α = {alpha_nonpar})")
                            else:
                                st.warning(f"✅ **No se rechaza la hipótesis nula** (p = {p_value:.4f} ≥ α = {alpha_nonpar})")
                                        
                    except Exception as e:
                        st.error(f"Error en Chi-cuadrado: {e}")
                else:
                    st.warning("Se necesitan al menos 2 variables categóricas")

# Mensaje final si no hay datos cargados
else:
    st.info("👆 Por favor, carga un archivo de datos en la barra lateral para comenzar el análisis.")

# Footer - CAMBIO 2: Generalizar el pie de página
st.markdown("---")
st.markdown(
    "**Analytics Statistics Bot** - Herramienta para análisis estadísticos aplicados"
)