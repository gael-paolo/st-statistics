# -*- coding: utf-8 -*-
"""
Analizador Estadístico Universal - VERSIÓN COMPLETA CORREGIDA
Correcciones aplicadas a todos los errores críticos y mejoras importantes
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
from statsmodels.stats.power import TTestIndPower, TTestPower
from ydata_profiling import ProfileReport
import io
from openai import OpenAI
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN DE LA APLICACIÓN
# ============================================================================

st.set_page_config(
    page_title="Analizador Estadístico Universal - CORREGIDO",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🤖 Analizador Estadístico Inteligente - VERSIÓN CORREGIDA")
st.markdown("""
**⚠️ VERSIÓN CON CORRECCIONES CRÍTICAS APLICADAS**  
Errores estadísticos corregidos y análisis mejorados para mayor precisión.
""")

# Sidebar
st.sidebar.header("🔧 Configuración")
openai_api_key = st.sidebar.text_input("Ingresa tu API Key de OpenAI:", type="password")
openai_client = None

if openai_api_key:
    try:
        openai_client = OpenAI(api_key=openai_api_key)
        st.sidebar.success("✅ OpenAI configurado")
    except:
        st.sidebar.error("Error configurando OpenAI")

def consultar_openai(prompt, max_tokens=2000, temperature=0.7, model="gpt-4"):
    """Consulta a OpenAI GPT"""
    try:
        if not openai_client:
            return "Error: Cliente OpenAI no configurado"
        
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Eres un experto en estadística aplicada."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# ============================================================================
# FUNCIONES CORREGIDAS Y MEJORADAS
# ============================================================================

# 1. CORRECCIÓN: Tamaño de muestra para población finita
def calculate_sample_size_corrected(population_size, margin_of_error=0.05, confidence_level=0.95, proportion=0.5):
    """Calcula el tamaño de muestra - CORREGIDO"""
    if not (0 < margin_of_error < 1):
        raise ValueError("Margen de error debe estar entre 0 y 1")
    if not (0 < confidence_level < 1):
        raise ValueError("Nivel de confianza debe estar entre 0 y 1")
    if not (0 <= proportion <= 1):
        raise ValueError("Proporción debe estar entre 0 y 1")
    if population_size <= 0:
        raise ValueError("Tamaño de población debe ser > 0")
    
    z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)
    n_infinite = (z_score**2 * proportion * (1 - proportion)) / (margin_of_error**2)
    
    if population_size < n_infinite:
        return int(population_size)
    
    n_adjusted = n_infinite / (1 + (n_infinite - 1) / population_size)
    return int(np.ceil(n_adjusted))

# 2. CORRECCIÓN: Muestreo estratificado mejorado
def generate_sample_corrected(df, sample_size, method="simple", stratify_col=None, random_state=None):
    """Genera muestra - CORREGIDO"""
    if isinstance(sample_size, float):
        if sample_size <= 0 or sample_size > 1:
            raise ValueError("Si 'sample_size' es porcentaje, debe estar entre 0 y 1")
        sample_size_abs = int(len(df) * sample_size)
    else:
        sample_size_abs = sample_size
    
    if sample_size_abs <= 0 or sample_size_abs > len(df):
        raise ValueError("Tamaño de muestra inválido")
    
    if method == "simple":
        return df.sample(n=sample_size_abs, random_state=random_state).reset_index(drop=True)
    
    elif method == "stratified":
        if stratify_col is None:
            raise ValueError("Se requiere 'stratify_col'")
        if stratify_col not in df.columns:
            raise ValueError(f"Columna '{stratify_col}' no existe")
        
        proportions = df[stratify_col].value_counts(normalize=True)
        sample_dfs = []
        
        for stratum, prop in proportions.items():
            stratum_size = max(1, int(np.round(sample_size_abs * prop)))
            stratum_data = df[df[stratify_col] == stratum]
            
            if len(stratum_data) >= stratum_size:
                sample_dfs.append(stratum_data.sample(n=stratum_size, random_state=random_state))
            else:
                sample_dfs.append(stratum_data)
        
        return pd.concat(sample_dfs, ignore_index=True)
    
    else:
        raise ValueError("Método debe ser 'simple' o 'stratified'")

# 3. CORRECCIÓN: Pruebas de normalidad mejoradas
def run_normality_tests_corrected(data, alpha=0.05):
    """Pruebas de normalidad - CORREGIDAS"""
    results = {}
    n = len(data)
    
    # Shapiro-Wilk
    if 3 <= n <= 5000:
        shapiro_stat, shapiro_p = shapiro(data)
        results['shapiro'] = {
            'test': 'Shapiro-Wilk',
            'statistic': shapiro_stat,
            'p_value': shapiro_p,
            'is_normal': shapiro_p > alpha,
            'weight': 3,
            'note': f'Apropiado para 3 ≤ n ≤ 5000'
        }
    else:
        results['shapiro'] = {
            'test': 'Shapiro-Wilk',
            'statistic': None,
            'p_value': None,
            'is_normal': None,
            'weight': 1,
            'note': f'n={n} fuera de rango recomendado'
        }
    
    # Anderson-Darling CORREGIDO
    try:
        ad_result = anderson(data, dist='norm')
        alpha_levels = [0.15, 0.10, 0.05, 0.025, 0.01]
        critical_values = ad_result.critical_values
        
        idx = min(range(len(alpha_levels)), key=lambda i: abs(alpha_levels[i] - alpha))
        critical_value = critical_values[idx]
        actual_alpha = alpha_levels[idx]
        
        results['anderson'] = {
            'test': 'Anderson-Darling',
            'statistic': ad_result.statistic,
            'critical_value': critical_value,
            'alpha_used': actual_alpha,
            'is_normal': ad_result.statistic < critical_value,
            'weight': 2,
            'note': f'α={actual_alpha:.3f} (cercano a α={alpha:.3f})'
        }
    except Exception as e:
        results['anderson'] = {
            'test': 'Anderson-Darling',
            'statistic': None,
            'critical_value': None,
            'is_normal': None,
            'note': f'Error: {str(e)}'
        }
    
    # Lilliefors
    if 4 <= n <= 1000:
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
            results['lilliefors'] = {'test': 'Lilliefors', 'error': 'No calculable'}
    
    return results

# 4. CORRECCIÓN Y MEJORA: Tamaño del efecto con intervalos de confianza
def calculate_effect_size_complete(test_type, **kwargs):
    """Calcula tamaño del efecto con IC - MEJORADO"""
    try:
        if test_type == "Una muestra":
            data = kwargs['data1']
            pop_mean = kwargs.get('pop_mean', 0)
            n = len(data)
            d = (np.mean(data) - pop_mean) / np.std(data, ddof=1)
            se = np.sqrt(1/n + d**2/(2*n))
            
        elif test_type == "Muestras independientes":
            data1 = kwargs['data1']
            data2 = kwargs['data2']
            n1, n2 = len(data1), len(data2)
            
            # Varianza agrupada corregida
            var1 = np.var(data1, ddof=1)
            var2 = np.var(data2, ddof=1)
            pooled_var = ((n1-1)*var1 + (n2-1)*var2) / (n1 + n2 - 2)
            
            if pooled_var == 0:
                return {'d': 0, 'se': 0, 'interpretation': 'No calculable (varianza cero)'}
            
            d = (np.mean(data1) - np.mean(data2)) / np.sqrt(pooled_var)
            
            # Corrección de Hedge para muestras pequeñas
            df = n1 + n2 - 2
            correction = 1 - 3/(4*df - 1) if df > 1 else 1
            g = d * correction
            
            # Error estándar CORREGIDO
            se = np.sqrt((n1 + n2)/(n1*n2) + d**2/(2*(n1 + n2)))
            
            d = g  # Usar la versión corregida
            
        elif test_type == "Muestras pareadas":
            paired_data = kwargs['paired_data']
            var_before = kwargs['var_before']
            var_after = kwargs['var_after']
            
            differences = paired_data[var_after] - paired_data[var_before]
            n = len(differences)
            d = np.mean(differences) / np.std(differences, ddof=1)
            
            # Para muestras pareadas, necesitamos la correlación
            corr = np.corrcoef(paired_data[var_before], paired_data[var_after])[0, 1]
            se = np.sqrt((1/n + d**2/(2*n)) * 2*(1-corr))
            
        else:
            return {'d': 0, 'se': 0, 'interpretation': 'Tipo de prueba no soportado'}
        
        # Calcular intervalo de confianza del 95%
        z = stats.norm.ppf(0.975)
        ci_lower = d - z * se
        ci_upper = d + z * se
        
        # Interpretación
        abs_d = abs(d)
        if abs_d < 0.2:
            interpretation = "Muy pequeño"
            color = "#808080"
        elif abs_d < 0.5:
            interpretation = "Pequeño"
            color = "#FF6B6B"
        elif abs_d < 0.8:
            interpretation = "Mediano"
            color = "#FFA726"
        elif abs_d < 1.2:
            interpretation = "Grande"
            color = "#4CAF50"
        else:
            interpretation = "Muy grande"
            color = "#2E7D32"
        
        # Determinar significancia
        effect_significant = (ci_lower > 0 and ci_upper > 0) or (ci_lower < 0 and ci_upper < 0)
        
        return {
            'd': d,
            'se': se,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'interpretation': interpretation,
            'color': color,
            'effect_significant': effect_significant,
            'n': n if 'n' in locals() else len(kwargs.get('data1', []))
        }
        
    except Exception as e:
        return {'d': 0, 'se': 0, 'interpretation': f'Error: {str(e)}'}

# 5. CORRECCIÓN: P-valores unilaterales en pruebas T
def adjust_one_tailed_pvalue(t_stat, p_value_two_tailed, alternative):
    """Ajusta p-valor para pruebas unilaterales - CORREGIDO"""
    if alternative == "two-sided":
        return p_value_two_tailed
    
    if alternative == "less":
        # H1: media < valor
        if t_stat <= 0:
            return p_value_two_tailed / 2
        else:
            return 1 - p_value_two_tailed / 2
    
    elif alternative == "greater":
        # H1: media > valor
        if t_stat >= 0:
            return p_value_two_tailed / 2
        else:
            return 1 - p_value_two_tailed / 2
    
    return p_value_two_tailed

# 6. NUEVA: Verificación de supuestos ANOVA
def check_anova_assumptions(data, group_var, value_var, alpha=0.05):
    """Verifica supuestos para ANOVA"""
    results = {'assumptions': {}, 'recommendations': []}
    
    groups = data.groupby(group_var)[value_var]
    group_data = [group.dropna().values for _, group in groups]
    group_names = list(groups.groups.keys())
    
    # 1. Normalidad por grupo
    normality_tests = []
    for name, gdata in zip(group_names, group_data):
        if len(gdata) >= 3:
            _, p = stats.shapiro(gdata)
            normality_tests.append({
                'group': name,
                'n': len(gdata),
                'p_value': p,
                'normal': p > alpha
            })
    
    results['assumptions']['normality'] = normality_tests
    
    # 2. Homogeneidad de varianzas
    if len(group_data) >= 2:
        levene_stat, levene_p = stats.levene(*group_data)
        results['assumptions']['homogeneity'] = {
            'statistic': levene_stat,
            'p_value': levene_p,
            'homogeneous': levene_p > alpha
        }
        
        # Ratio de varianzas
        variances = [np.var(g, ddof=1) for g in group_data if len(g) > 1]
        if variances:
            var_ratio = max(variances) / min(variances) if min(variances) > 0 else float('inf')
            results['assumptions']['variance_ratio'] = var_ratio
    
    # 3. Recomendaciones
    n_counts = [len(g) for g in group_data]
    min_n = min(n_counts)
    
    if min_n >= 30:
        results['recommendations'].append("✅ ANOVA robusto (n ≥ 30 en todos los grupos)")
    elif all(n.get('normal', False) for n in normality_tests):
        results['recommendations'].append("✅ Todos los grupos parecen normales")
    else:
        results['recommendations'].append("⚠️ Considerar prueba no paramétrica (Kruskal-Wallis)")
    
    if results['assumptions'].get('homogeneity', {}).get('homogeneous', True):
        results['recommendations'].append("✅ Varianzas homogéneas")
    else:
        results['recommendations'].append("⚠️ Considerar Welch's ANOVA (varianzas diferentes)")
    
    return results

# 7. CORRECCIÓN: Prueba Chi-cuadrado con verificaciones mejoradas
def chi_square_test_corrected(contingency_table, alpha=0.05):
    """Chi-cuadrado con verificaciones - CORREGIDO"""
    results = {'assumptions': {}, 'test_results': {}, 'recommendations': []}
    
    # Calcular prueba
    chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    results['test_results'] = {
        'chi2': chi2_stat,
        'p_value': p_value,
        'df': dof,
        'expected': expected
    }
    
    # Verificar supuestos
    total_cells = contingency_table.size
    expected_flat = expected.flatten()
    
    # Criterios modernos
    low_expected = np.sum(expected_flat < 5)
    percent_low = (low_expected / total_cells) * 100
    
    very_low_expected = np.sum(expected_flat < 1)
    
    results['assumptions'] = {
        'cells_lt_5': low_expected,
        'percent_lt_5': percent_low,
        'cells_lt_1': very_low_expected,
        'all_expected_ge_5': percent_low == 0,
        'no_expected_lt_1': very_low_expected == 0
    }
    
    # Recomendaciones
    if percent_low > 20:
        results['recommendations'].append("⚠️ Más del 20% de celdas con frecuencia esperada < 5")
    
    if very_low_expected > 0:
        results['recommendations'].append("⚠️ Algunas celdas con frecuencia esperada < 1")
    
    if contingency_table.shape == (2, 2) and np.any(expected < 10):
        results['recommendations'].append("⚠️ Para tabla 2x2, considerar Fisher's exact test")
    
    # Medidas de efecto
    n = contingency_table.sum().sum()
    if contingency_table.shape == (2, 2):
        phi = np.sqrt(chi2_stat / n)
        results['effect_size'] = {'phi': phi}
    else:
        min_dim = min(contingency_table.shape)
        cramers_v = np.sqrt(chi2_stat / (n * (min_dim - 1)))
        results['effect_size'] = {'cramers_v': cramers_v}
    
    return results

# 8. NUEVA: Análisis de potencia estadística
def calculate_power_analysis(effect_size, alpha=0.05, power=0.8, n=None, test_type='t_test_two_sample'):
    """Calcula potencia o tamaño muestral necesario"""
    try:
        if test_type == 't_test_two_sample':
            analysis = TTestIndPower()
            
            if n is None:
                # Calcular n necesario
                n_per_group = analysis.solve_power(
                    effect_size=effect_size,
                    alpha=alpha,
                    power=power,
                    ratio=1.0
                )
                return {
                    'required_n_per_group': int(np.ceil(n_per_group)),
                    'total_n': int(np.ceil(2 * n_per_group)),
                    'effect_size': effect_size,
                    'power': power,
                    'alpha': alpha
                }
            else:
                # Calcular potencia alcanzable
                actual_power = analysis.power(
                    effect_size=effect_size,
                    nobs1=n/2,
                    alpha=alpha,
                    ratio=1.0
                )
                return {
                    'achievable_power': actual_power,
                    'current_n_per_group': n/2,
                    'effect_size': effect_size,
                    'alpha': alpha
                }
    except Exception as e:
        return {'error': str(e)}
    
    return {'error': 'Tipo de prueba no soportado'}

# ============================================================================
# INTERFAZ DE USUARIO - ASISTENTE TEÓRICO
# ============================================================================

st.subheader("📚 Asistente Teórico en Estadística")

theory_question = st.text_area(
    "Haz tu pregunta sobre conceptos estadísticos:",
    placeholder="Ej: ¿Cuándo usar prueba T vs ANOVA? ¿Cómo interpreto un p-valor?",
    height=100
)

if st.button("Consultar teoría estadística") and theory_question:
    if openai_api_key:
        with st.spinner("Consultando al experto..."):
            prompt = f"""
            Como experto en estadística, responde:
            {theory_question}
            
            Incluye:
            1. Explicación conceptual clara
            2. Cuándo aplicarlo
            3. Supuestos requeridos
            4. Cómo interpretar resultados
            5. Limitaciones
            6. Ejemplos prácticos
            """
            response = consultar_openai(prompt, max_tokens=2000)
            st.markdown("---")
            st.markdown(response)
            st.markdown("---")
    else:
        st.error("Necesitas configurar OpenAI API Key")

# ============================================================================
# CARGA DE DATOS
# ============================================================================

st.sidebar.subheader("Carga de Datos")
uploaded_file = st.sidebar.file_uploader("Sube tu archivo Excel o CSV", type=['xlsx', 'csv', 'xls'])

@st.cache_data
def load_data(file):
    """Carga datos desde archivo"""
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, encoding='utf-8')
        else:
            df = pd.read_excel(file, engine='openpyxl')
        
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
        
        st.subheader("📋 Vista previa de los datos")
        st.dataframe(df.head(), use_container_width=True)
        
        # Identificar tipos
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

# ============================================================================
# SECCIÓN DE ANÁLISIS ESTADÍSTICOS CORREGIDOS
# ============================================================================

if df is not None:
    st.header("📊 Análisis Estadísticos - VERSIÓN CORREGIDA")
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "🎯 Muestreo", "📈 Descriptivos", "🔍 Normalidad", "📉 Correlaciones",
        "⚖️ Homogeneidad", "✅ Pruebas T", "📊 ANOVA", "🔄 No Paramétricas", "📋 Reportes"
    ])
    
    # ========================================================================
    # PESTAÑA 1: MUESTREO CORREGIDO
    # ========================================================================
    with tab1:
        st.subheader("🎯 Análisis de Muestreo - CORREGIDO")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📊 Generar Muestra - CORREGIDO")
            
            sample_method = st.radio(
                "Método de muestreo:",
                ["simple", "stratified"],
                format_func=lambda x: "🎲 Aleatorio Simple" if x == "simple" else "📊 Estratificado",
                horizontal=True
            )
            
            sample_size_type = st.radio(
                "Tipo de tamaño:",
                ["percentage", "absolute"],
                format_func=lambda x: "Porcentaje" if x == "percentage" else "Número absoluto"
            )
            
            if sample_size_type == "percentage":
                sample_size_input = st.slider("Porcentaje:", 1, 50, 20)
                sample_size = sample_size_input / 100.0
            else:
                sample_size_input = st.number_input("Tamaño:", 1, len(df), min(100, len(df)))
                sample_size = sample_size_input
            
            if sample_method == "stratified" and categorical_cols:
                stratify_column = st.selectbox("Variable para estratificación:", categorical_cols)
            else:
                stratify_column = None
            
            if st.button("🎲 Generar Muestra (Versión Corregida)", use_container_width=True):
                try:
                    sample_df = generate_sample_corrected(
                        df, sample_size, method=sample_method, 
                        stratify_col=stratify_column, random_state=42
                    )
                    
                    st.success(f"✅ Muestra generada: {len(sample_df):,} registros")
                    st.dataframe(sample_df.head(), use_container_width=True)
                    
                    # Comparación de distribuciones
                    if sample_method == "stratified" and stratify_column:
                        st.markdown("#### 📊 Distribución Estratificada")
                        sample_dist = sample_df[stratify_column].value_counts()
                        original_dist = df[stratify_column].value_counts()
                        
                        comparison = pd.DataFrame({
                            'Original': original_dist,
                            'Muestra': sample_dist,
                            '% Original': (original_dist / len(df)) * 100,
                            '% Muestra': (sample_dist / len(sample_df)) * 100
                        })
                        st.dataframe(comparison, use_container_width=True)
                
                except Exception as e:
                    st.error(f"❌ Error: {e}")
        
        with col2:
            st.markdown("### 🧮 Calcular Tamaño de Muestra - CORREGIDO")
            
            population_size = st.number_input("Tamaño población:", 1, value=len(df))
            margin_error = st.slider("Margen error (%):", 1, 10, 5) / 100.0
            confidence_level = st.slider("Confianza (%):", 80, 99, 95) / 100.0
            proportion = st.slider("Proporción (%):", 1, 99, 50) / 100.0
            
            if st.button("📐 Calcular (Versión Corregida)", use_container_width=True):
                try:
                    sample_size = calculate_sample_size_corrected(
                        population_size=population_size,
                        margin_of_error=margin_error,
                        confidence_level=confidence_level,
                        proportion=proportion
                    )
                    
                    st.success(f"🎯 **Tamaño de muestra recomendado:** `{sample_size:,}`")
                    
                    # Información adicional
                    z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)
                    st.info(f"""
                    **📋 Parámetros:**
                    - Población: {population_size:,}
                    - Margen error: ±{margin_error*100:.1f}%
                    - Confianza: {confidence_level*100:.1f}% (Z={z_score:.3f})
                    - Proporción: {proportion*100:.1f}%
                    
                    **📊 Comparación con dataset actual:**
                    - Tu dataset: {len(df):,} registros
                    - Cobertura: {(sample_size/len(df))*100:.1f}%
                    """)
                    
                except Exception as e:
                    st.error(f"❌ Error: {e}")
    
    # ========================================================================
    # PESTAÑA 2: DESCRIPTIVOS (SIN CAMBIOS CRÍTICOS)
    # ========================================================================
    with tab2:
        st.subheader("📈 Análisis Descriptivo")
        
        if numeric_cols:
            selected_numeric = st.multiselect(
                "Variables numéricas para análisis:",
                numeric_cols,
                default=numeric_cols[:min(5, len(numeric_cols))]
            )
            
            if selected_numeric:
                # Estadísticas básicas
                desc_stats = df[selected_numeric].describe().T
                desc_stats['cv'] = (desc_stats['std'] / desc_stats['mean']) * 100
                desc_stats['missing'] = df[selected_numeric].isnull().sum()
                desc_stats['missing_pct'] = (desc_stats['missing'] / len(df)) * 100
                
                st.dataframe(desc_stats.style.format({
                    'mean': '{:.4f}', 'std': '{:.4f}', '50%': '{:.4f}',
                    'cv': '{:.2f}%', 'missing_pct': '{:.2f}%'
                }), use_container_width=True)
                
                # Visualización
                if st.button("📊 Mostrar visualizaciones"):
                    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                    
                    # Histogramas
                    for i, var in enumerate(selected_numeric[:2]):
                        ax = axes[i] if len(selected_numeric) > 1 else axes
                        df[var].hist(ax=ax, bins=30, alpha=0.7)
                        ax.set_title(f'Distribución de {var}')
                        ax.set_xlabel(var)
                        ax.set_ylabel('Frecuencia')
                    
                    st.pyplot(fig)
    
    # ========================================================================
    # PESTAÑA 3: NORMALIDAD CORREGIDA
    # ========================================================================
    with tab3:
        st.subheader("🔍 Pruebas de Normalidad - CORREGIDAS")
        
        if numeric_cols:
            selected_var = st.selectbox("Selecciona variable:", numeric_cols)
            alpha_normal = st.slider("Nivel de significancia (α):", 0.01, 0.10, 0.05, 0.01)
            
            if st.button("📊 Ejecutar Pruebas (Versión Corregida)", use_container_width=True):
                try:
                    data = df[selected_var].dropna()
                    n = len(data)
                    
                    if n < 3:
                        st.error("Se necesitan al menos 3 observaciones")
                    else:
                        st.info(f"**📋 Tamaño de muestra:** {n:,} observaciones")
                        
                        # Ejecutar pruebas corregidas
                        results = run_normality_tests_corrected(data, alpha_normal)
                        
                        # Mostrar resultados
                        results_data = []
                        for key, result in results.items():
                            if result.get('statistic') is not None:
                                results_data.append({
                                    'Prueba': result['test'],
                                    'Estadístico': f"{result['statistic']:.4f}",
                                    'p-valor/Crítico': f"{result.get('p_value', result.get('critical_value', 'N/A')):.4f}",
                                    'Normal': '✅ Sí' if result['is_normal'] else '❌ No',
                                    'Nota': result.get('note', '')
                                })
                        
                        if results_data:
                            results_df = pd.DataFrame(results_data)
                            st.dataframe(results_df, use_container_width=True, hide_index=True)
                            
                            # Consenso ponderado
                            valid_tests = [r for r in results.values() if r.get('is_normal') is not None]
                            if valid_tests:
                                weights = [r.get('weight', 1) for r in valid_tests]
                                passed = [r['is_normal'] for r in valid_tests]
                                
                                consensus = sum(w for w, p in zip(weights, passed) if p) / sum(weights)
                                
                                st.metric("Consenso Ponderado", f"{consensus*100:.1f}%")
                                
                                # Conclusión
                                if consensus >= 0.7:
                                    st.success("✅ LOS DATOS PARECEN SEGUIR UNA DISTRIBUCIÓN NORMAL")
                                elif consensus >= 0.4:
                                    st.warning("⚠️ EVIDENCIA MIXTA SOBRE NORMALIDAD")
                                else:
                                    st.error("❌ LOS DATOS NO PARECEN SEGUIR UNA DISTRIBUCIÓN NORMAL")
                        
                        # Visualizaciones
                        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
                        
                        # Histograma
                        sns.histplot(data, kde=True, ax=ax1, bins=30)
                        ax1.set_title(f'Distribución de {selected_var}')
                        
                        # Q-Q Plot
                        stats.probplot(data, dist="norm", plot=ax2)
                        ax2.set_title('Q-Q Plot')
                        
                        # Boxplot
                        sns.boxplot(y=data, ax=ax3)
                        ax3.set_title('Boxplot')
                        
                        # ECDF
                        ecdf = np.arange(1, n + 1) / n
                        sorted_data = np.sort(data)
                        mu, sigma = data.mean(), data.std()
                        ax4.plot(sorted_data, ecdf, 'b-', label='Empírica')
                        ax4.plot(sorted_data, stats.norm.cdf(sorted_data, mu, sigma), 'r--', label='Teórica')
                        ax4.set_title('Función de Distribución Acumulada')
                        ax4.legend()
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                
                except Exception as e:
                    st.error(f"Error: {e}")
    
    # ========================================================================
    # PESTAÑA 4: CORRELACIONES
    # ========================================================================
    with tab4:
        st.subheader("📉 Análisis de Correlación")
        
        if len(numeric_cols) >= 2:
            var1 = st.selectbox("Variable X:", numeric_cols, key="corr1")
            var2 = st.selectbox("Variable Y:", [v for v in numeric_cols if v != var1], key="corr2")
            
            correlation_method = st.radio(
                "Método:",
                ["pearson", "spearman", "kendall"],
                format_func=lambda x: {
                    "pearson": "Pearson (lineal)",
                    "spearman": "Spearman (monotónica)",
                    "kendall": "Kendall (rangos)"
                }[x],
                horizontal=True
            )
            
            if st.button("🔍 Analizar Correlación"):
                try:
                    clean_data = df[[var1, var2]].dropna()
                    
                    if correlation_method == "pearson":
                        corr, p_value = stats.pearsonr(clean_data[var1], clean_data[var2])
                    elif correlation_method == "spearman":
                        corr, p_value = stats.spearmanr(clean_data[var1], clean_data[var2])
                    else:
                        corr, p_value = stats.kendalltau(clean_data[var1], clean_data[var2])
                    
                    # Resultados
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Coeficiente", f"{corr:.4f}")
                    col2.metric("p-valor", f"{p_value:.4f}")
                    col3.metric("n", len(clean_data))
                    
                    # Interpretación
                    abs_corr = abs(corr)
                    if abs_corr < 0.1:
                        strength = "muy débil"
                    elif abs_corr < 0.3:
                        strength = "débil"
                    elif abs_corr < 0.5:
                        strength = "moderada"
                    elif abs_corr < 0.7:
                        strength = "fuerte"
                    else:
                        strength = "muy fuerte"
                    
                    st.info(f"""
                    **📈 Interpretación:**
                    - **Fuerza:** {strength} (|r| = {abs_corr:.3f})
                    - **Dirección:** {'Positiva' if corr > 0 else 'Negativa'}
                    - **Significancia:** {'Significativa (p < 0.05)' if p_value < 0.05 else 'No significativa'}
                    """)
                    
                    # Gráfico
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.scatter(clean_data[var1], clean_data[var2], alpha=0.6)
                    
                    if correlation_method == "pearson":
                        # Línea de tendencia
                        z = np.polyfit(clean_data[var1], clean_data[var2], 1)
                        p = np.poly1d(z)
                        x_range = np.linspace(clean_data[var1].min(), clean_data[var1].max(), 100)
                        ax.plot(x_range, p(x_range), "r--", linewidth=2)
                    
                    ax.set_xlabel(var1)
                    ax.set_ylabel(var2)
                    ax.set_title(f'Correlación: {var1} vs {var2}')
                    ax.grid(True, alpha=0.3)
                    
                    st.pyplot(fig)
                
                except Exception as e:
                    st.error(f"Error: {e}")
    
    # ========================================================================
    # PESTAÑA 5: HOMOGENEIDAD
    # ========================================================================
    with tab5:
        st.subheader("⚖️ Homogeneidad de Varianzas")
        
        if numeric_cols and categorical_cols:
            num_var = st.selectbox("Variable numérica:", numeric_cols, key="homo_num")
            cat_var = st.selectbox("Variable categórica:", categorical_cols, key="homo_cat")
            
            if st.button("⚖️ Ejecutar Pruebas"):
                try:
                    groups_data = []
                    group_names = []
                    
                    for group in df[cat_var].dropna().unique():
                        group_data = df[df[cat_var] == group][num_var].dropna()
                        if len(group_data) >= 2:
                            groups_data.append(group_data)
                            group_names.append(str(group))
                    
                    if len(groups_data) < 2:
                        st.error("Se necesitan al menos 2 grupos")
                    else:
                        # Levene's test
                        levene_stat, levene_p = stats.levene(*groups_data)
                        bartlett_stat, bartlett_p = stats.bartlett(*groups_data)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Prueba", "Levene")
                            st.metric("Estadístico", f"{levene_stat:.4f}")
                            st.metric("p-valor", f"{levene_p:.4f}")
                        
                        with col2:
                            st.metric("Prueba", "Bartlett")
                            st.metric("Estadístico", f"{bartlett_stat:.4f}")
                            st.metric("p-valor", f"{bartlett_p:.4f}")
                        
                        # Interpretación
                        levene_homo = levene_p > 0.05
                        bartlett_homo = bartlett_p > 0.05
                        
                        if levene_homo and bartlett_homo:
                            st.success("✅ VARIANZAS HOMOGÉNEAS")
                        elif not levene_homo and not bartlett_homo:
                            st.error("❌ VARIANZAS NO HOMOGÉNEAS")
                        else:
                            st.warning("⚠️ RESULTADOS MIXTOS")
                
                except Exception as e:
                    st.error(f"Error: {e}")
    
    # ========================================================================
    # PESTAÑA 6: PRUEBAS T CORREGIDAS
    # ========================================================================
    with tab6:
        st.subheader("✅ Pruebas T - VERSIÓN CORREGIDA")
        
        test_type = st.radio(
            "Tipo de prueba T:",
            ["Una muestra", "Muestras independientes", "Muestras pareadas"],
            horizontal=True
        )
        
        alpha_ttest = st.slider("α:", 0.01, 0.10, 0.05, 0.01, key="alpha_ttest")
        alternative = st.selectbox(
            "Hipótesis alternativa:",
            ["two-sided", "less", "greater"],
            format_func=lambda x: {"two-sided": "Bilateral (≠)", "less": "Unilateral (<)", "greater": "Unilateral (>)"}[x]
        )
        
        # PRUEBA T UNA MUESTRA CORREGIDA
        if test_type == "Una muestra" and numeric_cols:
            var_onesample = st.selectbox("Variable:", numeric_cols, key="onesample_var")
            pop_mean = st.number_input("Media poblacional de referencia:", value=0.0)
            
            if st.button("📊 Ejecutar Prueba T (Corregida)", use_container_width=True):
                try:
                    data = df[var_onesample].dropna()
                    
                    if len(data) < 2:
                        st.error("Se necesitan al menos 2 observaciones")
                    else:
                        # Prueba T original
                        t_stat, p_value_two_tailed = stats.ttest_1samp(data, pop_mean)
                        
                        # CORRECCIÓN: Ajustar p-valor para unilateral
                        p_value = adjust_one_tailed_pvalue(t_stat, p_value_two_tailed, alternative)
                        
                        # Intervalo de confianza
                        ci_low, ci_high = stats.t.interval(
                            1 - alpha_ttest, len(data)-1, 
                            loc=data.mean(), scale=stats.sem(data)
                        )
                        
                        # Tamaño del efecto MEJORADO
                        effect = calculate_effect_size_complete(
                            "Una muestra", 
                            data1=data, 
                            pop_mean=pop_mean
                        )
                        
                        # Resultados
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("t", f"{t_stat:.4f}")
                        col2.metric("p-valor", f"{p_value:.4f}")
                        col3.metric("gl", len(data)-1)
                        col4.metric("Significativo", "✅ Sí" if p_value < alpha_ttest else "❌ No")
                        
                        # Estadísticas
                        st.markdown("### 📊 Estadísticas")
                        col_stats1, col_stats2 = st.columns(2)
                        with col_stats1:
                            st.metric("Media muestral", f"{data.mean():.4f}")
                            st.metric("Desviación", f"{data.std():.4f}")
                        with col_stats2:
                            st.metric("Media referencia", f"{pop_mean:.4f}")
                            st.metric(f"IC {int((1-alpha_ttest)*100)}%", f"[{ci_low:.4f}, {ci_high:.4f}]")
                        
                        # Tamaño del efecto con IC
                        if effect['interpretation'] != 'Error':
                            st.markdown("### 📈 Tamaño del Efecto (Cohen's d)")
                            col_eff1, col_eff2 = st.columns(2)
                            with col_eff1:
                                st.metric("d", f"{effect['d']:.4f}")
                                st.markdown(f"**Interpretación:** {effect['interpretation']}")
                            with col_eff2:
                                st.metric(f"IC 95% para d", f"[{effect['ci_lower']:.4f}, {effect['ci_upper']:.4f}]")
                                if effect['effect_significant']:
                                    st.success("✅ Efecto estadísticamente significativo")
                                else:
                                    st.info("ℹ️ Efecto no estadísticamente significativo")
                        
                        # Interpretación
                        st.info(f"""
                        **📝 Interpretación:**
                        - H₀: μ = {pop_mean}
                        - H₁: μ {'≠' if alternative == 'two-sided' else '<' if alternative == 'less' else '>'} {pop_mean}
                        - Decisión: {'Rechazar H₀' if p_value < alpha_ttest else 'No rechazar H₀'}
                        """)
                
                except Exception as e:
                    st.error(f"Error: {e}")
        
        # PRUEBA T INDEPENDIENTE CORREGIDA
        elif test_type == "Muestras independientes" and numeric_cols and categorical_cols:
            var_independent = st.selectbox("Variable numérica:", numeric_cols, key="indep_var")
            group_var = st.selectbox("Variable categórica (2 grupos):", categorical_cols, key="group_var")
            
            unique_groups = df[group_var].dropna().unique()
            if len(unique_groups) == 2:
                group1, group2 = unique_groups
                
                if st.button("📊 Ejecutar Prueba T Independiente (Corregida)"):
                    try:
                        data1 = df[df[group_var] == group1][var_independent].dropna()
                        data2 = df[df[group_var] == group2][var_independent].dropna()
                        
                        if len(data1) < 2 or len(data2) < 2:
                            st.error("Cada grupo necesita ≥ 2 observaciones")
                        else:
                            # Prueba de homogeneidad
                            _, levene_p = stats.levene(data1, data2)
                            equal_var = levene_p > 0.05
                            
                            # Prueba T
                            t_stat, p_value_two_tailed = stats.ttest_ind(data1, data2, equal_var=equal_var)
                            
                            # CORRECCIÓN: Ajustar p-valor
                            p_value = adjust_one_tailed_pvalue(t_stat, p_value_two_tailed, alternative)
                            
                            # Tamaño del efecto MEJORADO
                            effect = calculate_effect_size_complete(
                                "Muestras independientes",
                                data1=data1,
                                data2=data2
                            )
                            
                            # Resultados
                            st.markdown("### 📋 Resultados")
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("t", f"{t_stat:.4f}")
                            col2.metric("p-valor", f"{p_value:.4f}")
                            col3.metric("Prueba", "Welch" if not equal_var else "Student")
                            col4.metric("Significativo", "✅ Sí" if p_value < alpha_ttest else "❌ No")
                            
                            # Estadísticas por grupo
                            st.markdown("### 📊 Estadísticas por Grupo")
                            colg1, colg2 = st.columns(2)
                            with colg1:
                                st.metric(f"{group1} (n={len(data1)})", f"{data1.mean():.4f}")
                                st.metric(f"Desviación {group1}", f"{data1.std():.4f}")
                            with colg2:
                                st.metric(f"{group2} (n={len(data2)})", f"{data2.mean():.4f}")
                                st.metric(f"Desviación {group2}", f"{data2.std():.4f}")
                            
                            # Homogeneidad
                            st.metric("Prueba Levene p-valor", f"{levene_p:.4f}")
                            st.metric("Varianzas", "Homogéneas" if equal_var else "Diferentes")
                            
                            # Tamaño del efecto
                            if effect['interpretation'] != 'Error':
                                st.markdown("### 📈 Tamaño del Efecto")
                                cole1, cole2 = st.columns(2)
                                with cole1:
                                    st.metric("d", f"{effect['d']:.4f}")
                                    st.markdown(f"**Interpretación:** {effect['interpretation']}")
                                with cole2:
                                    st.metric(f"IC 95%", f"[{effect['ci_lower']:.4f}, {effect['ci_upper']:.4f}]")
                            
                            # Visualización
                            fig, ax = plt.subplots(figsize=(10, 6))
                            plot_data = pd.DataFrame({
                                'Grupo': [group1]*len(data1) + [group2]*len(data2),
                                'Valor': list(data1) + list(data2)
                            })
                            sns.boxplot(data=plot_data, x='Grupo', y='Valor', ax=ax)
                            ax.set_title(f'Comparación: {group1} vs {group2}')
                            ax.set_ylabel(var_independent)
                            ax.grid(True, alpha=0.3)
                            st.pyplot(fig)
                    
                    except Exception as e:
                        st.error(f"Error: {e}")
    
    # ========================================================================
    # PESTAÑA 7: ANOVA CORREGIDO
    # ========================================================================
    with tab7:
        st.subheader("📊 ANOVA - CON VERIFICACIÓN DE SUPUESTOS")
        
        if numeric_cols and categorical_cols:
            num_var = st.selectbox("Variable numérica:", numeric_cols, key="anova_num")
            cat_var = st.selectbox("Variable categórica:", categorical_cols, key="anova_cat")
            
            if st.button("📊 Ejecutar ANOVA con Verificación"):
                try:
                    # Verificar supuestos primero
                    st.markdown("### 🔍 Verificación de Supuestos")
                    assumptions = check_anova_assumptions(df, cat_var, num_var)
                    
                    for rec in assumptions.get('recommendations', []):
                        if "✅" in rec:
                            st.success(rec)
                        elif "⚠️" in rec:
                            st.warning(rec)
                        else:
                            st.info(rec)
                    
                    # Separar datos por grupo
                    groups = df.groupby(cat_var)[num_var]
                    group_data = [group.dropna().values for _, group in groups]
                    group_names = list(groups.groups.keys())
                    
                    if len(group_data) < 2:
                        st.error("Se necesitan al menos 2 grupos")
                    else:
                        # Ejecutar ANOVA
                        f_stat, p_value = stats.f_oneway(*group_data)
                        
                        # Resultados
                        st.markdown("### 📋 Resultados ANOVA")
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("F", f"{f_stat:.4f}")
                        col2.metric("p-valor", f"{p_value:.4f}")
                        col3.metric("Grupos", len(group_data))
                        col4.metric("Significativo", "✅ Sí" if p_value < 0.05 else "❌ No")
                        
                        # Estadísticas por grupo
                        st.markdown("### 📊 Estadísticas por Grupo")
                        stats_data = []
                        for name, data in zip(group_names, group_data):
                            stats_data.append({
                                'Grupo': name,
                                'n': len(data),
                                'Media': f"{np.mean(data):.4f}",
                                'Desviación': f"{np.std(data):.4f}"
                            })
                        
                        stats_df = pd.DataFrame(stats_data)
                        st.dataframe(stats_df, use_container_width=True)
                        
                        # Interpretación
                        if p_value < 0.05:
                            st.success(f"✅ Existen diferencias significativas entre los grupos de {cat_var}")
                            
                            # Post-hoc si es significativo y hay más de 2 grupos
                            if len(group_data) > 2:
                                st.markdown("### 🔍 Comparaciones Múltiples (Tukey HSD)")
                                try:
                                    tukey_data = df[[num_var, cat_var]].dropna()
                                    tukey = pairwise_tukeyhsd(tukey_data[num_var], tukey_data[cat_var], alpha=0.05)
                                    result_df = pd.DataFrame(
                                        data=tukey._results_table.data[1:],
                                        columns=tukey._results_table.data[0]
                                    )
                                    st.dataframe(result_df, use_container_width=True)
                                except:
                                    st.warning("No se pudo realizar análisis post-hoc")
                        else:
                            st.info(f"ℹ️ No hay diferencias significativas entre los grupos de {cat_var}")
                        
                        # Visualización
                        fig, ax = plt.subplots(figsize=(10, 6))
                        plot_data = []
                        for name, data in zip(group_names, group_data):
                            for value in data:
                                plot_data.append({'Grupo': name, 'Valor': value})
                        
                        plot_df = pd.DataFrame(plot_data)
                        sns.boxplot(data=plot_df, x='Grupo', y='Valor', ax=ax)
                        ax.set_title(f'ANOVA: {num_var} por {cat_var}')
                        ax.set_xlabel(cat_var)
                        ax.set_ylabel(num_var)
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                
                except Exception as e:
                    st.error(f"Error: {e}")
    
    # ========================================================================
    # PESTAÑA 8: PRUEBAS NO PARAMÉTRICAS
    # ========================================================================
    with tab8:
        st.subheader("🔄 Pruebas No Paramétricas")
        
        nonpar_test = st.radio(
            "Selecciona prueba:",
            ["Mann-Whitney U", "Wilcoxon", "Kruskal-Wallis", "Chi-cuadrado"],
            horizontal=True
        )
        
        alpha_nonpar = st.slider("α:", 0.01, 0.10, 0.05, 0.01, key="alpha_nonpar")
        
        # Mann-Whitney U
        if nonpar_test == "Mann-Whitney U" and numeric_cols and categorical_cols:
            mw_var = st.selectbox("Variable:", numeric_cols, key="mw_var")
            mw_group = st.selectbox("Grupo (2 categorías):", categorical_cols, key="mw_group")
            
            unique_groups = df[mw_group].dropna().unique()
            if len(unique_groups) == 2:
                group1, group2 = unique_groups
                
                if st.button("Ejecutar Mann-Whitney U"):
                    try:
                        data1 = df[df[mw_group] == group1][mw_var].dropna()
                        data2 = df[df[mw_group] == group2][mw_var].dropna()
                        
                        u_stat, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("U", f"{u_stat:.4f}")
                        col2.metric("p-valor", f"{p_value:.4f}")
                        col3.metric("Significativo", "✅ Sí" if p_value < alpha_nonpar else "❌ No")
                        
                        # Estadísticas
                        col_stats1, col_stats2 = st.columns(2)
                        with col_stats1:
                            st.metric(f"Mediana {group1}", f"{np.median(data1):.4f}")
                        with col_stats2:
                            st.metric(f"Mediana {group2}", f"{np.median(data2):.4f}")
                    
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        # Chi-cuadrado CORREGIDO
        elif nonpar_test == "Chi-cuadrado" and len(categorical_cols) >= 2:
            chi_var1 = st.selectbox("Variable 1:", categorical_cols, key="chi1")
            chi_var2 = st.selectbox("Variable 2:", [c for c in categorical_cols if c != chi_var1], key="chi2")
            
            if st.button("Ejecutar Chi-cuadrado (Versión Corregida)"):
                try:
                    contingency = pd.crosstab(df[chi_var1], df[chi_var2])
                    
                    st.markdown("### 📊 Tabla de Contingencia")
                    st.dataframe(contingency, use_container_width=True)
                    
                    # Prueba corregida
                    results = chi_square_test_corrected(contingency.values, alpha_nonpar)
                    
                    st.markdown("### 📋 Resultados")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("χ²", f"{results['test_results']['chi2']:.4f}")
                    col2.metric("p-valor", f"{results['test_results']['p_value']:.4f}")
                    col3.metric("gl", results['test_results']['df'])
                    
                    # Supuestos
                    st.markdown("### 🔍 Verificación de Supuestos")
                    if results['assumptions'].get('all_expected_ge_5', False):
                        st.success("✅ Todas las frecuencias esperadas ≥ 5")
                    else:
                        st.warning(f"⚠️ {results['assumptions'].get('percent_lt_5', 0):.1f}% de celdas con frecuencia esperada < 5")
                    
                    if results['assumptions'].get('no_expected_lt_1', False):
                        st.success("✅ Ninguna frecuencia esperada < 1")
                    else:
                        st.warning(f"⚠️ {results['assumptions'].get('cells_lt_1', 0)} celdas con frecuencia esperada < 1")
                    
                    # Recomendaciones
                    if results['recommendations']:
                        st.markdown("### 💡 Recomendaciones")
                        for rec in results['recommendations']:
                            st.warning(rec)
                    
                    # Medida de efecto
                    if 'effect_size' in results:
                        if 'phi' in results['effect_size']:
                            st.metric("Phi (efecto)", f"{results['effect_size']['phi']:.4f}")
                        elif 'cramers_v' in results['effect_size']:
                            st.metric("Cramer's V", f"{results['effect_size']['cramers_v']:.4f}")
                
                except Exception as e:
                    st.error(f"Error: {e}")
    
    # ========================================================================
    # PESTAÑA 9: REPORTES Y ANÁLISIS DE POTENCIA
    # ========================================================================
    with tab9:
        st.subheader("📋 Reportes y Análisis de Potencia")
        
        st.markdown("### ⚡ Análisis de Potencia Estadística")
        
        col_power1, col_power2 = st.columns(2)
        
        with col_power1:
            effect_size_input = st.number_input("Tamaño del efecto (Cohen's d):", 0.1, 2.0, 0.5, 0.1)
            alpha_power = st.number_input("Nivel α:", 0.01, 0.10, 0.05, 0.01)
            power_target = st.number_input("Potencia deseada:", 0.5, 0.99, 0.8, 0.05)
        
        with col_power2:
            current_n = st.number_input("Tamaño muestral actual (opcional):", 10, 10000, 100, 10)
            test_type_power = st.selectbox("Tipo de prueba:", ["t_test_two_sample"])
        
        if st.button("🔍 Calcular Potencia/Tamaño Muestral"):
            try:
                if current_n:
                    # Calcular potencia alcanzable
                    power_result = calculate_power_analysis(
                        effect_size=effect_size_input,
                        alpha=alpha_power,
                        n=current_n,
                        test_type=test_type_power
                    )
                    
                    if 'achievable_power' in power_result:
                        st.success(f"### 📊 Potencia Alcanzable: {power_result['achievable_power']:.3f}")
                        
                        if power_result['achievable_power'] >= power_target:
                            st.success(f"✅ Potencia suficiente (≥ {power_target})")
                        else:
                            st.warning(f"⚠️ Potencia insuficiente (< {power_target})")
                            
                            # Calcular n necesario
                            n_result = calculate_power_analysis(
                                effect_size=effect_size_input,
                                alpha=alpha_power,
                                power=power_target,
                                test_type=test_type_power
                            )
                            
                            if 'required_n_per_group' in n_result:
                                st.info(f"""
                                **Para alcanzar potencia de {power_target}:**
                                - Necesitas {n_result['required_n_per_group']} por grupo
                                - Total: {n_result['total_n']} observaciones
                                """)
                else:
                    # Calcular n necesario
                    n_result = calculate_power_analysis(
                        effect_size=effect_size_input,
                        alpha=alpha_power,
                        power=power_target,
                        test_type=test_type_power
                    )
                    
                    if 'required_n_per_group' in n_result:
                        st.success(f"""
                        ### 🎯 Tamaño Muestral Requerido
                        
                        **Para detectar efecto d = {effect_size_input:.2f} con:**
                        - α = {alpha_power}
                        - Potencia = {power_target}
                        
                        **Se necesitan:**
                        - {n_result['required_n_per_group']} observaciones por grupo
                        - {n_result['total_n']} observaciones totales
                        
                        **Tu dataset actual:**
                        - Tiene {len(df)} observaciones
                        - Cobertura: {(n_result['total_n']/len(df))*100:.1f}% del requerido
                        """)
                        
                        if n_result['total_n'] <= len(df):
                            st.success("✅ Tu dataset es suficiente para este análisis")
                        else:
                            deficit = n_result['total_n'] - len(df)
                            st.warning(f"⚠️ Necesitas {deficit} observaciones más")
            
            except Exception as e:
                st.error(f"Error en análisis de potencia: {e}")
        
        # Reporte EDA
        st.markdown("---")
        st.markdown("### 📊 Reporte Exploratorio de Datos")
        
        if st.button("📈 Generar Reporte EDA", use_container_width=True):
            with st.spinner("Generando reporte..."):
                try:
                    profile = ProfileReport(df, title="Análisis Exploratorio", explorative=True)
                    html_content = profile.to_html()
                    
                    st.success("✅ Reporte generado")
                    
                    # Descargar
                    st.download_button(
                        label="📥 Descargar Reporte HTML",
                        data=html_content,
                        file_name=f"reporte_eda_{datetime.now().strftime('%Y%m%d')}.html",
                        mime="text/html",
                        use_container_width=True
                    )
                    
                    # Vista previa
                    if st.checkbox("Mostrar vista previa del reporte"):
                        st.components.v1.html(html_content, height=600, scrolling=True)
                
                except Exception as e:
                    st.error(f"Error: {e}")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
**📊 Analizador Estadístico Universal - VERSIÓN CORREGIDA**  
✅ **Correcciones aplicadas:**
1. Cálculo de tamaño de muestra para población finita
2. Pruebas de normalidad mejoradas
3. Tamaños del efecto con intervalos de confianza
4. P-valores unilaterales en pruebas T
5. Verificación de supuestos para ANOVA
6. Prueba Chi-cuadrado con criterios modernos
7. Análisis de potencia estadística

**🔒** Tus datos se procesan localmente. **📚** Para más información, consulta la documentación.
""")