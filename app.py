# -*- coding: utf-8 -*-
"""
Analizador Estadístico Universal - VERSIÓN SIMPLIFICADA Y ROBUSTA
Sin dependencias problemáticas - Todo incluido
"""

import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import anderson, shapiro
import io
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

st.set_page_config(
    page_title="Analizador Estadístico Universal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🤖 Analizador Estadístico Inteligente")
st.markdown("""
Herramienta completa para análisis estadísticos descriptivos e inferenciales.
Todas las funciones están autocontenidas - sin dependencias externas problemáticas.
""")

# ============================================================================
# FUNCIONES ESTADÍSTICAS CORREGIDAS Y ROBUSTAS
# ============================================================================

def calculate_sample_size_corrected(population_size, margin_of_error=0.05, confidence_level=0.95, proportion=0.5):
    """Calcula tamaño de muestra - CORREGIDO"""
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

def run_normality_tests_robust(data, alpha=0.05):
    """Pruebas de normalidad robustas"""
    results = {}
    n = len(data)
    
    # Shapiro-Wilk
    if 3 <= n <= 5000:
        try:
            shapiro_stat, shapiro_p = shapiro(data)
            results['shapiro'] = {
                'test': 'Shapiro-Wilk',
                'statistic': shapiro_stat,
                'p_value': shapiro_p,
                'is_normal': shapiro_p > alpha,
                'note': f'Apropiado para 3 ≤ n ≤ 5000'
            }
        except:
            results['shapiro'] = {'test': 'Shapiro-Wilk', 'error': 'No calculable'}
    else:
        results['shapiro'] = {'test': 'Shapiro-Wilk', 'note': f'n={n} fuera de rango'}
    
    # Anderson-Darling
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
            'note': f'α={actual_alpha:.3f} usado'
        }
    except:
        results['anderson'] = {'test': 'Anderson-Darling', 'error': 'No calculable'}
    
    # D'Agostino K^2 para n >= 20
    if n >= 20:
        try:
            k2_stat, k2_p = stats.normaltest(data)
            results['dagostino'] = {
                'test': "D'Agostino K^2",
                'statistic': k2_stat,
                'p_value': k2_p,
                'is_normal': k2_p > alpha,
                'note': f'Bueno para n ≥ 20'
            }
        except:
            pass
    
    return results

def calculate_effect_size_with_ci(test_type, **kwargs):
    """Calcula tamaño del efecto con intervalo de confianza"""
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
            
            var1 = np.var(data1, ddof=1)
            var2 = np.var(data2, ddof=1)
            pooled_var = ((n1-1)*var1 + (n2-1)*var2) / (n1 + n2 - 2)
            
            if pooled_var == 0:
                return {'d': 0, 'se': 0, 'interpretation': 'Varianza cero'}
            
            d = (np.mean(data1) - np.mean(data2)) / np.sqrt(pooled_var)
            se = np.sqrt((n1 + n2)/(n1*n2) + d**2/(2*(n1 + n2)))
            
        elif test_type == "Muestras pareadas":
            paired_data = kwargs['paired_data']
            var_before = kwargs['var_before']
            var_after = kwargs['var_after']
            
            differences = paired_data[var_after] - paired_data[var_before]
            n = len(differences)
            d = np.mean(differences) / np.std(differences, ddof=1)
            se = np.sqrt(1/n + d**2/(2*n))
            
        else:
            return {'d': 0, 'se': 0, 'interpretation': 'Tipo no soportado'}
        
        # Intervalo de confianza 95%
        z = stats.norm.ppf(0.975)
        ci_lower = d - z * se
        ci_upper = d + z * se
        
        # Interpretación
        abs_d = abs(d)
        if abs_d < 0.2:
            interpretation = "Muy pequeño"
        elif abs_d < 0.5:
            interpretation = "Pequeño"
        elif abs_d < 0.8:
            interpretation = "Mediano"
        elif abs_d < 1.2:
            interpretation = "Grande"
        else:
            interpretation = "Muy grande"
        
        return {
            'd': d,
            'se': se,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'interpretation': interpretation,
            'significant': (ci_lower > 0 and ci_upper > 0) or (ci_lower < 0 and ci_upper < 0)
        }
        
    except Exception as e:
        return {'d': 0, 'se': 0, 'interpretation': f'Error: {str(e)[:50]}'}

def adjust_one_tailed_pvalue(t_stat, p_value_two_tailed, alternative):
    """Ajusta p-valor para pruebas unilaterales"""
    if alternative == "two-sided":
        return p_value_two_tailed
    
    if alternative == "less":
        return p_value_two_tailed / 2 if t_stat <= 0 else 1 - p_value_two_tailed / 2
    
    if alternative == "greater":
        return p_value_two_tailed / 2 if t_stat >= 0 else 1 - p_value_two_tailed / 2
    
    return p_value_two_tailed

def check_anova_assumptions_simple(data, group_var, value_var, alpha=0.05):
    """Verificación simple de supuestos ANOVA"""
    results = {'warnings': [], 'recommendations': []}
    
    groups = data.groupby(group_var)[value_var]
    group_data = [group.dropna().values for _, group in groups]
    
    # Normalidad por grupo
    normality_issues = 0
    for i, gdata in enumerate(group_data):
        if len(gdata) >= 3:
            _, p = stats.shapiro(gdata)
            if p < alpha:
                normality_issues += 1
    
    if normality_issues > 0:
        n_counts = [len(g) for g in group_data]
        min_n = min(n_counts)
        if min_n >= 30:
            results['recommendations'].append("ANOVA robusto (n ≥ 30)")
        else:
            results['warnings'].append(f"{normality_issues} grupos no normales")
            results['recommendations'].append("Considerar Kruskal-Wallis")
    
    # Homogeneidad de varianzas
    if len(group_data) >= 2:
        try:
            _, levene_p = stats.levene(*group_data)
            if levene_p < alpha:
                results['warnings'].append("Varianzas no homogéneas")
                results['recommendations'].append("Considerar Welch's ANOVA")
        except:
            pass
    
    return results

def chi_square_assumptions_check(contingency_table):
    """Verifica supuestos para Chi-cuadrado"""
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    
    results = {
        'chi2': chi2,
        'p_value': p,
        'df': dof,
        'warnings': [],
        'recommendations': []
    }
    
    # Verificar frecuencias esperadas
    expected_flat = expected.flatten()
    low_expected = np.sum(expected_flat < 5)
    total_cells = contingency_table.size
    
    if low_expected > 0:
        percent_low = (low_expected / total_cells) * 100
        if percent_low > 20:
            results['warnings'].append(f"{percent_low:.1f}% celdas con frecuencia esperada < 5")
        
        if np.any(expected_flat < 1):
            results['warnings'].append("Algunas celdas con frecuencia esperada < 1")
    
    # Para tablas 2x2
    if contingency_table.shape == (2, 2) and np.any(expected < 10):
        results['recommendations'].append("Para tabla 2x2, considerar Fisher's exact test")
    
    return results

# ============================================================================
# INTERFAZ PRINCIPAL
# ============================================================================

# Sidebar para carga de datos
st.sidebar.header("📁 Carga de Datos")
uploaded_file = st.sidebar.file_uploader(
    "Sube tu archivo de datos",
    type=['csv', 'xlsx', 'xls'],
    help="Formatos soportados: CSV, Excel"
)

# Función para cargar datos
def load_data_simple(file):
    """Carga datos de forma robusta"""
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, encoding='utf-8')
        else:
            # Para Excel, intentar diferentes motores
            try:
                df = pd.read_excel(file, engine='openpyxl')
            except:
                df = pd.read_excel(file, engine='xlrd')
        
        # Limpiar columnas
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        return df
    except Exception as e:
        st.error(f"Error cargando archivo: {str(e)[:100]}")
        return None

# Cargar datos si se subió archivo
df = None
if uploaded_file is not None:
    df = load_data_simple(uploaded_file)
    if df is not None:
        st.sidebar.success(f"✅ Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
        
        # Mostrar vista previa
        with st.expander("📋 Vista previa de datos", expanded=True):
            st.dataframe(df.head(), use_container_width=True)
            
            # Información básica
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Filas", df.shape[0])
            col2.metric("Columnas", df.shape[1])
            col3.metric("Valores faltantes", df.isnull().sum().sum())
            memory_mb = df.memory_usage(deep=True).sum() / 1024**2
            col4.metric("Memoria", f"{memory_mb:.1f} MB")
        
        # Identificar tipos de variables
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Mostrar tipos
        with st.expander("🔍 Tipos de variables"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Variables numéricas:**")
                if numeric_cols:
                    for col in numeric_cols[:10]:
                        st.write(f"- {col}: {df[col].dtype}")
                    if len(numeric_cols) > 10:
                        st.write(f"... y {len(numeric_cols)-10} más")
                else:
                    st.write("No hay variables numéricas")
            
            with col2:
                st.write("**Variables categóricas:**")
                if categorical_cols:
                    for col in categorical_cols[:10]:
                        st.write(f"- {col}: {df[col].dtype}")
                    if len(categorical_cols) > 10:
                        st.write(f"... y {len(categorical_cols)-10} más")
                else:
                    st.write("No hay variables categóricas")

# ============================================================================
# MENÚ DE ANÁLISIS
# ============================================================================

if df is not None:
    st.header("📊 Análisis Estadísticos")
    
    # Crear pestañas
    tab_names = [
        "📈 Descriptivos", 
        "🎯 Muestreo", 
        "🔍 Normalidad", 
        "📉 Correlación",
        "✅ Pruebas T",
        "📊 ANOVA",
        "🔄 No Paramétricas",
        "📋 Reporte"
    ]
    
    tabs = st.tabs(tab_names)
    
    # ========================================================================
    # PESTAÑA 1: ANÁLISIS DESCRIPTIVO
    # ========================================================================
    with tabs[0]:
        st.subheader("📈 Análisis Descriptivo")
        
        if numeric_cols:
            selected_vars = st.multiselect(
                "Selecciona variables numéricas:",
                numeric_cols,
                default=numeric_cols[:min(3, len(numeric_cols))]
            )
            
            if selected_vars:
                # Estadísticas descriptivas
                desc = df[selected_vars].describe().T
                desc['CV%'] = (desc['std'] / desc['mean']) * 100
                desc['IQR'] = desc['75%'] - desc['25%']
                desc['Missing'] = df[selected_vars].isnull().sum()
                
                # Formatear
                desc_display = desc[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'CV%', 'IQR', 'Missing']]
                desc_display.columns = ['n', 'Media', 'Desv', 'Mín', 'Q1', 'Mediana', 'Q3', 'Máx', 'CV%', 'IQR', 'Faltantes']
                
                st.dataframe(
                    desc_display.style.format({
                        'Media': '{:.4f}',
                        'Desv': '{:.4f}',
                        'CV%': '{:.2f}%',
                        'IQR': '{:.4f}'
                    }),
                    use_container_width=True
                )
                
                # Visualizaciones
                if st.button("Mostrar gráficos", key="desc_viz"):
                    n_vars = len(selected_vars)
                    n_cols = min(2, n_vars)
                    n_rows = (n_vars + n_cols - 1) // n_cols
                    
                    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
                    if n_vars == 1:
                        axes = np.array([axes])
                    
                    axes = axes.flatten()
                    
                    for i, var in enumerate(selected_vars):
                        if i < len(axes):
                            ax = axes[i]
                            data_clean = df[var].dropna()
                            
                            # Histograma
                            ax.hist(data_clean, bins=30, alpha=0.7, edgecolor='black')
                            ax.set_title(f'Distribución de {var}')
                            ax.set_xlabel(var)
                            ax.set_ylabel('Frecuencia')
                            ax.grid(True, alpha=0.3)
                            
                            # Agregar estadísticas
                            stats_text = f"n = {len(data_clean)}\n"
                            stats_text += f"μ = {data_clean.mean():.2f}\n"
                            stats_text += f"σ = {data_clean.std():.2f}"
                            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                                   fontsize=9, verticalalignment='top',
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
                    
                    # Ocultar ejes vacíos
                    for i in range(len(selected_vars), len(axes)):
                        axes[i].set_visible(False)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
        
        else:
            st.warning("No hay variables numéricas para análisis descriptivo")
    
    # ========================================================================
    # PESTAÑA 2: MUESTREO CORREGIDO
    # ========================================================================
    with tabs[1]:
        st.subheader("🎯 Análisis de Muestreo - CORREGIDO")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Generar Muestra")
            
            sample_method = st.selectbox(
                "Método:",
                ["Aleatorio simple", "Estratificado"],
                key="sample_method"
            )
            
            sample_size_type = st.radio(
                "Tamaño:",
                ["Porcentaje", "Número absoluto"],
                horizontal=True,
                key="sample_size_type"
            )
            
            if sample_size_type == "Porcentaje":
                sample_percent = st.slider("Porcentaje:", 1, 100, 20)
                sample_size = sample_percent / 100.0
            else:
                sample_abs = st.number_input("Número:", 1, len(df), min(100, len(df)))
                sample_size = sample_abs
            
            if sample_method == "Estratificado" and categorical_cols:
                stratify_var = st.selectbox("Variable para estratificar:", categorical_cols)
            else:
                stratify_var = None
            
            if st.button("🎲 Generar Muestra", key="generate_sample"):
                try:
                    method_code = "simple" if sample_method == "Aleatorio simple" else "stratified"
                    
                    # Usar función simple para muestreo
                    if method_code == "simple":
                        if isinstance(sample_size, float):
                            n_samples = int(len(df) * sample_size)
                        else:
                            n_samples = sample_size
                        sample_df = df.sample(n=n_samples, random_state=42)
                    else:
                        # Muestreo estratificado simple
                        if stratify_var:
                            proportions = df[stratify_var].value_counts(normalize=True)
                            sample_dfs = []
                            
                            for stratum, prop in proportions.items():
                                stratum_data = df[df[stratify_var] == stratum]
                                if isinstance(sample_size, float):
                                    n_stratum = max(1, int(len(stratum_data) * sample_size))
                                else:
                                    n_stratum = max(1, int(sample_size * prop))
                                
                                if len(stratum_data) >= n_stratum:
                                    sample_dfs.append(stratum_data.sample(n=n_stratum, random_state=42))
                                else:
                                    sample_dfs.append(stratum_data)
                            
                            sample_df = pd.concat(sample_dfs, ignore_index=True)
                        else:
                            st.error("Selecciona variable para estratificación")
                            sample_df = None
                    
                    if sample_df is not None:
                        st.success(f"✅ Muestra generada: {len(sample_df)} registros")
                        st.dataframe(sample_df.head(), use_container_width=True)
                        
                        # Estadísticas de la muestra
                        col_s1, col_s2, col_s3 = st.columns(3)
                        col_s1.metric("Tamaño muestra", len(sample_df))
                        col_s2.metric("% del total", f"{(len(sample_df)/len(df))*100:.1f}%")
                        col_s3.metric("Reducción", len(df) - len(sample_df))
                
                except Exception as e:
                    st.error(f"Error: {str(e)[:100]}")
        
        with col2:
            st.markdown("#### Calcular Tamaño de Muestra - CORREGIDO")
            
            population = st.number_input("Tamaño población:", 1, value=len(df), key="pop_size")
            margin_error = st.slider("Margen error (%):", 1, 20, 5, key="margin_error") / 100.0
            confidence = st.slider("Confianza (%):", 80, 99, 95, key="confidence") / 100.0
            proportion = st.slider("Proporción (%):", 1, 99, 50, key="proportion") / 100.0
            
            if st.button("📐 Calcular Tamaño", key="calc_sample_size"):
                try:
                    n_required = calculate_sample_size_corrected(
                        population_size=population,
                        margin_of_error=margin_error,
                        confidence_level=confidence,
                        proportion=proportion
                    )
                    
                    st.success(f"**Tamaño muestral requerido:** {n_required:,}")
                    
                    # Comparación con dataset actual
                    coverage = (n_required / len(df)) * 100
                    
                    st.info(f"""
                    **Comparación con tu dataset:**
                    - Tu dataset tiene: {len(df):,} registros
                    - Se requieren: {n_required:,} registros
                    - Cobertura: {coverage:.1f}%
                    """)
                    
                    if n_required > len(df):
                        deficit = n_required - len(df)
                        st.warning(f"⚠️ Faltan {deficit:,} registros")
                    else:
                        st.success("✅ Tu dataset es suficiente")
                
                except Exception as e:
                    st.error(f"Error: {str(e)[:100]}")
    
    # ========================================================================
    # PESTAÑA 3: NORMALIDAD CORREGIDA
    # ========================================================================
    with tabs[2]:
        st.subheader("🔍 Pruebas de Normalidad - CORREGIDAS")
        
        if numeric_cols:
            selected_var = st.selectbox("Selecciona variable:", numeric_cols, key="norm_var")
            alpha_norm = st.slider("Nivel α:", 0.01, 0.10, 0.05, 0.01, key="alpha_norm")
            
            if st.button("Ejecutar Pruebas", key="run_normality"):
                data = df[selected_var].dropna()
                n = len(data)
                
                if n < 3:
                    st.error("Se necesitan ≥ 3 observaciones")
                else:
                    st.info(f"**Tamaño muestra:** {n} observaciones")
                    
                    # Ejecutar pruebas
                    results = run_normality_tests_robust(data, alpha_norm)
                    
                    # Mostrar resultados
                    results_list = []
                    for test_name, test_result in results.items():
                        if 'error' not in test_result:
                            row = {
                                'Prueba': test_result['test'],
                                'Estadístico': f"{test_result.get('statistic', 'N/A'):.4f}" 
                                if test_result.get('statistic') is not None else 'N/A',
                                'Valor': f"{test_result.get('p_value', test_result.get('critical_value', 'N/A')):.4f}",
                                'Normal': '✅ Sí' if test_result.get('is_normal') else '❌ No' 
                                if test_result.get('is_normal') is not None else 'N/A',
                                'Nota': test_result.get('note', '')
                            }
                            results_list.append(row)
                    
                    if results_list:
                        results_df = pd.DataFrame(results_list)
                        st.dataframe(results_df, use_container_width=True, hide_index=True)
                        
                        # Determinar consenso
                        normal_tests = [r for r in results.values() 
                                      if r.get('is_normal') is not None]
                        if normal_tests:
                            n_normal = sum(1 for r in normal_tests if r['is_normal'])
                            consensus = n_normal / len(normal_tests)
                            
                            st.metric("Consenso", f"{consensus*100:.1f}%")
                            
                            if consensus >= 0.67:
                                st.success("✅ Los datos parecen normales")
                            elif consensus >= 0.33:
                                st.warning("⚠️ Evidencia mixta sobre normalidad")
                            else:
                                st.error("❌ Los datos no parecen normales")
                    
                    # Visualización
                    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
                    
                    # Histograma
                    ax1.hist(data, bins=30, alpha=0.7, density=True, edgecolor='black')
                    x_range = np.linspace(data.min(), data.max(), 100)
                    ax1.plot(x_range, stats.norm.pdf(x_range, data.mean(), data.std()), 
                           'r-', linewidth=2, label='Normal teórica')
                    ax1.set_title('Histograma con curva normal')
                    ax1.legend()
                    
                    # Q-Q plot
                    stats.probplot(data, dist="norm", plot=ax2)
                    ax2.set_title('Q-Q Plot')
                    
                    # Boxplot
                    ax3.boxplot(data, vert=False)
                    ax3.set_title('Boxplot')
                    
                    # ECDF
                    ecdf = np.arange(1, n+1) / n
                    sorted_data = np.sort(data)
                    ax4.plot(sorted_data, ecdf, 'b-', label='Empírica')
                    ax4.plot(sorted_data, stats.norm.cdf(sorted_data, data.mean(), data.std()), 
                           'r--', label='Teórica')
                    ax4.set_title('Función de distribución acumulada')
                    ax4.legend()
                    
                    plt.tight_layout()
                    st.pyplot(fig)
        
        else:
            st.warning("No hay variables numéricas")
    
    # ========================================================================
    # PESTAÑA 4: CORRELACIÓN
    # ========================================================================
    with tabs[3]:
        st.subheader("📉 Análisis de Correlación")
        
        if len(numeric_cols) >= 2:
            var_x = st.selectbox("Variable X:", numeric_cols, key="corr_x")
            var_y = st.selectbox("Variable Y:", 
                                [v for v in numeric_cols if v != var_x], 
                                key="corr_y")
            
            method = st.radio(
                "Método:",
                ["Pearson", "Spearman", "Kendall"],
                horizontal=True,
                key="corr_method"
            )
            
            if st.button("Calcular Correlación", key="calc_corr"):
                clean_data = df[[var_x, var_y]].dropna()
                n = len(clean_data)
                
                if n < 2:
                    st.error("Se necesitan ≥ 2 observaciones")
                else:
                    # Calcular correlación
                    if method == "Pearson":
                        corr, p_value = stats.pearsonr(clean_data[var_x], clean_data[var_y])
                        method_name = "Pearson (lineal)"
                    elif method == "Spearman":
                        corr, p_value = stats.spearmanr(clean_data[var_x], clean_data[var_y])
                        method_name = "Spearman (monotónica)"
                    else:
                        corr, p_value = stats.kendalltau(clean_data[var_x], clean_data[var_y])
                        method_name = "Kendall (rangos)"
                    
                    # Resultados
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Método", method_name)
                    col2.metric("Coeficiente", f"{corr:.4f}")
                    col3.metric("p-valor", f"{p_value:.4f}")
                    col4.metric("n", n)
                    
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
                    **Interpretación:**
                    - **Fuerza:** {strength} (|r| = {abs_corr:.3f})
                    - **Dirección:** {'Positiva' if corr > 0 else 'Negativa'}
                    - **Significancia:** {'Significativa (p < 0.05)' if p_value < 0.05 else 'No significativa'}
                    """)
                    
                    # Gráfico
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.scatter(clean_data[var_x], clean_data[var_y], alpha=0.6, s=50)
                    
                    if method == "Pearson":
                        # Línea de tendencia
                        z = np.polyfit(clean_data[var_x], clean_data[var_y], 1)
                        p = np.poly1d(z)
                        x_range = np.linspace(clean_data[var_x].min(), clean_data[var_x].max(), 100)
                        ax.plot(x_range, p(x_range), "r--", linewidth=2)
                    
                    ax.set_xlabel(var_x)
                    ax.set_ylabel(var_y)
                    ax.set_title(f'{method_name}: {var_x} vs {var_y}')
                    ax.grid(True, alpha=0.3)
                    
                    st.pyplot(fig)
        
        else:
            st.warning("Se necesitan ≥ 2 variables numéricas")
    
    # ========================================================================
    # PESTAÑA 5: PRUEBAS T CORREGIDAS
    # ========================================================================
    with tabs[4]:
        st.subheader("✅ Pruebas T - CORREGIDAS")
        
        test_type = st.radio(
            "Tipo de prueba:",
            ["Una muestra", "Dos muestras independientes", "Muestras pareadas"],
            horizontal=True,
            key="ttest_type"
        )
        
        alpha_ttest = st.slider("α:", 0.01, 0.10, 0.05, 0.01, key="alpha_tt")
        alternative = st.selectbox(
            "Hipótesis alternativa:",
            ["two-sided", "less", "greater"],
            format_func=lambda x: {"two-sided": "Bilateral", "less": "Unilateral <", "greater": "Unilateral >"}[x],
            key="ttest_alt"
        )
        
        # PRUEBA T UNA MUESTRA
        if test_type == "Una muestra" and numeric_cols:
            ttest_var = st.selectbox("Variable:", numeric_cols, key="ttest1_var")
            pop_mean = st.number_input("Media poblacional de referencia:", value=0.0, key="ttest1_mean")
            
            if st.button("Ejecutar Prueba T Una Muestra", key="run_ttest1"):
                data = df[ttest_var].dropna()
                n = len(data)
                
                if n < 2:
                    st.error("Se necesitan ≥ 2 observaciones")
                else:
                    # Prueba T
                    t_stat, p_two_tailed = stats.ttest_1samp(data, pop_mean)
                    
                    # CORRECCIÓN: Ajustar p-valor
                    p_value = adjust_one_tailed_pvalue(t_stat, p_two_tailed, alternative)
                    
                    # Intervalo de confianza
                    ci_low, ci_high = stats.t.interval(
                        1 - alpha_ttest, n-1,
                        loc=data.mean(), scale=stats.sem(data)
                    )
                    
                    # Tamaño del efecto
                    effect = calculate_effect_size_with_ci(
                        "Una muestra",
                        data1=data,
                        pop_mean=pop_mean
                    )
                    
                    # Resultados
                    st.markdown("### 📋 Resultados")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("t", f"{t_stat:.4f}")
                    col2.metric("p-valor", f"{p_value:.4f}")
                    col3.metric("gl", n-1)
                    col4.metric("Significativo", "✅ Sí" if p_value < alpha_ttest else "❌ No")
                    
                    # Estadísticas
                    st.markdown("### 📊 Estadísticas")
                    col_s1, col_s2 = st.columns(2)
                    with col_s1:
                        st.metric("Media muestral", f"{data.mean():.4f}")
                        st.metric("Desviación", f"{data.std():.4f}")
                    with col_s2:
                        st.metric("Media referencia", f"{pop_mean:.4f}")
                        st.metric(f"IC {(1-alpha_ttest)*100:.0f}%", f"[{ci_low:.4f}, {ci_high:.4f}]")
                    
                    # Tamaño del efecto
                    if 'd' in effect and effect['d'] != 0:
                        st.markdown("### 📈 Tamaño del Efecto")
                        col_e1, col_e2 = st.columns(2)
                        with col_e1:
                            st.metric("Cohen's d", f"{effect['d']:.4f}")
                            st.write(f"**Interpretación:** {effect['interpretation']}")
                        with col_e2:
                            st.metric("IC 95% para d", f"[{effect['ci_lower']:.4f}, {effect['ci_upper']:.4f}]")
                    
                    # Interpretación
                    direction = "≠" if alternative == "two-sided" else "<" if alternative == "less" else ">"
                    st.info(f"""
                    **📝 Interpretación:**
                    - H₀: μ = {pop_mean}
                    - H₁: μ {direction} {pop_mean}
                    - Decisión: {'Rechazar H₀' if p_value < alpha_ttest else 'No rechazar H₀'}
                    """)
        
        # PRUEBA T DOS MUESTRAS INDEPENDIENTES
        elif test_type == "Dos muestras independientes" and numeric_cols and categorical_cols:
            ttest2_var = st.selectbox("Variable numérica:", numeric_cols, key="ttest2_var")
            ttest2_group = st.selectbox("Variable categórica (2 grupos):", categorical_cols, key="ttest2_group")
            
            # Verificar que tenga exactamente 2 grupos
            unique_groups = df[ttest2_group].dropna().unique()
            if len(unique_groups) == 2:
                group1, group2 = unique_groups
                
                if st.button("Ejecutar Prueba T Independiente", key="run_ttest2"):
                    data1 = df[df[ttest2_group] == group1][ttest2_var].dropna()
                    data2 = df[df[ttest2_group] == group2][ttest2_var].dropna()
                    
                    if len(data1) < 2 or len(data2) < 2:
                        st.error("Cada grupo necesita ≥ 2 observaciones")
                    else:
                        # Prueba de homogeneidad
                        _, levene_p = stats.levene(data1, data2)
                        equal_var = levene_p > 0.05
                        
                        # Prueba T
                        t_stat, p_two_tailed = stats.ttest_ind(data1, data2, equal_var=equal_var)
                        
                        # CORRECCIÓN: Ajustar p-valor
                        p_value = adjust_one_tailed_pvalue(t_stat, p_two_tailed, alternative)
                        
                        # Tamaño del efecto
                        effect = calculate_effect_size_with_ci(
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
                        col_g1, col_g2 = st.columns(2)
                        with col_g1:
                            st.metric(f"{group1} (n={len(data1)})", f"{data1.mean():.4f}")
                            st.metric(f"Desviación {group1}", f"{data1.std():.4f}")
                        with col_g2:
                            st.metric(f"{group2} (n={len(data2)})", f"{data2.mean():.4f}")
                            st.metric(f"Desviación {group2}", f"{data2.std():.4f}")
                        
                        # Homogeneidad
                        st.metric("Levene p-valor", f"{levene_p:.4f}")
                        st.metric("Varianzas", "Homogéneas" if equal_var else "Diferentes")
                        
                        # Tamaño del efecto
                        if 'd' in effect and effect['d'] != 0:
                            st.markdown("### 📈 Tamaño del Efecto")
                            col_e1, col_e2 = st.columns(2)
                            with col_e1:
                                st.metric("Cohen's d", f"{effect['d']:.4f}")
                                st.write(f"**Interpretación:** {effect['interpretation']}")
                            with col_e2:
                                st.metric("IC 95%", f"[{effect['ci_lower']:.4f}, {effect['ci_upper']:.4f}]")
                        
                        # Visualización
                        fig, ax = plt.subplots(figsize=(10, 6))
                        plot_data = pd.DataFrame({
                            'Grupo': [group1]*len(data1) + [group2]*len(data2),
                            'Valor': list(data1) + list(data2)
                        })
                        sns.boxplot(data=plot_data, x='Grupo', y='Valor', ax=ax)
                        ax.set_title(f'Comparación: {group1} vs {group2}')
                        ax.set_ylabel(ttest2_var)
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
            else:
                st.warning(f"La variable '{ttest2_group}' tiene {len(unique_groups)} grupos. Debe tener exactamente 2.")
    
    # ========================================================================
    # PESTAÑA 6: ANOVA
    # ========================================================================
    with tabs[5]:
        st.subheader("📊 ANOVA con Verificación de Supuestos")
        
        if numeric_cols and categorical_cols:
            anova_var = st.selectbox("Variable dependiente:", numeric_cols, key="anova_var")
            anova_group = st.selectbox("Variable independiente:", categorical_cols, key="anova_group")
            
            if st.button("Ejecutar ANOVA", key="run_anova"):
                # Verificar supuestos primero
                assumptions = check_anova_assumptions_simple(df, anova_group, anova_var)
                
                if assumptions['warnings']:
                    st.warning("### ⚠️ Advertencias")
                    for warning in assumptions['warnings']:
                        st.write(f"- {warning}")
                
                if assumptions['recommendations']:
                    st.info("### 💡 Recomendaciones")
                    for rec in assumptions['recommendations']:
                        st.write(f"- {rec}")
                
                # Separar datos por grupo
                groups = df.groupby(anova_group)[anova_var]
                group_data = [group.dropna().values for _, group in groups]
                group_names = list(groups.groups.keys())
                
                if len(group_data) < 2:
                    st.error("Se necesitan ≥ 2 grupos")
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
                            'Desviación': f"{np.std(data):.4f}",
                            'Mediana': f"{np.median(data):.4f}"
                        })
                    
                    stats_df = pd.DataFrame(stats_data)
                    st.dataframe(stats_df, use_container_width=True)
                    
                    # Interpretación
                    if p_value < 0.05:
                        st.success(f"✅ Existen diferencias significativas entre los grupos")
                        
                        # Si hay más de 2 grupos, sugerir post-hoc
                        if len(group_data) > 2:
                            st.info("""
                            **💡 Para identificar qué grupos difieren:**
                            - Considerar pruebas post-hoc (Tukey, Bonferroni)
                            - Realizar comparaciones por pares
                            """)
                    else:
                        st.info(f"ℹ️ No hay diferencias significativas entre los grupos")
                    
                    # Visualización
                    fig, ax = plt.subplots(figsize=(10, 6))
                    plot_data = []
                    for name, data in zip(group_names, group_data):
                        for value in data:
                            plot_data.append({'Grupo': name, 'Valor': value})
                    
                    plot_df = pd.DataFrame(plot_data)
                    sns.boxplot(data=plot_df, x='Grupo', y='Valor', ax=ax)
                    ax.set_title(f'ANOVA: {anova_var} por {anova_group}')
                    ax.set_xlabel(anova_group)
                    ax.set_ylabel(anova_var)
                    ax.grid(True, alpha=0.3)
                    
                    # Agregar estadística F
                    ax.text(0.02, 0.98, f"F = {f_stat:.3f}\np = {p_value:.4f}",
                           transform=ax.transAxes, fontsize=11, verticalalignment='top',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
                    
                    st.pyplot(fig)
        
        else:
            st.warning("Se necesitan variables numéricas y categóricas")
    
    # ========================================================================
    # PESTAÑA 7: PRUEBAS NO PARAMÉTRICAS
    # ========================================================================
    with tabs[6]:
        st.subheader("🔄 Pruebas No Paramétricas")
        
        nonpar_test = st.selectbox(
            "Selecciona prueba:",
            ["Mann-Whitney U", "Wilcoxon (pareado)", "Kruskal-Wallis", "Chi-cuadrado"],
            key="nonpar_select"
        )
        
        alpha_nonpar = st.slider("α:", 0.01, 0.10, 0.05, 0.01, key="alpha_nonpar")
        
        # MANN-WHITNEY U
        if nonpar_test == "Mann-Whitney U" and numeric_cols and categorical_cols:
            mw_var = st.selectbox("Variable:", numeric_cols, key="mw_var")
            mw_group = st.selectbox("Grupo (2 categorías):", categorical_cols, key="mw_group")
            
            unique_groups = df[mw_group].dropna().unique()
            if len(unique_groups) == 2:
                group1, group2 = unique_groups
                
                if st.button("Ejecutar Mann-Whitney U", key="run_mw"):
                    data1 = df[df[mw_group] == group1][mw_var].dropna()
                    data2 = df[df[mw_group] == group2][mw_var].dropna()
                    
                    if len(data1) < 3 or len(data2) < 3:
                        st.error("Cada grupo necesita ≥ 3 observaciones")
                    else:
                        u_stat, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                        
                        # Resultados
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Estadístico U", f"{u_stat:.4f}")
                        col2.metric("p-valor", f"{p_value:.4f}")
                        col3.metric("Significativo", "✅ Sí" if p_value < alpha_nonpar else "❌ No")
                        
                        # Estadísticas
                        col_s1, col_s2 = st.columns(2)
                        with col_s1:
                            st.metric(f"Mediana {group1}", f"{np.median(data1):.4f}")
                            st.metric(f"n {group1}", len(data1))
                        with col_s2:
                            st.metric(f"Mediana {group2}", f"{np.median(data2):.4f}")
                            st.metric(f"n {group2}", len(data2))
                        
                        # Interpretación
                        st.info(f"""
                        **Interpretación:**
                        - H₀: Las distribuciones de {group1} y {group2} son iguales
                        - H₁: Las distribuciones son diferentes
                        - {'Diferencias significativas' if p_value < alpha_nonpar else 'No hay diferencias significativas'}
                        """)
            else:
                st.warning(f"La variable debe tener exactamente 2 grupos (tiene {len(unique_groups)})")
        
        # CHI-CUADRADO CORREGIDO
        elif nonpar_test == "Chi-cuadrado" and len(categorical_cols) >= 2:
            chi_var1 = st.selectbox("Variable 1:", categorical_cols, key="chi1")
            chi_var2 = st.selectbox("Variable 2:", 
                                   [c for c in categorical_cols if c != chi_var1], 
                                   key="chi2")
            
            if st.button("Ejecutar Chi-cuadrado", key="run_chi"):
                try:
                    # Crear tabla de contingencia
                    contingency = pd.crosstab(df[chi_var1], df[chi_var2])
                    
                    st.markdown("### 📊 Tabla de Contingencia")
                    st.dataframe(contingency, use_container_width=True)
                    
                    # Verificar supuestos
                    assumptions = chi_square_assumptions_check(contingency.values)
                    
                    # Mostrar advertencias
                    if assumptions['warnings']:
                        st.warning("### ⚠️ Consideraciones sobre supuestos")
                        for warning in assumptions['warnings']:
                            st.write(f"- {warning}")
                    
                    if assumptions['recommendations']:
                        st.info("### 💡 Recomendaciones")
                        for rec in assumptions['recommendations']:
                            st.write(f"- {rec}")
                    
                    # Resultados de la prueba
                    st.markdown("### 📋 Resultados Chi-cuadrado")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("χ²", f"{assumptions['chi2']:.4f}")
                    col2.metric("p-valor", f"{assumptions['p_value']:.4f}")
                    col3.metric("gl", assumptions['df'])
                    
                    # Interpretación
                    st.info(f"""
                    **Interpretación:**
                    - H₀: {chi_var1} y {chi_var2} son independientes
                    - H₁: {chi_var1} y {chi_var2} están asociadas
                    - {'Asociación significativa' if assumptions['p_value'] < alpha_nonpar else 'No hay asociación significativa'}
                    """)
                    
                    # Calcular medida de efecto
                    n = contingency.sum().sum()
                    if contingency.shape == (2, 2):
                        phi = np.sqrt(assumptions['chi2'] / n)
                        st.metric("Phi (efecto)", f"{phi:.4f}")
                    else:
                        min_dim = min(contingency.shape)
                        cramers_v = np.sqrt(assumptions['chi2'] / (n * (min_dim - 1)))
                        st.metric("Cramer's V", f"{cramers_v:.4f}")
                
                except Exception as e:
                    st.error(f"Error: {str(e)[:100]}")
    
    # ========================================================================
    # PESTAÑA 8: REPORTE
    # ========================================================================
    with tabs[7]:
        st.subheader("📋 Reporte de Análisis")
        
        st.markdown("### 📊 Generar Reporte Completo")
        
        if st.button("📈 Generar Reporte Estadístico", use_container_width=True):
            with st.spinner("Generando reporte..."):
                try:
                    # Crear reporte básico
                    report_content = f"""
                    {'='*60}
                    REPORTE DE ANÁLISIS ESTADÍSTICO
                    {'='*60}
                    
                    Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                    Archivo: {uploaded_file.name if uploaded_file else 'No especificado'}
                    
                    {'='*60}
                    1. RESUMEN DEL DATASET
                    {'='*60}
                    - Filas totales: {df.shape[0]:,}
                    - Columnas totales: {df.shape[1]}
                    - Variables numéricas: {len(numeric_cols)}
                    - Variables categóricas: {len(categorical_cols)}
                    - Valores faltantes: {df.isnull().sum().sum():,}
                    - Porcentaje completitud: {(1 - df.isnull().sum().sum()/(df.shape[0]*df.shape[1]))*100:.1f}%
                    
                    {'='*60}
                    2. VARIABLES NUMÉRICAS PRINCIPALES
                    {'='*60}
                    """
                    
                    if numeric_cols:
                        for i, var in enumerate(numeric_cols[:5]):  # Limitar a 5
                            data = df[var].dropna()
                            if len(data) > 0:
                                report_content += f"\n{var}:"
                                report_content += f"\n  - n = {len(data)}"
                                report_content += f"\n  - Media = {data.mean():.4f}"
                                report_content += f"\n  - Mediana = {data.median():.4f}"
                                report_content += f"\n  - Desviación = {data.std():.4f}"
                                report_content += f"\n  - Rango = [{data.min():.4f}, {data.max():.4f}]"
                    
                    report_content += f"""
                    
                    {'='*60}
                    3. RECOMENDACIONES DE ANÁLISIS
                    {'='*60}
                    """
                    
                    # Recomendaciones basadas en los datos
                    if numeric_cols:
                        report_content += "\n- Realizar análisis descriptivo de variables numéricas"
                    
                    if len(numeric_cols) >= 2:
                        report_content += "\n- Analizar correlaciones entre variables numéricas"
                    
                    if numeric_cols and categorical_cols:
                        report_content += "\n- Comparar grupos usando pruebas T o ANOVA"
                        report_content += "\n- Verificar normalidad antes de pruebas paramétricas"
                    
                    if len(categorical_cols) >= 2:
                        report_content += "\n- Analizar asociación entre variables categóricas (Chi-cuadrado)"
                    
                    report_content += f"""
                    
                    {'='*60}
                    4. NOTAS IMPORTANTES
                    {'='*60}
                    - Verificar supuestos antes de cada prueba
                    - Considerar pruebas no paramétricas si no se cumple normalidad
                    - Los resultados deben interpretarse en contexto
                    - Consultar con experto para análisis complejos
                    
                    {'='*60}
                    FIN DEL REPORTE
                    {'='*60}
                    """
                    
                    # Mostrar reporte
                    st.text_area("Reporte Estadístico", report_content, height=400)
                    
                    # Opción para descargar
                    st.download_button(
                        label="📥 Descargar Reporte",
                        data=report_content,
                        file_name=f"reporte_estadistico_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                    
                    # Exportar datos procesados
                    st.markdown("---")
                    st.markdown("### 💾 Exportar Datos")
                    
                    col_exp1, col_exp2 = st.columns(2)
                    
                    with col_exp1:
                        # Exportar a CSV
                        csv_data = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📄 Exportar a CSV",
                            data=csv_data,
                            file_name=f"datos_procesados_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    with col_exp2:
                        # Exportar a Excel
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            df.to_excel(writer, sheet_name='Datos', index=False)
                            # Agregar resumen
                            summary_df = pd.DataFrame({
                                'Métrica': ['Filas', 'Columnas', 'Numéricas', 'Categóricas', 'Faltantes'],
                                'Valor': [df.shape[0], df.shape[1], len(numeric_cols), len(categorical_cols), df.isnull().sum().sum()]
                            })
                            summary_df.to_excel(writer, sheet_name='Resumen', index=False)
                        
                        st.download_button(
                            label="📊 Exportar a Excel",
                            data=output.getvalue(),
                            file_name=f"datos_procesados_{datetime.now().strftime('%Y%m%d')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
                
                except Exception as e:
                    st.error(f"Error generando reporte: {str(e)[:100]}")

# ============================================================================
# INSTRUCCIONES CUANDO NO HAY DATOS
# ============================================================================

if df is None:
    st.info("""
    ## 👋 ¡Bienvenido al Analizador Estadístico!
    
    **Para comenzar:**
    1. 📁 Sube tu archivo de datos en la barra lateral (CSV o Excel)
    2. 📊 Espera a que se carguen los datos
    3. 🔍 Explora las diferentes pestañas de análisis
    
    **📋 Formatos soportados:**
    - Archivos CSV (valores separados por comas)
    - Archivos Excel (.xlsx, .xls)
    
    **📊 Análisis disponibles:**
    - Estadísticas descriptivas
    - Pruebas de normalidad (corregidas)
    - Correlaciones (Pearson, Spearman, Kendall)
    - Pruebas T (una muestra, independientes)
    - ANOVA con verificación de supuestos
    - Pruebas no paramétricas
    - Reportes completos
    
    **✅ Características:**
    - Todas las funciones autocontenidas
    - Sin dependencias problemáticas
    - Cálculos estadísticos corregidos
    - Interfaz intuitiva
    - Exportación de resultados
    """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
<small>
📊 <strong>Analizador Estadístico Universal</strong> | 
✅ <strong>Versión Corregida y Robusta</strong> |
🔒 <strong>Todos los cálculos se realizan localmente</strong><br>
📚 <strong>Desarrollado con Streamlit, Pandas, NumPy, SciPy</strong> |
🎯 <strong>Para análisis estadísticos precisos y confiables</strong>
</small>
</div>
""", unsafe_allow_html=True)