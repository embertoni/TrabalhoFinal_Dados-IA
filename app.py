import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.set_page_config(
    page_title="Análise de Dados - Trainee IA",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

CORES = ['#1e6b7a', '#e8a020', '#1a3a5c', '#4a9aba', '#c0392b']
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

@st.cache_data
def carregar_e_otimizar_dados():
    df = pd.read_csv("dados/ImpactoAI.csv")
    memoria_original = df.memory_usage(deep=True).sum() / (1024 ** 2)
    
    if "Student_ID" in df.columns:
        df = df.drop(columns=["Student_ID"])
        
    colunas_float = ["Pre_Semester_GPA", "Post_Semester_GPA", "Skill_Retention_Score", "Weekly_GenAI_Hours", "Traditional_Study_Hours"]
    for col in colunas_float:
        if col in df.columns:
            df[col] = df[col].astype("float32")
            
    colunas_int = ["Anxiety_Level_During_Exams", "Perceived_AI_Dependency", "Tool_Diversity"]
    for col in colunas_int:
        if col in df.columns:
            df[col] = df[col].astype("int32")
            
    if "Paid_Subscription" in df.columns:
        df["Paid_Subscription"] = df["Paid_Subscription"].astype(bool)
        
    memoria_otimizada = df.memory_usage(deep=True).sum() / (1024 ** 2)
    return df, memoria_original, memoria_otimizada

df, mem_orig, mem_otim = carregar_e_otimizar_dados()

st.title("Análise de Dados e Correlações")
st.markdown("Visualização interativa baseada nos conceitos práticos aprendidos durante o Treinamento")

st.sidebar.header("Menu de Navegação")
pagina = st.sidebar.radio("Selecione uma página:", ["Visão Geral", "Correlações", "Visualizações", "Download"])

if pagina == "Visão Geral":
    st.header("Visão Geral da Base de Dados")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total de Registros (Linhas)", len(df))
    with col2:
        st.metric("Número de Variáveis (Colunas)", len(df.columns))
    with col3:
        st.metric("Dados Ausentes detectados", df.isnull().sum().sum())
    
    with st.expander("Relatório de Otimização de Memória"):
        c1, c2, c3 = st.columns(3)
        c1.metric("Uso de Memória Original", f"{mem_orig:.2f} MB")
        c2.metric("Uso de Memória Otimizado", f"{mem_otim:.2f} MB")
        economia = ((mem_orig - mem_otim) / mem_orig) * 100
        c3.metric("Economia de RAM", f"{economia:.1f}%", delta=f"-{economia:.1f}%")
        st.info("A economia foi gerada através da remoção do ID único e conversão de tipos primitivos (Ex: float64 -> float32).")

    st.subheader("Visualização Amostral (Primeiras 10 linhas)")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.subheader("Estatística Descritiva Resumida")
    st.dataframe(df.describe().T, use_container_width=True)
    
    st.subheader("Distribuição de Variáveis Categóricas")
    col_cat = st.selectbox("Escolha uma categoria para analisar a frequência:", df.select_dtypes(include=['object', 'bool']).columns)
    frequencias = df[col_cat].value_counts()
    c_grafico, c_tabela = st.columns([2, 1])
    with c_grafico:
        st.bar_chart(frequencias, color=CORES[0])
    with c_tabela:
        st.dataframe(frequencias, use_container_width=True)

elif pagina == "Correlações":
    st.header("Matriz de Correlações Numéricas")
    numeric_df = df.select_dtypes(include=[np.number])
    correlacao = numeric_df.corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlacao, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
    ax.set_title("Mapa de Calor de Correlações", fontsize=12, fontweight='bold', pad=15)
    st.pyplot(fig)
    
    st.subheader("Correlações Mais Expressivas")
    limiar = st.slider("Selecione o limiar de sensibilidade da correlação (módulo):", 0.1, 0.9, 0.4, step=0.05)
    
    correlacoes_fortes = []
    for i in range(len(correlacao.columns)):
        for j in range(i+1, len(correlacao.columns)):
            val = correlacao.iloc[i, j]
            if abs(val) >= limiar:
                correlacoes_fortes.append({
                    'Variável 1': correlacao.columns[i],
                    'Variável 2': correlacao.columns[j],
                    'Coeficiente': round(val, 3),
                    'Tipo': 'Positiva' if val > 0 else 'Negativa'
                })
    
    if correlacoes_fortes:
        st.dataframe(pd.DataFrame(correlacoes_fortes), use_container_width=True)
    else:
        st.info(f"Nenhuma correlação com força maior ou igual a {limiar} foi detectada.")

elif pagina == "Visualizações":
    st.header("Visualizações Personalizadas e Gráficos Científicos")
    
    tipo_grafico = st.selectbox("Selecione o Tipo de Gráfico:", ["Scatter Plot Avançado", "Histograma Otimizado", "Box Plot Comparativo"])
    
    numeric_df = df.select_dtypes(include=[np.number])
    categoric_df = df.select_dtypes(include=['object', 'bool'])
    
    col1, col2 = st.columns(2)
    
    if tipo_grafico == "Scatter Plot Avançado":
        with col1:
            var_x = st.selectbox("Métrica para o Eixo X (Numérica):", numeric_df.columns, index=1)
        with col2:
            var_y = st.selectbox("Métrica para o Eixo Y (Numérica):", numeric_df.columns, index=5)
            
    elif tipo_grafico == "Histograma Otimizado":
        with col1:
            var_x = st.selectbox("Métrica para o Histograma (Numérica):", numeric_df.columns, index=1)
            
    elif tipo_grafico == "Box Plot Comparativo":
        with col1:
            var_x = st.selectbox("Agrupar por (Categoria no Eixo X):", categoric_df.columns, index=1)
        with col2:
            var_y = st.selectbox("Métrica Analisada (Numérica no Eixo Y):", numeric_df.columns, index=5)
            
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if tipo_grafico == "Scatter Plot Avançado":
        scatter = ax.scatter(
            df[var_x], 
            df[var_y], 
            c=df["Anxiety_Level_During_Exams"] if "Anxiety_Level_During_Exams" in df.columns else None,
            s=df["Perceived_AI_Dependency"] * 25 if "Perceived_AI_Dependency" in df.columns else 40,
            cmap="viridis", 
            alpha=0.75,
            edgecolors='none'
        )
        ax.set_xlabel(var_x, fontsize=10, fontweight='bold')
        ax.set_ylabel(var_y, fontsize=10, fontweight='bold')
        ax.set_title(f"Dispersão: {var_x} vs {var_y}", fontsize=12, fontweight='bold', pad=10)
        
        if "Anxiety_Level_During_Exams" in df.columns:
            cbar = fig.colorbar(scatter, ax=ax)
            cbar.set_label("Nível de Ansiedade durante os Exames", fontsize=9)
            
        st.pyplot(fig)
        
    elif tipo_grafico == "Histograma Otimizado":
        sns.histplot(df[var_x], bins=25, color=CORES[0], kde=True, ax=ax, alpha=0.8)
        ax.set_xlabel(var_x, fontsize=10, fontweight='bold')
        ax.set_ylabel("Contagem de Registros", fontsize=10, fontweight='bold')
        ax.set_title(f"Distribuição de Frequência: {var_x}", fontsize=12, fontweight='bold', pad=10)
        st.pyplot(fig)
        
    elif tipo_grafico == "Box Plot Comparativo":
        sns.boxplot(data=df, x=var_x, y=var_y, palette=CORES, ax=ax, width=0.5)
        ax.set_xlabel(var_x, fontsize=10, fontweight='bold')
        ax.set_ylabel(var_y, fontsize=10, fontweight='bold')
        ax.set_title(f"Análise de Quartis: {var_y} por {var_x}", fontsize=12, fontweight='bold', pad=10)
        plt.xticks(rotation=15)
        st.pyplot(fig)

elif pagina == "Download":
    st.header("Exportação de Resultados")
    numeric_df = df.select_dtypes(include=[np.number])
    correlacao = numeric_df.corr()
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Matriz de Correlação")
        csv_corr = correlacao.to_csv()
        st.download_button(
            label="Baixar Matriz de Correlação (CSV)",
            data=csv_corr,
            file_name="matriz_correlacoes.csv",
            mime="text/csv"
        )
    with col2:
        st.subheader("Dados Tratados e Otimizados")
        csv_dados = df.to_csv(index=False)
        st.download_button(
            label="Baixar Base Limpa e Otimizada (CSV)",
            data=csv_dados,
            file_name="ImpactoAI_otimizado.csv",
            mime="text/csv"
        )