import streamlit as st # type: ignore
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
import os
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport  # Movido este import para o topo
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Carregando a imagem do logotipo
st.set_page_config(page_title='Previsão de Renda',
                   page_icon='https://img2.gratispng.com/20180218/pbw/kisspng-growth-chart-bar-chart-clip-art-business-growth-chart-png-transparent-images-5a89d6af726216.6282995115189828314685.jpg',
                   layout='wide')

# Barra lateral para entrada do usuário
st.sidebar.header("Dados do Usuário")

# Adicionar campos de entrada para a entrada do usuário
idade = st.sidebar.number_input("Idade", min_value=18, max_value=100, value=30)
educacao = st.sidebar.selectbox("Nível de Educação", ["Ensino Médio", "Graduação", "Mestrado", "Doutorado"])

# Página Sobre
st.sidebar.markdown("### Sobre")
st.sidebar.write("Para uma análise completa e eficaz dos dados dos clientes salvos no banco de dados do banco, é importante realizar uma série de etapas de entendimento dos dados. Essas análises ajudarão a obter insights valiosos sobre os dados dos clientes, identificar padrões relevantes e preparar os dados de forma adequada para a construção do modelo preditivo de renda."
)

# Sidebar markdown section with download buttons
st.sidebar.markdown("### Arquivos Salvos")

# Function to download files
def download_file(file_path, file_name):
    with open(file_path, 'rb') as file:
        data = file.read()
    st.sidebar.download_button(label=file_name, data=data, file_name=file_name)

# Define file paths and names
file_paths = [
    ("C:/Users/lpell/Downloads/16/projeto 2/input/previsao_de_renda.csv", "previsao_de_renda.csv"),
    ("C:/Users/lpell/Downloads/16/projeto 2/output/renda_analysis.html", "renda_analysis.html"),
    ("C:/Users/lpell/Downloads/16/projeto 2/projeto-2.ipynb", "projeto-2.ipynb")
]

# Create download buttons for each file
for file_path, file_name in file_paths:
    download_file(file_path, file_name)

# Seção de Ajuda
st.sidebar.header("Ajuda")
st.sidebar.write("Bem-vindo à seção de ajuda! Aqui você encontrará informações sobre como usar o aplicativo e dicas úteis para navegar melhor.")


# Botão de Feedback
st.sidebar.header("Feedback")
st.sidebar.write("Nos ajude a melhorar! Envie seu feedback ou sugestões para o e-mail lupellozzo@gmail.com.")


def main():
    st.title('Previsão de Renda')
    
    st.title('Etapa 1 CRISP - DM: Entendimento do negócio')
    st.write("""
    Em uma instituição financeira, entender o perfil dos novos clientes é crucial para diversas finalidades, como avaliar a capacidade de pagamento, prever comportamentos financeiros e dimensionar limites de crédito de forma responsável. Essa análise também desempenha um papel importante na mitigação de riscos, como a inadimplência, e na personalização das ofertas de produtos e serviços para atender às necessidades dos clientes de maneira eficaz.
    """)
    
    st.title('Etapa 2 CRISP-DM: Entendimento dos dados')
    st.write("""
    Para uma análise completa e eficaz dos dados dos clientes salvos no banco de dados do banco, é importante realizar uma série de etapas de entendimento dos dados.

    Essas análises ajudarão a obter insights valiosos sobre os dados dos clientes, identificar padrões relevantes e preparar os dados de forma adequada para a construção do modelo preditivo de renda.
    """)
    
    st.subheader('Dicionário de dados')
    st.write("""
    Aqui temos informações sobre as variáveis nos seus dados de uma instituição financeira, como o nome da variável, a descrição e o tipo de dados.
    """)
    
    # Tabela com o dicionário de dados
    data_dict = {
        'Variável': ['data_ref', 'id_cliente', 'sexo', 'posse_de_veiculo', 'posse_de_imovel', 'qtd_filhos', 'tipo_renda', 'educacao', 'estado_civil', 'tipo_residencia', 'idade', 'tempo_emprego', 'qt_pessoas_residencia', 'renda'],
        'Descrição': ['Data de referência', 'Identificador do cliente', 'Gênero', 'Possui veículo', 'Possui imóvel', 'Quantidade de filhos', 'Tipo de renda', 'Nível de educação', 'Estado civil', 'Tipo de residência', 'Idade', 'Tempo de emprego (em anos)', 'Quantidade de pessoas na residência', 'Renda'],
        'Tipo': ['object', 'int', 'object', 'Bool(binário)', 'Bool(binário)', 'int', 'object', 'object', 'object', 'object', 'float', 'float', 'float', 'float']
    }
    st.table(data_dict)

    st.subheader('Carregando os pacotes')
    st.write("Codigo usado:")
    imports_code = """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from ydata_profiling import ProfileReport
    import os
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeRegressor
    from sklearn import tree
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    %matplotlib inline
    """
    st.code(imports_code, language='python')
    
    st.subheader('Carregando os dados')
    st.write("""
    O comando pd.read_csv é um comando da biblioteca pandas (pd.) e carrega os dados do arquivo csv indicado para um objeto dataframe do pandas.

    Aqui geramos um DataFrame chamado 'renda', e as primeiras linhas dos dados são exibidas usando o método head().
    """)
    st.write("Codigo usado:")
    imports_code = """
    renda = pd.read_csv('./input/previsao_de_renda.csv')
    renda.head()
    """
    st.code(imports_code, language='python')

    # Exemplo de visualização dos dados
    data_preview = pd.DataFrame({
        'Unnamed: 0': [0, 1, 2, 3, 4],
        'data_ref': ['2015-01-01', '2015-01-01', '2015-01-01', '2015-01-01', '2015-01-01'],
        'id_cliente': [15056, 9968, 4312, 10639, 7064],
        'sexo': ['F', 'M', 'F', 'F', 'M'],
        'posse_de_veiculo': [False, True, True, False, True],
        'posse_de_imovel': [True, True, True, True, False],
        'qtd_filhos': [0, 0, 0, 1, 0],
        'tipo_renda': ['Empresário', 'Assalariado', 'Empresário', 'Servidor público', 'Assalariado'],
        'educacao': ['Secundário', 'Superior completo', 'Superior completo', 'Superior completo', 'Secundário'],
        'estado_civil': ['Solteiro', 'Casado', 'Casado', 'Casado', 'Solteiro'],
        'tipo_residencia': ['Casa', 'Casa', 'Casa', 'Casa', 'Governamental'],
        'idade': [26, 28, 35, 30, 33],
        'tempo_emprego': [6.602740, 7.183562, 0.838356, 4.846575, 4.293151],
        'qt_pessoas_residencia': [1.0, 2.0, 2.0, 3.0, 1.0],
        'renda': [8060.34, 1852.15, 2253.89, 6600.77, 6475.97]
    })
    st.write(data_preview)

if __name__ == '__main__':
    main()

# Carregar os dados
renda = pd.read_csv('./input/previsao_de_renda.csv')

st.subheader('Entendimento dos Dados - Univariada')
st.write("Nesta etapa tipicamente avaliamos a distribuição de todas as variáveis.")

# Análise Univariada
# Função para gerar e salvar o relatório em HTML
def gerar_relatorio():
    prof = ProfileReport(renda, minimal=True)
    caminho_saida = './output/renda_analysis.html'
    prof.to_file(caminho_saida)
    return caminho_saida

# Gerando e salvando o relatório
caminho_relatorio = gerar_relatorio()

# Exibindo o link de download para o relatório em HTML
st.markdown(f"Relatório gerado e salvo em: {caminho_relatorio}")
st.write("Para visualizar o relatório completo, faça o download do arquivo abaixo:")
with open(caminho_relatorio, 'rb') as arquivo:
    dados = arquivo.read()
st.download_button(label='Download Relatório', data=dados, file_name='renda_analysis.html')

def univariate_analysis():
    # Exibir estatísticas descritivas
    st.subheader('Estatísticas Descritivas')
    desc_stats = renda.describe()
    desc_stats_transposed = desc_stats.transpose()
    st.write(desc_stats_transposed)

    # Exibir a matriz de correlação
    st.subheader('Matriz de Correlação')
    num_vars = renda.select_dtypes(include=['int64', 'float64'])
    correlation_matrix = num_vars.corr()
    fig, ax = plt.subplots(figsize=(12, 10))  # Cria uma figura e um eixo
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)  # Plota no eixo
    st.pyplot(fig)  # Exibe a figura no Streamlit
    # Selecionar apenas as colunas numéricas
    colunas_numericas = renda.select_dtypes(include=['float64', 'int64'])
    # Calcular a correlação entre as colunas numéricas
    correlacao = colunas_numericas.corr()
    # Exibir a última linha da matriz de correlação
    ultima_linha_correlacao = correlacao.tail(1)
    st.subheader('Ultima linha de correlação')
    ultima_linha_correlacao

    st.write(" Analisando a matriz de correlação, identificamos que a variável mais correlacionada com a renda é o tempo de emprego, apresentando um índice de correlação de 38,5%. Esse resultado é um insight valioso para compreender e prever o perfil dos clientes de forma mais precisa.")
    #matriz de dispersão
    st.subheader('Matriz de Dispersão')

    st.write(" - Divida os dados em grupos com base em uma variável categórica relevante e compare as distribuições de outras variáveis entre esses grupos.")
    #para o usuario ver o codigo usado
    st.write("Codigo usado:")
    imports_code = """
    sns.pairplot(data=renda,
    hue='tipo_renda',
    vars=['qtd_filhos', 
                   'idade', 
                   'tempo_emprego', 
                   'qt_pessoas_residencia', 
                   'renda'], 
             diag_kind='hist')
             plt.show()
    """
    st.code(imports_code, language='python')
    st.set_option('deprecation.showPyplotGlobalUse', False)

    sns.pairplot(data=renda, 
                 hue='tipo_renda', 
                 vars=['qtd_filhos', 'idade', 'tempo_emprego', 'qt_pessoas_residencia', 'renda'], 
                 diag_kind='hist',
                 height=3,  # Altura dos gráficos
                 aspect=1.5)  # Proporção largura-altura dos gráficos

    # Exibir o gráfico no Streamlit
    st.pyplot()  # Passando a figura gerada diretamente para st.pyplot()

    st.write('Ao examinar o pairplot, que é uma representação gráfica da matriz de dispersão, é possível detectar alguns outliers na variável de renda. Embora ocorram com baixa frequência, esses outliers podem afetar a análise de tendências, por exemplo. Além disso, observa-se uma correlação baixa entre as variáveis quantitativas, o que reforça os resultados encontrados na matriz de correlação.')
    # Exibir clustermap
    st.write("""
                 - Clustermap de Correlação""")

    sns.clustermap(data=correlation_matrix, cmap='coolwarm', figsize=(10, 10))
    st.pyplot()
    
    # Exibir análise de variáveis qualitativas
    st.write('- Análise de relevância preditiva com variáveis booleanas')
    
    # Pointplot Posse de Imóvel vs. Renda
    plt.rc('figure', figsize=(12,4))
    fig, axes = plt.subplots(nrows=1, ncols=2)

    sns.pointplot(x='posse_de_imovel', 
                  y='renda',  
                  data=renda, 
                  dodge=True, 
                  ax=axes[0])

    sns.pointplot(x='posse_de_veiculo', 
                  y='renda', 
                  data=renda, 
                  dodge=True, 
                  ax=axes[1])

    st.pyplot()

    st.write("""
    **Posse de Imóvel vs. Renda:**
    - No gráfico de posse de imóvel versus renda, observamos uma sobreposição considerável nos intervalos de confiança para clientes que possuem ou não imóvel. Isso sugere que a posse de imóvel pode ter uma influência limitada na predição da renda, pois as médias das rendas para ambos os grupos estão próximas.

    **Posse de Veículo vs. Renda:**
    - Já no gráfico de posse de veículo versus renda, notamos uma distância mais significativa entre os intervalos de confiança para clientes que possuem ou não veículo. Essa diferença sugere que a posse de veículo pode ser um fator mais relevante na predição da renda, pois as médias das rendas para os grupos de posse e não posse de veículo estão mais distantes.

    Essa comparação reforça a ideia de que a posse de veículo pode ter uma influência maior na renda do cliente em comparação com a posse de imóvel, tornando-a uma variável mais relevante na análise preditiva da renda.
    """)
    
    # Exibir análise das variáveis qualitativas ao longo do tempo
    st.write("- **Análise das variáveis qualitativas ao longo do tempo**")

    # Converter a coluna de data para datetime
    renda['data_ref'] = pd.to_datetime(arg=renda['data_ref'])
    qualitativas = renda.select_dtypes(include=['object', 'boolean']).columns
    plt.rc('figure', figsize=(16,4))
    for col in qualitativas:
        fig, axes = plt.subplots(nrows=1, ncols=2)
        fig.subplots_adjust(wspace=.6)
        
        tick_labels = renda['data_ref'].map(lambda x: x.strftime('%b/%Y')).unique()
        
        # barras empilhadas:
        renda_crosstab = pd.crosstab(index=renda['data_ref'], 
                                     columns=renda[col], 
                                     normalize='index')
        ax0 = renda_crosstab.plot.bar(stacked=True, 
                                      ax=axes[0])
        ax0.set_xticklabels(labels=tick_labels, rotation=45)
        axes[0].legend(bbox_to_anchor=(1, .5), loc=6, title=f"'{col}'")
        
        # perfis médios no tempo: 
        ax1 = sns.pointplot(x='data_ref', y='renda', hue=col, data=renda, dodge=True, errorbar=('ci', 95), ax=axes[1])
        ax1.set_xticklabels(labels=tick_labels, rotation=45)
        axes[1].legend(bbox_to_anchor=(1, .5), loc=6, title=f"'{col}'")
        
        st.pyplot(fig) 
    st.write("""
             ### Entendimento dos dados - Bivariadas
             Para realizar uma análise bivariada dos dados fornecidos, podemos utilizar gráficos e estatísticas para entender as relações entre diferentes variáveis.

             **Aluguel vs. Renda:** Podemos observar que a proporção de residências alugadas permanece relativamente estável ao longo do tempo, com pequenas variações mensais entre 0.006 e 0.021. Isso sugere uma certa estabilidade na preferência por residências alugadas entre os clientes ao longo dos meses analisados.

             **Casa vs. Renda:** A categoria de residências em casa mostra uma variação mais ampla, oscilando entre 0.87 e 0.92 ao longo do período. Isso indica uma relativa consistência na preferência por casas, mas com algumas flutuações mensais.

             **Com os pais vs. Renda:** A proporção de pessoas vivendo com os pais é menor em comparação com as outras categorias, variando principalmente entre 0.034 e 0.061. Parece haver uma tendência de queda nessa categoria ao longo do período analisado.

             **Comunitário vs. Renda:** A categoria de residências comunitárias mostra uma variação baixa, variando entre 0.001 e 0.008. Isso sugere uma escolha menos frequente desse tipo de residência entre os clientes.

             **Estúdio vs. Renda:** A proporção de estúdios parece ser estável ao longo do tempo, com variações entre 0.003 e 0.009. Isso indica uma relativa consistência na preferência por estúdios entre os clientes.

             **Governamental vs. Renda:** A categoria de residências governamentais mostra uma variação baixa, variando principalmente entre 0.015 e 0.041. Isso sugere que essa categoria não é tão frequente entre os clientes.

             Em resumo, podemos observar que a maioria dos clientes prefere residências em casa ou alugadas, com uma tendência de queda na proporção de pessoas vivendo com os pais ao longo do tempo. As outras categorias de residências têm proporções relativamente menores e flutuações mais estáveis ao longo do período analisado.
             """)
    st.subheader('Análise Bivariada')

    # Relação entre Tipo de Residência e Renda
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='tipo_residencia', y='renda', data=renda)
    plt.title('Relação entre Tipo de Residência e Renda')
    plt.xlabel('Tipo de Residência')
    plt.ylabel('Renda')
    plt.xticks(rotation=45)
    plt.show()
    st.pyplot()

    # Relação entre Idade e Renda
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='idade', y='renda', data=renda)
    plt.title('Relação entre Idade e Renda')
    plt.xlabel('Idade')
    plt.ylabel('Renda')
    plt.show()
    st.pyplot()

    # Relação entre Tempo de Emprego e Renda
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='tempo_emprego', y='renda', data=renda)
    plt.title('Relação entre Tempo de Emprego e Renda')
    plt.xlabel('Tempo de Emprego')
    plt.ylabel('Renda')
    plt.show()
    st.pyplot()

    # Relação entre Quantidade de Filhos e Renda
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='qtd_filhos', y='renda', data=renda)
    plt.title('Relação entre Quantidade de Filhos e Renda')
    plt.xlabel('Quantidade de Filhos')
    plt.ylabel('Renda')
    plt.show()
    st.pyplot()

    # Texto de análise bivariada
    st.write("""
    A análise bivariada realizada nos dados apresentou alguns insights relevantes para entender os fatores que podem influenciar a renda dentro do contexto do seu conjunto de dados:

    - **Tipo de Residência e Renda:** Observou-se uma relação entre o tipo de residência e a renda, onde clientes com diferentes tipos de residência apresentaram variações na renda. Isso sugere que o tipo de moradia pode impactar a renda dos clientes.

    - **Idade e Renda:** Não foi identificada uma relação clara entre a idade e a renda. A dispersão dos pontos no gráfico de dispersão não mostrou um padrão definido de aumento ou diminuição da renda com a idade, indicando que outros fatores podem ter maior influência na determinação da renda dos clientes.

    - **Tempo de Emprego e Renda:** Foi observada uma tendência de aumento da renda com o aumento do tempo de emprego. Isso sugere que clientes com mais tempo de experiência no emprego tendem a ter uma renda maior, o que pode ser resultado de progressão na carreira e aumento de salário ao longo do tempo.

    - **Quantidade de Filhos e Renda:** Verificou-se que a quantidade de filhos não apresentou uma relação clara com a renda. Os boxplots mostraram que a renda não varia significativamente com o número de filhos, indicando que outros fatores podem ser mais determinantes para a renda dos clientes.

    A importância de considerar múltiplos fatores ao analisar e prever a renda dos clientes, incluindo não apenas características demográficas como idade e número de filhos, mas também aspectos relacionados ao emprego e à moradia.
    """)
    st.title('Etapa 3 Crisp-DM: Preparação dos Dados')

    # Seleção de Dados
    st.subheader('Seleção de Dados')
    st.write("""
    - Verificar se os dados selecionados são adequados para a análise proposta. Caso contrário, pode ser necessário revisar e incluir/excluir variáveis conforme necessário.
    """)
    st.write("Codigo usado:")
    imports_code = """
    renda.drop(columns='data_ref', inplace=True)
    """
    st.code(imports_code, language='python')
    renda.drop(columns='data_ref', inplace=True)

    # Limpeza de Dados
    st.subheader('Limpeza de Dados')
    st.write("""
    - Identificar e tratar dados faltantes (valores ausentes) de maneira adequada, utilizando técnicas como imputação de valores médios, medianas, ou remoção de registros com valores faltantes dependendo do contexto.
    """)
    st.write("Codigo usado:")
    imports_code = """
    renda.dropna(inplace=True)
    """
    st.code(imports_code, language='python')
    renda.dropna(inplace=True)

    # Construção de Novas Variáveis
    st.subheader('Construção de Novas Variáveis')
    st.write("""
    - Avaliar se é necessário criar novas variáveis com base nas existentes para melhorar a análise e a modelagem posterior. Por exemplo, pode-se criar variáveis categóricas a partir de variáveis numéricas, criar variáveis de interação, etc.
    """)
    st.write("Codigo usado:")
    imports_code = """
    new_variables_df = pd.DataFrame(index=renda.columns, 
                                    data={'tipos_dados': renda.dtypes, 
                                          'qtd_valores': renda.notna().sum(), 
                                          'qtd_categorias': renda.nunique()})
    st.write(new_variables_df)
    """
    st.code(imports_code, language='python')
    new_variables_df = pd.DataFrame(index=renda.columns, 
                                    data={'tipos_dados': renda.dtypes, 
                                          'qtd_valores': renda.notna().sum(), 
                                          'qtd_categorias': renda.nunique()})
    st.write(new_variables_df)

    # Integração de Dados
    st.subheader('Integração de Dados')
    st.write("""
    - Se estiver trabalhando com múltiplas fontes de dados, é necessário integrar esses dados de forma adequada para que possam ser utilizados de maneira conjunta na análise.
    """)

    # Formatação de Dados
    st.subheader('Formatação de Dados')
    st.write("""
    - Garantir que os dados estejam em formatos úteis para a análise, como datas no formato correto, strings padronizadas, etc. Além disso, converter dados categóricos em formatos numéricos, se necessário, para facilitar a modelagem.
    """)
    st.write("Codigo usado:")
    imports_code = """
    renda_dummies= pd.get_dummies(data=renda)
    st.write(renda_dummies.head())
    """
    st.code(imports_code, language='python')
    renda_dummies= pd.get_dummies(data=renda)
    st.write(renda_dummies.head())

    # Exibir gráficos adicionais
    st.write('Visualizações em gráficos')
    st.write("""
    - Histograma da Posse de Veículo
    """)
    plt.figure(figsize=(10, 6))
    sns.histplot(data=renda, x='posse_de_veiculo', bins=30, kde=True)
    plt.title('Distribuição da Posse de Veículo')
    plt.xlabel('Posse de Veículo')
    st.pyplot()

    st.write("""
    - Gráfico de Dispersão: Quantidade de Pessoas Residência vs Renda
    """)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=renda, x='qt_pessoas_residencia', y='renda')
    plt.title('Quantidade pessoas residencia vs Renda')
    plt.xlabel('Quantidade pessoas residencia')
    plt.ylabel('Renda')
    st.pyplot()

    st.write("""
    - Boxplot: Posse de Imóvel vs Renda
    """)
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=renda, x='posse_de_imovel', y='renda')
    plt.title('Posse de imovel')
    plt.xlabel('Posse de imovel')
    plt.ylabel('Renda')
    st.pyplot()

    # Correlação
    correlacao_posse_renda = renda['posse_de_veiculo'].corr(renda['renda'])
    st.write(f"Correlação Posse de Veículo vs Renda: {correlacao_posse_renda}")
    correlacao_pessoas_renda = renda['qt_pessoas_residencia'].corr(renda['renda'])
    st.write(f"Quantidade pessoas residencia vs Renda: {correlacao_posse_renda}")
    correlacao_imovel_renda = renda['posse_de_imovel'].corr(renda['renda'])
    st.write(f"Posse de imovel vs Renda: {correlacao_posse_renda}")

    # Etapa 4 Crisp-DM: Modelagem
    st.title('Etapa 4 Crisp-DM: Modelagem')

    # Técnica de modelagem
    st.subheader('Técnica de Modelagem')
    st.write("""
    A técnica de modelagem escolhida é o Decision Tree Regressor. Esta escolha se baseia na capacidade do algoritmo em lidar eficazmente com problemas de regressão, como a previsão de renda dos clientes. Além disso, as árvores de decisão são conhecidas por sua facilidade de interpretação, o que é crucial para entender como o modelo faz suas previsões e quais atributos são mais relevantes.
    """)

    # Desenho do teste
    st.subheader('Desenho do Teste')
    st.write("""
    Aqui está o código para dividir os dados em conjunto de treinamento e teste, e para treinar o modelo de Árvore de Decisão Regressora:
    """)
    X = renda_dummies.drop(columns='renda')
    y = renda_dummies['renda']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    score_data = []
    for max_depth in range(1, 21):
        for min_samples_leaf in range(1, 31):
            reg_tree = DecisionTreeRegressor(random_state=42, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
            reg_tree.fit(X_train, y_train)
            score_data.append({'max_depth': max_depth, 'min_samples_leaf': min_samples_leaf, 'score': reg_tree.score(X=X_test, y=y_test)})

    score = pd.DataFrame(score_data)
    score_sorted = score.sort_values(by='score', ascending=False)

    best_max_depth = int(score_sorted.iloc[0]['max_depth'])
    best_min_samples_leaf = int(score_sorted.iloc[0]['min_samples_leaf'])

    reg_tree = DecisionTreeRegressor(random_state=42, max_depth=best_max_depth, min_samples_leaf=best_min_samples_leaf)
    reg_tree.fit(X_train, y_train)

    st.write("Melhores hiperparâmetros:")
    st.write(f"max_depth: {best_max_depth}")
    st.write(f"min_samples_leaf: {best_min_samples_leaf}")

    # Avaliação do modelo
    st.subheader('Avaliação do Modelo')
    st.write("""
    Aqui está a representação visual da Árvore de Decisão Regressora treinada:
    """)
    plt.figure(figsize=(18, 9))
    tp = tree.plot_tree(decision_tree=reg_tree, feature_names=X.columns.tolist(), filled=True)
    st.pyplot()

    st.write("Árvore de Decisão Regressora treinada de forma escrita: ")
    text_tree_print = tree.export_text(decision_tree=reg_tree, feature_names=list(X.columns))
    st.write(text_tree_print)

    # Etapa 5 Crisp-DM: Avaliação dos resultados
    st.title('Etapa 5 Crisp-DM: Avaliação dos resultados')

    st.write("""
    Nesta etapa, avaliamos os resultados do modelo construído. Vamos calcular o coeficiente de determinação (R2) para os conjuntos de treinamento e teste, criar uma coluna 'renda_predict' com as previsões da renda, e calcular o Erro Médio Quadrático (RMSE) e o Erro Absoluto Médio (MAE) para os conjuntos de treinamento e teste.
    """)

    # Calcular o coeficiente de determinação (R2) para os conjuntos de treinamento e teste
    r2_train = reg_tree.score(X=X_train, y=y_train)
    r2_test = reg_tree.score(X=X_test, y=y_test)

    st.write(f"O coeficiente de determinação (R2) para o conjunto de treinamento é: {r2_train:.2f}")
    st.write(f"O coeficiente de determinação (R2) para o conjunto de teste é: {r2_test:.2f}")

    # Criar uma coluna 'renda_predict' com as previsões da renda
    renda_dummies['renda_predict'] = reg_tree.predict(X)

    # Mostrar as colunas 'renda' e 'renda_predict'
    st.write("Previsão da renda:")
    st.write(renda_dummies[['renda', 'renda_predict']])

    # Calcular RMSE e MAE para os conjuntos de treinamento e teste
    rmse_train = mean_squared_error(y_train, reg_tree.predict(X_train), squared=False)
    rmse_test = mean_squared_error(y_test, reg_tree.predict(X_test), squared=False)

    mae_train = mean_absolute_error(y_train, reg_tree.predict(X_train))
    mae_test = mean_absolute_error(y_test, reg_tree.predict(X_test))

    st.write("\nErro Médio Quadrático (RMSE) para os conjuntos de treinamento e teste:")
    st.write(f"RMSE treino: {rmse_train:.2f}")
    st.write(f"RMSE teste: {rmse_test:.2f}")

    st.write("\nErro Absoluto Médio (MAE) para os conjuntos de treinamento e teste:")
    st.write(f"MAE treino: {mae_train:.2f}")
    st.write(f"MAE teste: {mae_test:.2f}")

    st.write("""
    Com a técnica usada com as árvores, conseguimos prever a renda, segundo a tabela acima.
    """)
    # Etapa 6 Crisp-DM: Implantação
    st.title('Etapa 6 Crisp-DM: Implantação')

    st.write("""
    Nesta etapa, colocamos em uso o modelo desenvolvido. Normalmente, isso envolve a implementação do modelo em um ambiente de produção, onde ele pode tomar decisões com algum nível de automação.
    """)

    entrada = pd.DataFrame([{'sexo': 'M', 
                             'posse_de_veiculo': False, 
                             'posse_de_imovel': True, 
                             'qtd_filhos': 1, 
                             'tipo_renda': 'Assalariado', 
                             'educacao': 'Superior completo', 
                             'estado_civil': 'Solteiro', 
                             'tipo_residencia': 'Casa', 
                             'idade': 34, 
                             'tempo_emprego': None, 
                             'qt_pessoas_residencia': 1}])
    entrada = pd.concat([X, pd.get_dummies(entrada)]).fillna(value=0).tail(1)
    renda_estimada = reg_tree.predict(entrada).item()

    st.write(f"Renda estimada: R${renda_estimada:.2f}")

if __name__ == '__main__':
    univariate_analysis()