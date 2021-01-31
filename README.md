# MAC0499
# Trabalho Supervisionado de Formatura

# Informações gerais 

## Estudante

Pedro Henrique Barbosa de Almeida

## Orientadora 
 Nina Sumiko Tomita Hirata

## Título do trabalho

Um estudo sobre segmentação de imagens com redes totalmente convolucionais - O problema da segmentação de texto em mangás

## Descrição 

Os processamentos de imagens, em geral, baseiam-se em combinações de vários tipos de transformações, tarefa realizada pelos operadores de imagens. Vários desses operadores de imagens são transformações locais, caracterizadas por uma função local. Por função local, referimo-nos a uma função cuja entrada é em geral uma pequena região da imagem centrada num pixel. Essa função é aplicada pixel a pixel para gerar a imagem transformada. Desta forma, torna-se possı́vel modelar o problema de projetar um operador como um problema de aprendizado dessas funções locais.

Neste trabalho, estudamos o impacto da variação de alguns parâmetros de uma CNN, chamada U-Net, no contexto de poucos exemplos de formação. Para analisar os efeitos das mudanças, optou-se por aplicar a rede ao problema da segmentação do texto na manga, ao estilo da história em quadrinhos japonesa. Mais especificamente, queremos treinar um classificador que preveja quais pixels pertencem à classe "texto" e quais não.

# Estrutura do repositório:

* ```monografia.pdf```: monografia final apresentada no âmbito da disciplina de Trabalho Supervisionado de Formatura; 
* ```apresentação.pdf```: apresentação final apresentada no âmbito da disciplina de Trabalho Supervisionado de Formatura; 
* ```/experimentos```: códigos utilizados para realizar os experimentos conduzidos no estudo (vide monografia). Em geral, os scripts ``driver.py`` treinam e avaliam um modelo, gerando, na maior parte das vezes, um arquivo ```.csv``` com as métricas, bem como imagens ```.png``` com a evolução das curvas. Estão subdivididos em: 
    * ```/1-baseline```: contém o modelo base usado.
    * ```/2-depths```: experimenta com diversas profundidades da rede. 
    * ```/3-normalizations```: experimenta diferentes técnicas de normalização. Contém ainda: 
        * ```/on input```: experimenta com a normalização de entrada. 
        * ```/on batch```: experimenta com a normalização de batch.
        * ```/on input+batch```: experimenta com a combinação de normalização de entrada com a de batch. 
    * ```/4-losses```: experimenta com diveras funções de perda. Contém ainda: 
        * ```/on d1```: experimenta com o conjunto de treino formado pelas 10 páginas com mais componentes conexos. 
        * ```/on d2```: experimenta com o conjunto de treino formado pelas 10 primeiras páginas de uma obra. 
        * ```/on d3```: experimenta com o conjunto de treino formado pela 5 páginas com mais componentes conexos. 
    * ```/5-titles```: experimenta com diferentes títulos de mangá. 
    * ```/6-cross-eval```: experimenta treinar o modelo em um título e avaliar em outro. 
    * ```/7-no-training-examples```: experimenta treinar com diferentes números de exemplos de treino. 
* ```/subconjuntos```: contém diversas variações de conjuntos utilizados para treinar e validar as performances. Está subdividido em: 
    * ```D1_ds0```: conjunto com as 10 páginas com mais componentes conexos do título "EvaLady"
    * ```D1_ds1```: conjunto com as 10 páginas com mais componentes conexos do título "AosugiruHaru"
    * ```D1_ds2```: conjunto com as 10 páginas com mais componentes conexos do título "JijiBabaFight"
    * ```D1_ds3```: conjunto com as 10 páginas com mais componentes conexos do título "MariaSamaNihaNaisyo"
    * ```D2_ds0```: conjunto com as 10 primeiras imagens do título "EvaLady"
    * ```TT_ds0```: conjunto de imagens remanescentes do título "EvaLady"
    * ```TT_ds1```: conjunto de imagens remanescentes do título "AosugiruHaru"
    * ```TT_ds2```: conjunto de imagens remanescentes do título "JijiBabaFight"
    * ```TT_ds3```: conjunto de imagens remanescentes do título "MariaSamaNihaNaisyo"
    
 Observação: os conjuntos de validação V_1 e V_2 são construídos a partir de um dos conjuntos TT_ds#. 
