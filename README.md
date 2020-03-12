# MAC0499
## Trabalho Supervisionado de Formatura

**Nome do estudante:** Pedro Henrique Barbosa de Almeida

**Nome da orientadora:** Nina Sumiko Tomita Hirata

**Tema do trabalho:** Segmentação de imagens com redes totalmente convolucionais

**Descrição:**

Os processamentos de imagens, em geral, baseiam-se em combinações de vários tipos de transformações, tarefa realizada pelos operadores de imagens. Vários desses operadores de imagens são transformações locais, caracterizadas por uma função local. Por função local, referimo-nos a uma função cuja entrada é em geral uma pequena região da imagem centrada num pixel. Essa função é aplicada pixel a pixel para gerar a imagem transformada. Desta forma, torna-se possı́vel modelar o problema de projetar um operador como um problema de aprendizado dessas funções locais.

As abordagens mais recentes para transformação imagem-para-imagem utilizam modelos de redes totalmente convolucionais (Fully Convolutional Networks ou simplesmente FCN, em inglês), que são capazes de processar uma imagem inteira de uma só vez. 

O objetivo deste Trabalho Supervisionado de Formatura é estudar e aplicar as FCN em tarefas que usualmente são realizadas via classificação de pixels, como segmentação de vasos da retina, segmentação de textos em imagens de documentos ou a remoção de linhas em partituras de música. 

Em particular, estamos interessados em adaptar uma conhecida técnica de combinação de transformações locais, que requer múltiplos passos de treinamento, para o contexto de aprendizado profundo. Desta forma, espera-se que essa combinação de operadores possa ser treinada na forma ponta-a-ponta em apenas um passo, e ainda que a rede gerada seja do tipo FCN, o que permitirá o processamento de todos os pixels de uma só vez. Além disso, outro objetivo é avaliar a performance das redes resultantes, comparando-as com as contrapartes já existentes.
