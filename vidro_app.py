import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import pickle 
# formata o valor na moeda brasileira
import os

def converte_clasficacao_XGBoost_para_classe(classificacao_vidro_XGBoost):
    
    if classificacao_vidro_XGBoost == 0:
        classificacao_vidro = 1
    elif classificacao_vidro_XGBoost == 1:
        classificacao_vidro = 2
    elif classificacao_vidro_XGBoost == 2:
        classificacao_vidro = 3
    elif classificacao_vidro_XGBoost == 3:
        classificacao_vidro = 5
    elif classificacao_vidro_XGBoost == 4:
        classificacao_vidro = 6
    elif classificacao_vidro_XGBoost == 5:
        classificacao_vidro = 7

    return classificacao_vidro 

# Esconder os menu padrao
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

st.title('Classificação de Vidros')
st.sidebar.title("Informe os dados")

# variaveis 
atributos = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']


with st.sidebar:

    with st.form(key='my_form'):

        # [ 'idade', 'sexo', 'IMC', 'num_filhos', 'fumante', 'regiao']

        RI = st.number_input('Refractive Index', min_value=1.51115, max_value=1.53393, step=0.005, value=1.52000)

        Na = st.number_input('Sodium', min_value=10.73, max_value=17.38, step=0.5, value=15.5)
        
        Mg = st.number_input('Magnesium', min_value=0.0, max_value=4.49, step=0.5, value=2.0)

        Al = st.number_input('Aluminum', min_value=0.29, max_value=3.5, step=0.40, value=1.5)

        Si = st.number_input('Silicon', min_value=69.81, max_value=75.41, step=0.40, value=71.21)

        K = st.number_input('Potassium', min_value=0.0, max_value=6.21, step=0.60, value=2.51)
        
        Ca = st.number_input('Calcium', min_value=5.43, max_value=16.19, step=1.0, value=10.0)

        Ba = st.number_input('Barium', min_value=0.0, max_value=3.15, step=0.40, value=1.20)

        Fe = st.number_input('Iron', min_value=0.0, max_value=0.51, step=0.05, value=0.0)

        predict_button = st.form_submit_button(label='Resultado')



# Pagina pricipal
arquivo_modeloVidro = "ModeloXGBoostVidro.pkl"

with open(arquivo_modeloVidro, 'rb') as f:
        modeloVidro = pickle.load(f)



def previsao_vidro(modeloVidro ,RI , Na , Mg , Al , Si , K , Ca , Ba , Fe):

    new_X = np.array([RI , Na , Mg , Al , Si , K , Ca , Ba , Fe])
    valor_vidro = modeloVidro.predict(new_X.reshape(1, -1) )[0]
    return valor_vidro


#Carregar imagens das classes de vidro:
imagem1_classe1 = './images/class1-float-window.webp'
imagem2_classe1 = './images/class1-float-window2.jpg'
imagem3_classe1 = './images/class1-float-window3.jpg'
imagem1_classe2 = './images/class2-non-float-window.jpg'
imagem2_classe2 = './images/class2-non-float-window2.jpg'
imagem3_classe2 = './images/class2-non-float-window3.jpg'
imagem1_classe3 = './images/class3-float-car-window.webp'
imagem2_classe3 = './images/class3-float-car-window2.jpg'
imagem1_classe5 = './images/class5-containers.jpg'
imagem2_classe5 = './images/class5-containers2.jpg'
imagem3_classe5 = './images/class5-containers3.jpg'
imagem4_classe5 = './images/class5-containers4.webp'
imagem1_classe6 = './images/class6-tableware.jpg'
imagem2_classe6 = './images/class6-tableware2.webp'
imagem3_classe6 = './images/class6-tableware3.jpg'
imagem4_classe6 = './images/class6-tableware4.jpg'
imagem1_classe7 = './images/headlamps.jpeg'
imagem2_classe7 ='./images/headlamps2.jpg'


# Realizar a previsão da classe do vidro
if predict_button:
    valor_vidro  = converte_clasficacao_XGBoost_para_classe(previsao_vidro(modeloVidro, RI , Na , Mg , Al , Si , K , Ca , Ba , Fe))

 
    if valor_vidro == 1:
        st.markdown('## Classe de Vidro ' +  str(valor_vidro) + ': Vidros Float' )
        st.write('O vidro float é um vidro plano transparente, com espessura uniforme. \
                Não apresenta distorção óptica, e possui alta transmissão de luz. Mais de 90% dos vidros \
                 fabricados no Brasil são produzidos pelo processo de flutuação ou "float glass" \
                 A produção em grande escala e as características versáteis tornaram o vidro float um material popular em uma variedade de aplicações, incluindo janelas, portas, painéis solares, espelhos e muito mais.')

        col1, col2 = st.columns(2)

        image1 = Image.open(imagem2_classe1)
        col1.image(image1, use_column_width=True)

        image2 = Image.open(imagem1_classe1)
        col2.image(image2, use_column_width=True)

        st.write(" ")

        image3 = Image.open(imagem3_classe1)
        st.image(image3, use_column_width=True)
    elif valor_vidro == 2:
        st.markdown('## Classe de Vidro ' +  str(valor_vidro) + ': Vidros Não-Float' )
        st.write('O vidro que não passou pelo processo de flutuação, pode ter uma superfície menos plana e \
                 uniforme em comparação com o vidro Float. Geralmente tem uma aparência ondulada ou distorcida quando visto de lado.\
                 A transparência do vidro pode ser afetada, tendo variações na qualidade óptica em comparação com o vidro Float.')
        col1, col2 = st.columns(2)

        image1 = Image.open(imagem2_classe2)
        col1.image(image1, use_column_width=True)
        
        image3 = Image.open(imagem1_classe2)
        col2.image(image3, use_column_width=True)
        
        st.write(" ")
        
        image2 = Image.open(imagem3_classe2)
        st.image(image2, use_column_width=False, width= 300)
    elif valor_vidro == 3:
        st.markdown('## Classe de Vidro ' +  str(valor_vidro) + ': Vidros Float Automotivos' )
        st.write('Os vidros usados em janelas de carros, conhecidos como vidros automotivos, possuem características distintas \
                 que os diferenciam de outros tipos de vidros. São frequentemente temperados para aumentar a resistência mecânica. \
                 O vidro temperado é tratado termicamente para criar uma compressão superficial, tornando-o mais resistente a impactos.\
                 Muitas vezes, as janelas dos carros têm uma película de segurança laminada entre camadas de vidro. Isso evita que os estilhaços se espalhem em caso de quebra e oferece alguma resistência.')

        image1 = Image.open(imagem2_classe3)
        st.image(image1, use_column_width=True)
        
        st.write(" ")

        image1 = Image.open(imagem1_classe3)
        st.image(image1, use_column_width=True)
    elif valor_vidro == 5:
        st.markdown('## Classe de Vidro ' +  str(valor_vidro) + ': Vidro de Embalagem ou Vidro para Recipientes' )
        st.write('Os recipientes de vidro são amplamente utilizados para embalagem de alimentos, bebidas, produtos farmacêuticos e cosméticos. É geralmente composto por sílica (SiO2), soda cáustica (NaOH) e cal (CaCO₃).  \
                 É altamente reciclável, o que o torna uma escolha sustentável, podendo ser reciclado repetidamente sem perder qualidade. \
                 Ele é projetado para ser resistente mecânica e termicamente. Isso permite que suporte condições adversas durante o transporte e armazenamento de produtos.')

        col1, col2 = st.columns(2)

        image1 = Image.open(imagem1_classe5)
        col1.image(image1, use_column_width=True)
        
        image2 = Image.open(imagem2_classe5)
        col2.image(image2, use_column_width=True)

        image3 = Image.open(imagem3_classe5)
        col1.image(image3, use_column_width=True)
        
        image4 = Image.open(imagem4_classe5)
        col2.image(image4, use_column_width=True)
    elif valor_vidro == 6:
        st.markdown('## Classe de Vidro ' +  str(valor_vidro) + ': Vidro para utensílios de mesa (Tableware)' )
        st.write('O termo "tableware" refere-se ao conjunto de utensílios de mesa, incluindo pratos, copos, taças, talheres e outros itens relacionados. \
                 Esse tipo de vidro é geralmente tratado para aumentar sua durabilidade e resistência a choques térmicos e mecânicos. Isso é importante, especialmente para itens como pratos e copos que estão sujeitos a impactos e mudanças de temperatura. \
                 A composição típica de um vidro desse tipo normalmente consiste em Sílica, Soda cáustica, Cal e Alumina.')

        col1, col2 = st.columns(2)

        image1 = Image.open(imagem1_classe6)
        col1.image(image1, use_column_width=True)
        
        image2 = Image.open(imagem4_classe6)
        col2.image(image2, use_column_width=False, width= 300)

        image3 = Image.open(imagem3_classe6)
        col1.image(image3, use_column_width=True)
        
        image4 = Image.open(imagem2_classe6)
        col2.image(image4, use_column_width=False, width=250)
    elif valor_vidro == 7:
        st.markdown('## Classe de Vidro ' +  str(valor_vidro) + ': Vidro para faróis (Headlamps)' )
        st.write('Os vidros desse tipo são projetados para serem altamente duráveis e resistentes ao calor, uma vez que os faróis geram uma quantidade significativa de calor durante o uso. Além disso, eles são projetados para serem claros e transparentes para permitir a passagem máxima de luz.\
                 Geralmente, o vidro é produzido a partir de uma mistura de substâncias inorgânicas, que é denominada mistura vitrificável. Essa mistura é formada por sílica ou dióxido de silício (SiO2), barrilha ou soda (Na2CO3) e calcário (CaCO3). No entanto, a composição exata pode variar dependendo do tipo específico de vidro e das propriedades desejadas.')

        col1, col2 = st.columns(2)

        image1 = Image.open(imagem1_classe7)
        col1.image(image1, use_column_width=True)
        
        image2 = Image.open(imagem2_classe7)
        col2.image(image2, use_column_width=True)


        

        

    