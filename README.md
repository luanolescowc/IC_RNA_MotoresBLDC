Esse repositório tem o objetivo de descrever todo o trabalho desenvolvido durante a minha bolsa de iniciação científica vinculada a universidade estadual de Santa Catarina durante o período de agosto de 2019 até agosto de 2020. 
Motores brushless DC tem a característica de utilizar ímãs permanentes em sua construção para realizar a indução magnética, resultando em uma atratividade magnética rotacional do eixo do motor. por conta disso utiliza sensor hall para descrever a posição angular do motor a fim de fazer a inserção de corrente na espira correta resultando em uma rotação síncrona e continua. Porém com a utilização de sensor hall fica visível o aumento de volume e de manutenção, se tornando um processo caro. Com base nisso o objetivo dessa bolsa de iniciação científica foi realizar o estudo sobre motores brushless DC e apresentar uma solução ao problema do uso de sensor hall. O objetivo principal foi utilizar redes neurais artificiais para quê, com base em variáveis coletadas do motor, prever a posição angular para que se faça desnecessário o uso de sensor hall. 
Objetivo final do projeto era estabelecer um controle preditivo e adaptativo em tempo real para o motor. 

Resultados alcançados: 

- Estudos aprofundados sobre motores brushless DC. 
- Estudos sobre as constantes físicas envolvidas do motor. 
- Estudo e coleta de dados das variáveis envolvidas. 
- Tratamento de dados. 
- Modelo de regressão. 
- Rede neural artificial Afim de descrever a posição angular do motor. 
- 96,5% de precisão para velocidades constantes de rotação.

Problemas a serem resolvidos: 

- Organização e automatização do código. 
- Erro de predição considerável em variações bruscas de velocidade. 
- Erro de predição para velocidades abaixo de 600 rpm. 
- Erro de predição para velocidades acima de 2200 rpm. 
- Velocidade de processamento de dados para a predição em tempo real. 
- Aquisição de dados em tempo real. 
- Elaboração do controle adaptativo em tempo real. 

Variaveis de Input:
-Velocidade retórica do motor
-Corrente da fase.  
- BackEMF Trapezoidal 

Variavel de saida:
Posição angular do motor

Foi-se utilizada a linguagem de programação Python e as seguintes bibliotecas: pandas, numpy, matplotlib, sk-learng e tensorflow.

