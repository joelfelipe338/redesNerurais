from adaline import Adaline
from random import  randint

t1 = Adaline(4)
t2 = Adaline(4)
t3 = Adaline(4)
t4 = Adaline(4)
t5 = Adaline(4)

E = [[0.4329,-1.3719,0.7022,-0.8535],[0.3024,0.2286,0.8630,2.7909],[0.1349,-0.6445,1.0530,0.5687],[0.3374,-1.7163,0.3670,-0.6283],[1.1434,-0.0485,0.6637,1.2606],
     [1.3749,-0.5071,0.4464,1.3009],[0.7221,-0.7585,0.7681,-0.5592],[0.4403,-0.8072,0.5154,-0.3129],[-0.5231,0.3548,0.2538,1.5776],[0.3255,-2.0000,0.7112,-1.1209],
     [0.5824,1.3915,-0.2291,4.1735],[0.1340,0.6081,0.4450,3.2230],[0.1480,-0.2988,0.4778,0.8649],[0.73599,0.1869,-0.0872,2.3584],[0.7115,-1.1469,0.3394,0.9573],
     [0.8251,-1.2840,0.8452,1.2382],[0.1569,0.3712,0.8825,1.7633],[0.0033,0.6835,0.5389,2.8249],[0.4243,0.8313,0.2643,3.5855],[1.0490,0.1326,0.9138,1.9792],
     [1.4276,0.5331,-0.0145,3.7286],[0.5971,1.4865,0.2904,4.6069],[0.8475,2.1479,0.3179,5.8235],[1.3967,-0.4171,0.6443,1.3927],[0.0044,1.5378,0.6099,4.7755],
     [0.2201,-0.5668,0.0515,0.7829],[0.6300,-1.2480,0.8591,0.8093],[-0.2479,0.8960,0.0547,1.7381],[-0.3088,-0.0929,0.8659,1.5483],[-0.5180,1.4974,0.5453,2.3993],
     [0.6833,0.8266,0.0829,2.8864],[0.4353,-1.4066,0.4207,-0.4879],[-0.1069,-3.2329,0.1856,-2.4572],[0.4662,0.6261,0.7304,3.4370],[0.8298,-1.4089,0.3119,1.3235]]
S = [1,-1,-1,-1,1,1,1,1,-1,1,
     -1,-1,1,1,-1,-1,1,-1,-1,1,
     1,-1,-1,1,-1,1,-1,1,-1,1,
     1,1,-1,-1,-1]

treino = True
x = 0
teste = 1
tam = len(E)
while(treino):
    print("Geração: ",x)
    x += 1
    eqm_ant = t5.EQM(tam,E,S)
    for i in range(tam):
        t5.treino(E[i],S[i])
        teste = 2
    eqm_atual = t5.EQM(tam,E,S)
    if(t5.testeErro(eqm_ant,eqm_atual)):
        print("Treinamento Acabou!!")
        treino = False




print(t5.predict([0.9694,0.6909,0.4334,3.4965]))
print(t5.predict([0.5427,1.3832,0.6390,4.0352]))
print(t5.predict([0.6081,-0.9196,0.5925,0.1016]))
print(t5.predict([-0.1618,0.4694,0.2030,3.0117]))
print(t5.predict([0.1870,-0.2578,0.6124,1.7749]))
print(t5.predict([0.4891,-0.5279,0.4378,0.6439]))
print(t5.predict([0.3777,2.0149,0.7423,3.3932]))
print(t5.predict([1.1498,-0.4067,0.2469,1.5866]))
print(t5.predict([0.9325,1.0950,1.0359,3.3591]))
print(t5.predict([0.5060,1.3317,0.9222,3.7174]))
print(t5.predict([0.0497,-2.0656,0.6124,-0.6585]))
print(t5.predict([0.4004,3.5369,0.9766,5.3532]))
print(t5.predict([-0.1874,1.3343,0.5374,3.2189]))
print(t5.predict([0.5060,1.3317,0.9222,3.7174]))
teste = 1
print(t5.predict([1.6375,-0.7911,0.7537,0.5515]))
