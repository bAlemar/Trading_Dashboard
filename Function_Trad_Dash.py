import pycaret.regression as pr
import pycaret.classification as pc
import pandas as pd
import numpy as np
import datetime as dt
import talib as ta
import plotly.express as px
from sklearn.metrics import classification_report
from binance.client import Client
from binance.enums import *
import plotly.express 
import plotly.graph_objects as go
import re

#Carregando Cliente da binance
binance_api = open('Api_Binance.txt')
binance_api = binance_api.read().split('\n')
Api_key = binance_api[0]
Secret_key = binance_api[1]
client = Client(Api_key,Secret_key)
with open('func_ta_lib_copy.txt') as f:
    func_ta_lib = f.read()
teste2= re.findall(r'([\w\,\s]+)(\s\=\s)([\w]+)\((.+)\)',func_ta_lib) #Para Variaveis de Momento que TA retorna MAIS DE UM VALOR
teste = re.findall(r'([\w]+)\((.+)\)',func_ta_lib)
dict_feature2 = {'feature_name':[],'feature_eq':[],'feature_var':[]}
for i in teste2:
    func_name = i[2]
    func_eq = i[3]
    func_var = i[0].split('\n')[1:]
    if len(i[0].split('\n')[1]) > 4:
        dict_feature2['feature_name'].append(func_name)
        dict_feature2['feature_eq'].append(func_eq)
        dict_feature2['feature_var'].append(func_var)
dict_feature = {'feature_name':[],'feature_eq':[]}
for i in teste:
    func_name = i[0]
    func_eq = i[1]
    if func_name not in dict_feature2['feature_name']:
        dict_feature['feature_name'].append(func_name)
        dict_feature['feature_eq'].append(func_eq)

class Binance_data():
    def __init__(self):
        pass
    def load_data_binance(self,coin,trading_window,data_inicio='1999, 1 jan'):
        if trading_window == '1dia':
            trading_window = client.KLINE_INTERVAL_1DAY
        elif trading_window == '12h':
            trading_window = client.KLINE_INTERVAL_12HOUR
        elif trading_window == '1h':
            trading_window = client.KLINE_INTERVAL_1HOUR
        elif trading_window == '2h':
            trading_window = client.KLINE_INTERVAL_2HOUR
        elif trading_window == '30min':
            trading_window == client.KLINE_INTERVAL_30MINUTE
        btc = pd.DataFrame(client.get_historical_klines(f'{coin}USDT',trading_window,data_inicio))
        #Tratando os Dados pegos da Binance
        btc.columns = ['Open Time','Open','High','Low','Close','Volume','Close time','Quote asset volume','Number of trades','Taker but base asset volume','Taker but quote asset volume','Ignore']
        #Transformando colunas em float
        btc[['Open','High','Low','Close','Volume']] = btc[['Open','High','Low','Close','Volume']].astype('float') 
        #Transformando TimeStamp em Datetime
        btc['Open Time'] = pd.to_datetime(btc['Open Time'],unit='ms')
        #Criando as colunas de data, uma vez que pycaret nao lida diretamente com as datas
        btc['Month'] = btc['Open Time'].dt.month
        btc['Day'] = btc['Open Time'].dt.day
        btc['Year'] = btc['Open Time'].dt.year
        btc[['Month','Year','Day']] = btc[['Month','Year','Day']].astype('string')
        btc['Date'] = pd.to_datetime(btc['Year']+'-'+btc['Month']+'-'+btc['Day'])
        btc.drop(columns=['Open Time'],inplace=True)
        self.data = btc
        self.data_inicio = data_inicio
        return
    def return_coin(self):
        df = self.data
        df['Retorno'] = df['Close'].astype('float').pct_change()
        df['Retorno_Acumulado'] = df['Retorno'].cumsum()
        self.data = df
        return df
    def insert_feature(self,features_name='None'):
        df = self.data
        close = df['Close'].copy()
        low = df['Low'].copy()
        high = df['High'].copy()
        volume = df['Volume'].copy()
        open = df['Open'].copy()
        if features_name=='None':
            pass
        else:
            for i in features_name:
                if i in dict_feature['feature_name']: #Se essa Feature pertencer a Primeira parte
                    pos = list(dict_feature['feature_name']).index(f'{i}')
                    eq = dict_feature['feature_eq'][pos]
                    dict_args = {}
                    for j in eq.split(', '):
                        if '=' not in j:
                            dict_args.update({f'{j}':eval(j)})
                    try:
                        df[f'{i}'] = getattr(ta,f'{i}')(**(dict_args))
                    except:
                        df[f'{i}'] = getattr(ta,f'{i}')(close)
                else:
                    dict_args = {}
                    pos = list(dict_feature2['feature_name']).index(f'{i}')
                    eq = dict_feature2['feature_eq'][pos]
                    var = dict_feature2['feature_var'][pos]
                    #Pegando Close,High etc:
                    for j in eq.split(', '):
                        if '=' not in j:
                            dict_args.update({f'{j}':eval(j)})
                    for x in var:
                        try:
                            values = getattr(ta,f'{i}')(**(dict_args)) #assim funciona mas como sei se é arrownup ou arrowdown
                        except:
                            values = getattr(ta,f'{i}')(close)  
                        for num,z in enumerate(x.split(', ')):
                            df[f'{z.upper()}_{i}'] = values[num]
        df.drop(columns=['High', 'Low', 'Volume', 'Close time',
        'Quote asset volume', 'Number of trades', 'Taker but base asset volume',
        'Taker but quote asset volume', 'Ignore'],inplace=True)
        df.dropna(inplace=True)
        df[['Month','Day','Year']] = df[['Month','Day','Year']].astype('int')
        self.data = df
        return
    def insert_target(self,target='Close'):
        #O que voce quer prever em t+1?
        df = self.data
        df['Target'] = df[f'{target}'].pct_change().shift(-1)
        df['Target'] = np.where(df['Target']>0,1,0)
        df.drop(columns=['Open'],inplace=True)
        self.data = df
    def Info_acc(self):
        info = client.get_account()
        Info_acc = pd.DataFrame(info['balances'])
        Info_acc['free'] = Info_acc['free'].astype('float')
        Info_acc = Info_acc[Info_acc['free'] > 0]
        money = Info_acc[Info_acc['asset'] == 'USDT']['free'].values
        print(Info_acc[['asset','free']])
class Pycaret_Model():
    def __init__(self,df=None,size_train=None):
       pass
    def start(self,df,size_train):
        tamanho_treino = int(len(df)*size_train)
        treino = df.iloc[:tamanho_treino,:]
        teste = df.iloc[tamanho_treino:,:]
        teste_close = df.iloc[tamanho_treino:,:]
        #Esse df é para usarmos na coluna Retorno Acumulado
        treino.drop(columns='Close',inplace=True)
        teste.drop(columns='Close',inplace=True)
        self.teste_close = teste_close
        self.teste = teste
        self.treino = treino
    def run_model(self,model_name,target='Target'):
        teste = self.teste
        treino = self.treino
        s = pc.setup(data=treino,test_data=teste,target=f'{target}',fold_strategy='timeseries',session_id=123,silent=True,html=False)
        model = pc.create_model(f'{model_name}')
        self.model = model
        self.target = target
        pass
    def relatorio_model(self,show=True):
        model = self.model
        teste = self.teste
        predict = pc.predict_model(model,data=teste.drop([f'{self.target}'],axis=1))
        predict[['Month','Year','Day']] = predict[['Month','Year','Day']].astype('string')
        predict['Date'] = pd.to_datetime(predict['Year']+'-'+predict['Month']+'-'+predict['Day'])
        predict['Label'] = predict['Label'].astype('float')
        if show == True:
            fig = px.line(predict,x = 'Date', y=['Label'])
            #fig.show()
        self.predict = predict
        return fig
    def Retorno_Acumulado(self,date='2023'):
        predict = self.predict
        teste = self.teste
        teste_close = self.teste_close
        predict = predict[predict['Date'] > date]
        #Vendo quanto Rendeu o Modelo
        # Criando Colunas de Retornos e alvo bin para fazermos o algoritmo de compra
        predict['Retorno_Real'] = teste_close['Close'].pct_change()
        predict['Retorno_Bin_Real'] = np.where(predict['Retorno_Real'] > 0 ,1,0)
        predict.dropna(inplace=True)
        predict['Profit_Model'] = np.where(predict['Label'] == 1,predict['Retorno_Real'],0)
        predict['Profit_Model_Acumulado'] = predict['Profit_Model'].cumsum()
        predict['Retorno_Acumulado_Real'] = predict['Retorno_Real'].cumsum()
        #Printando Relatório
        #print(classification_report(predict['Retorno_Bin_Modelo'],predict['Retorno_Bin_Real']))
        #Plotando Gráfico
        fig = go.Figure()
        fig.add_traces(go.Scatter(x=predict['Date'],y=predict['Profit_Model_Acumulado'],name='Modelo'))
        fig.add_traces(go.Scatter(x=predict['Date'],y=predict['Retorno_Acumulado_Real'],name='Mercado'))
        fig.update_layout(title=f'Retorno Modelo',
                          title_x=0.5,
                          xaxis_title='Data',
                          yaxis_title='Retorno %')
        self.predict = predict
        return fig,predict
    def save_model(self,name):
        model = self.model
        pc.save_model(model,name)
    def load_model(self,name):
        model = pc.load_model(name)
        with open(f'{name}.txt') as f:
            features = f.read()
        name = name.split('_')
        coin = name[0]
        tw = name[1]
        self.features = features
        self.model = model
        self.coin = coin
        self.tw = tw
    def Simulacao(self,Io=100):
        predict = self.predict
        df_inv = predict[['Profit_Model_Acumulado','Profit_Model','Date']]
        #Criando Coluna
        df_inv['Investimento'] = np.nan
        #Valor Inicial
        df_inv.iloc[0,3] = Io
        for i in range(1,len(df_inv)):    
            if df_inv.iloc[i,1] > 0:
                df_inv.iloc[i,3] = (1+df_inv.iloc[i,1]) * df_inv.iloc[i-1,3]
            else:
                df_inv.iloc[i,3] = df_inv.iloc[i-1,3]
        fig = px.line(data_frame=df_inv,x='Date',y='Investimento')
        #fig.show()
        return fig
    def Prox_Pregao(self):
        model = self.model
        features = self.features
        features = re.findall(r'\'(\w+)\'',features)
        #Carregando base de Dados
        real_teste = Binance_data()
        real_teste.load_data_binance(self.coin,self.tw,'2022, 1 jan')
        real_teste.insert_feature(features_name = features)
        real_teste.insert_target()
        df = real_teste.data
        #Dropando Coluna Close(Vazamento de dados) e definindo o período
        df_semana = df.iloc[-10:,:]        
        Close = df_semana['Close']        
        predict_hoje = pc.predict_model(model,data=df_semana.drop('Close',axis=1))
        #Tratando a Base
        teste = df_semana
        predict = predict_hoje
        #Calculando Retorno
        predict['Retorno_Real'] = Close.pct_change()
        predict['Retorno_Bin_Real'] = np.where(predict['Retorno_Real'] > 0,1,0).astype('int')
        predict.dropna(inplace=True)
        predict['Profit_Model'] = np.where(predict['Label'] == 1,predict['Retorno_Real'],0)
        predict['Profit_Model_Acumulado'] = predict['Profit_Model'].cumsum()
        predict['Retorno_Acumulado_Real'] = predict['Retorno_Real'].cumsum()
        #Inserindo Date
        predict[['Month','Year','Day']] = predict[['Month','Year','Day']].astype('string')
        predict['Date'] = pd.to_datetime(predict['Year']+'-'+predict['Month']+'-'+predict['Day'])
        fig = go.Figure()
        fig.add_traces(go.Scatter(x=predict['Date'],y=predict['Profit_Model_Acumulado'],name='Modelo'))
        fig.add_traces(go.Scatter(x=predict['Date'],y=predict['Retorno_Acumulado_Real'],name='Mercado'))
        fig.update_layout(title=f'Retorno Modelo',
                          title_x=0.5,
                          xaxis_title='Data',
                          yaxis_title='Retorno %')
        self.predict = predict
        report = classification_report(predict['Retorno_Bin_Real'],predict['Label'],output_dict=True)
        return fig,predict,report
