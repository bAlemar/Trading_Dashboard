import dash
from dash import dcc 
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input, State
from dash import dash_table
import plotly.express as px
from Function_Trad_Dash import Binance_data
from Function_Trad_Dash import Pycaret_Model
import plotly.graph_objects as go
import re
import glob, os
import pandas as pd
import numpy as np

##==================//=======================================//=====================
# LISTA PARA INPUTS
##==================//=======================================//=====================
#Trading Windows
trading_window = ['1dia','12h','2h','1h','30min']

#Modelos de ML
with open('nome_modelos.txt') as f:
    modelos = f.read()
modelos =  modelos.split('\n')
list_model = [] #Sigla
list_model2 = [] #Nome
for i in modelos: #Podia Fazer com Regex
    if i != '':
        list_model.append(i.split('â€™')[0].strip('â€˜'))
        pos = i.find('-')
        list_model2.append(i[pos+2:])
dict_model = dict(zip(list_model2,list_model))

#Features:
with open(r'func_ta_lib_copy.txt') as f:
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
dict_feature_all = {'feature_name':[],'feature_eq':[]}
for i in teste:
    func_name = i[0]
    func_eq = i[1]
    if func_name not in dict_feature2['feature_name']:
        dict_feature['feature_name'].append(func_name)
        dict_feature['feature_eq'].append(func_eq)
    dict_feature_all['feature_name'].append(func_name)
    dict_feature_all['feature_eq'].append(func_eq)






app = dash.Dash(__name__,external_stylesheets=[dbc.themes.BOOTSTRAP],meta_tags=[{'content':'width-device,initial-scale=1.0'}])


##==================//=======================================//=====================
# Estrutura das Labels
##==================//=======================================//=====================
def Label1():
    #Select Coin market 
    grid22 = html.Div([
        #Input
        html.P('Insira o Ativo'),
        dcc.Input(id='coin-market',placeholder='Ex:BTC',style={'border-style':'outset'})
    
    ],style={'text-align':'center','color':'white'})

    ##DropDown Trade  Window
    grid23 = html.Div([
        #Trading Window Select
        html.P('Insira Janela de Trading'),
        dcc.Dropdown(id='trade-window',options=trading_window,style={'margin-bottom':'10px','color':'black','text-align':'left'})
    ],style={'text-align':'center','color':'white'})

    #Data de Inicio do Ativo
    grid24= html.Div([
        #Input
        html.P('Insira Data de Inicio'),
        dcc.Input(id='start_date',placeholder='Ex: 1999,1 jan',style={'border-style':'outset'})
    ],style={'text-align':'center','color':'white'})
    
     #Botao de Execução dos Inputs e Outputs
    grid31 = html.Div([
        #Button
        dbc.Button('Executar',id='Executar')
    ],style={'text-align':'left','color':'white','margin-left':'43px','margin-bottom':'10px'})
    
    #Grafico
    grid41 = html.Div(children=[html.Div(id='fig1')])

   

    #Nossa Estrutura para dado 1
    Botao_1_content = html.Div([
        dbc.Row([
            #Percebe-se que Max width col = 12, então SUM(WIDTH)= 12
            dbc.Col([],width=1), #Trick para dar Espaço Certinho Entre as box
            dbc.Col(grid22,width=3),
            dbc.Col(grid23,width=3),
            dbc.Col(grid24,width=3),
            dbc.Col([],width=1)
            ]),
        dbc.Row([
        dbc.Col([],width=1),
        dbc.Col(grid31)   
        ]),
        dbc.Row([
            dbc.Col([],width=1),
            dbc.Col(grid41)
        ])
    ])    
    return Botao_1_content
def Label2():
    #Coin
    grid11 = html.Div([
        html.P('Insira Cripto'),
        dcc.Input(id='coin',placeholder='Ex: BTC')
    ],style={'text-align':'center','color':'white'})
    
    #Janela de Trading
    grid12 = html.Div([
        html.P('Janela de Trading'),
        dcc.Dropdown(id='input_tw',options=trading_window,style={'textAlign':'left','color':'black'})
    ],style={'text-align':'center','color':'white'})
    
    #Insirando Features:
    grid13 = html.Div([
        html.P('Insira as Features',style={'text-align':'center'}),
        dcc.Dropdown(id='input_feature',
                    options=dict_feature['feature_name'],
                    multi=True,style={'textAlign':'left','color':'black'}),
        html.Button('Select All',id='select_all_dropdown1',n_clicks=0,style={'margin-top':'3.5px'})
    ],style={'text-align':'left','color':'white'})
    
    #Insira Modelo:
    grid14 = html.Div([
        html.P('Insira o Modelo'),
        dcc.Dropdown(id='input_model',options=list_model2,style={'textAlign':'left','color':'black'})
    ],style={'text-align':'center','color':'white'})
    
    #Botao:
    grid21 = html.Div([
        #Button
        dbc.Button('Executar',id='Executar2',style={'margin-bottom':'7px'}),
        dbc.Button('Salvar Modelo',id='save_model',style={'margin-left':'7px','margin-bottom':'7px'})
    ],style={'margin-left':'10px'})
    grid32 = html.Div(children=[html.Div(id='fig2')
    ])
    grid33 = html.Div(children=[html.Div(id='table2')],style={'margin-top':'50px'})

    
    Botao_2_content = html.Div([
        dbc.Row([
            dbc.Col([],width=1),
            dbc.Col(grid11),
            dbc.Col(grid12),
            dbc.Col(grid13),
            dbc.Col(grid14),
            dbc.Col([],width=1)
        ]),
        dbc.Row([
            dbc.Col([],width=1),
            dbc.Col(grid21)
        ]),
        dbc.Row([
            dbc.Col(grid32),
            dbc.Col(grid33)
        ],style={'background-color':'white'})
    ])
    return Botao_2_content
def Label3():
    #Load Modelos Salvos
    os.chdir('/Cods/DashBoard')
    modelos_salvos = []
    for file in glob.glob("*.pkl"):
        modelos_salvos.append(file[:-4])
    #Load Model
    grid11 = html.Div([
        #Load Model
        html.P(['Insira o Modelo'],style={'color':'white'}),
        dcc.Dropdown(id='Load_Model',options=modelos_salvos,style={'width':'500px','margin-bottom':'10px','color':'black'})
        ])
    #Relatorio Modelo
    grid21 = html.Div(children=[html.Div(id='fig3')
    ])
    #Classification Report
    grid22 = html.Div(children=[html.Div(id='table3_2')])
    #Tabela, saber prox pregao
    grid31 = html.Div(children=[html.Div(id='table3')
    ])
        
    Botao_3_content = html.Div([
        dbc.Row([
            dbc.Col(grid11)],style={'margin-left':'10px'}),
        dbc.Row([
            dbc.Col(grid21),
            dbc.Col(grid22,style={'margin-top':'50px'})],style={'background-color':'white'}),
        dbc.Row([
            dbc.Col(grid31)
        ]) 
            ])
    return Botao_3_content
#Titulo
grid11 = html.Div([
        #titulo
        html.H1('Criação Bot',style={'textAlign':'center','color':'wheat'},className='mb-4')
    ])

## LAYOUT
app.layout = (html.Div([
    dbc.Row([
        dbc.Col(grid11),
        dbc.Col(dbc.Button('Retorno do Ativo',id='Label_1',class_name='border border-dark mt-3',style={'background-color': '#636669','color':'white'})),
        dbc.Col(dbc.Button('Criação do Modelo',id='Label_2',class_name='border border-dark mt-3',style={'background-color': '#636669','color':'white'})),
        dbc.Col(dbc.Button('Modelos Salvos',id='Label_3',class_name='border border-dark mt-3',style={'background-color': '#636669','color':'white'}))
    ]),
    dbc.Row([
        html.Hr(style={'color':'white'})
    ]),
    dbc.Row([
        html.Div(children=[html.Div(id='display_label')])
    ])
],style={'background-color':'black'})
)



##==================//=======================================//=====================
#APPCALLBACK_LABELS -> Botoes das Paginas do dash 
##==================//=======================================//=====================
@app.callback(
    [Output('display_label','children'),
    ],
    [Input('Label_1','n_clicks'),
    Input('Label_2','n_clicks'),
    Input('Label_3','n_clicks')]
)
def Label_1(clicks1,clicks2,clicks3):
    ctx = dash.callback_context
    if clicks1 == None and clicks2 == None and clicks3 == None:
        return ''
    else:
        x = ctx.triggered[0]['prop_id'].split('.')[0]
        if x == 'Label_1':
            return [Label1()]
        elif x == 'Label_2':
            return [Label2()]
        elif x == 'Label_3':
            return [Label3()]
##==================//=======================================//=====================
#APPCALLBACK_Pagina_1 -> Retorna Gráfico de retorno do Ativo dada Data Inicio
##==================//=======================================//=====================
@app.callback(
    [Output('fig1','children')]
    ,
    [State('coin-market','value'),
    State('trade-window','value'),
    State('start_date','value'),
    Input('Executar','n_clicks')]
)
def Figura(input_coin,input_tw,input_dt,n_clicks):
    if n_clicks == 0:
        return ''
    else:
        coin_data = Binance_data()
        coin_data.load_data_binance(input_coin,input_tw,input_dt)
        coin_data.return_coin()
        df = coin_data.data
        fig = go.Figure()
        fig.add_traces(go.Scatter(x=df['Date'],y=df['Retorno_Acumulado']))
        fig.update_layout(title=f'Retorno do {input_coin}-USDT',
                        title_x=0.5,
                        xaxis_title='Data',
                        yaxis_title='Retorno %')
        grid_new = dcc.Graph(figure=fig)
        return [grid_new]
##==================//=======================================//=====================
#APPCALLBACK_Pagina_2 -> Retorna Gráfico quanto o modelo Rendeu
##==================//=======================================//=====================
@app.callback(
    [Output('fig2','children'),
    Output('table2','children')],
    [State('coin','value'),
    State('input_tw','value'),
    State('input_feature','value'),
    State('input_model','value'),
    Input('Executar2','n_clicks'),
    Input('save_model','n_clicks'),
    Input('select_all_dropdown1','n_clicks')]
)
def retorno_graphico(input_coin,input_tw,input_feature,input_model,n_clicks,n_clicks2,n_clicks3):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'select_all_dropdown1' in changed_id: #Select All features
        input_feature = dict_feature_all['feature_name']
    if n_clicks == 0:
        return ''
    elif n_clicks != 0 or n_clicks2 == None:    #Run Model
        modelo = dict_model[f'{input_model}']
        btc = Binance_data()
        btc.load_data_binance(input_coin,input_tw)
        #print('input_feature',input_feature)
        btc.insert_feature(input_feature)
        btc.insert_target()
        df = btc.data
        #df['Target'] = df['Close'].shift(-1)
        #df.drop(columns=['Close','Open'],inplace=True)
        df[['Month','Day','Year']] = df[['Month','Day','Year']].astype('int') 
        oi = Pycaret_Model()    
        oi.start(df,0.8)
        oi.run_model(modelo)
        oi.relatorio_model()
        fig2,df_result = oi.Retorno_Acumulado()
        df_result = df_result[['Date','Retorno_Real','Profit_Model_Acumulado','Label']]
        df_result = df_result.round(3)
        df_result = df_result.iloc[::-1]
        df_result = df_result.to_dict('records')
        table_new = dash_table.DataTable(data=df_result,
                                         style_table={'height': '300px', 'overflowY': 'auto'},
                                         page_size=10)
        grid_new = dcc.Graph(figure=fig2)
        aux = 1
    if n_clicks2 != None and aux == 1:      #Save model
        print('input_feature',input_feature)
        #Salvando Modelo
        oi.save_model(f'{input_coin}_{input_tw}_{input_model}')
        #Salvando os Input_feature
        with open(f'{input_coin}_{input_tw}_{input_model}.txt','w') as output:
            output.write(str(input_feature))
        aux = 0
    return [grid_new],[table_new]
##==================//=======================================//=====================
#APPCALLBACK_Pagina_3 -> LOAD_MODEL
##==================//=======================================//=====================
@app.callback(
    [Output('fig3','children'),
    Output('table3','children'),
    Output('table3_2','children')],
    [Input('Load_Model','value')]
)
def retorno_graphico3(load_model):
    oi = Pycaret_Model()
    oi.load_model(load_model)
    fig2,df_result,rel = oi.Prox_Pregao()
    df_result = df_result[['Date','Retorno_Real','Profit_Model_Acumulado','Label']]
    df_result['Date'] = pd.to_datetime(df_result['Date']).dt.date
    df_result = df_result.round(3)
    df_result = df_result.iloc[::-1]
    df_result = df_result.to_dict('records')
    table_new = dash_table.DataTable(data=df_result,
                                    style_table={'height': '300px', 'overflowY': 'auto'},
                                    page_size=10)
    grid_new = dcc.Graph(figure=fig2)
    rel = pd.DataFrame(rel).T
    rel = rel.round(3)
    rel = rel.iloc[:-2,:]
    rel.reset_index(inplace=True)
    rel = rel.to_dict('records')
    table2_new = dash_table.DataTable(data=rel)
    return [grid_new],[table_new],[table2_new]
if __name__ == '__main__':
    app.run_server(debug=True)