# import json
import math
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import dash
import dash_core_components as dcc
import dash_html_components as html
# import dash_bootstrap_components as dbc
import dash_table
from dash_table.Format import Format, Scheme
from dash.dependencies import Input, Output
import plotly.express as px

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# external_stylesheets = ['style.css']

app = dash.Dash(__name__)#, external_stylesheets=external_stylesheets)
server = app.server
app.title = "RacketSpace"
server.secret_key = os.environ.get('SECRET_KEY', 'my-secret-key')

# df = pd.read_csv('https://gist.githubusercontent.com/chriddyp/5d1ea79569ed194d432e56108a04d188/raw/a9f9e8076b837d541398e999dcbac2b2826a81f8/gdp-life-exp-2007.csv')	
# raw_df = pd.read_csv('data/racket_specs.csv')
df = pd.read_csv('data/racket_specs.csv')
df['Model'] = df['Model'].astype(str)

# initial data source filtering (NaNs & erroneous values)
# df = raw_df[(raw_df['Head Size (in)']>=60) & \
#             (raw_df['Head Size (in)']<=140) & \
#             (raw_df['Strung Weight (oz)']>=5) & \
#             (raw_df['Strung Weight (oz)']<=20) & \
#             (raw_df['Beam Width (avg. mm)']>=5) & \
#             (raw_df['Beam Width (avg. mm)']<=50) & \
#             (raw_df['Balance (pts)'] >= -20) & \
#             (raw_df['Balance (pts)'] <= 20) & \
#             (raw_df['String Density (X/in2)'] >= 1) & \
#             (raw_df['String Density (X/in2)'] <= 6) & \
#             (raw_df['Swingweight'] >= 100) & \
#             (raw_df['Swingweight'] <= 600) & \
#             (raw_df['Stiffness'] >= 10) & \
#             (raw_df['Stiffness'] <= 100)].copy()
#
# df.rename(columns={'url': 'tw_url', 'img_url': 'tw_img_url'},
#                    # 'Unnamed: 0': 'ID', 'mfr': 'Manufacturer'},
#           inplace=True)
#
#
# df['ID'] = df.index
# df['Distance'] = 0

head_size_jitter 	  = 0.5
strung_weight_jitter  = 0.05
beam_width_jitter 	  = 0.5
balance_jitter 		  = 0.5
string_density_jitter = 0.05
swingweight_jitter    = 0.5
stiffness_jitter      = 0.5

def add_jitter(input, max_jitter):
    return input + np.random.uniform(-max_jitter, max_jitter)

df['Head Size (in) jittered'] = df['Head Size (in)'].apply(add_jitter, max_jitter=head_size_jitter)
df['Strung Weight (oz) jittered'] = df['Strung Weight (oz)'].apply(add_jitter, max_jitter=strung_weight_jitter)
df['Beam Width (avg. mm) jittered'] = df['Beam Width (avg. mm)'].apply(add_jitter, max_jitter=beam_width_jitter)
df['Balance (pts) jittered'] = df['Balance (pts)'].apply(add_jitter, max_jitter=balance_jitter)
df['String Density (X/in2) jittered'] = df['String Density (X/in2)'].apply(add_jitter, max_jitter=string_density_jitter)
df['Swingweight jittered'] = df['Swingweight'].apply(add_jitter, max_jitter=swingweight_jitter)
df['Stiffness jittered'] = df['Stiffness'].apply(add_jitter, max_jitter=stiffness_jitter)

# Creating & initializing columns to hold standardized values
features_to_norm = ['Head Size (in)',
                    'Strung Weight (oz)',
                    'Balance (pts)',
                    'Stiffness',
                    'Beam Width (avg. mm)',
                    'String Density (X/in2)',
                    'Swingweight',]

scaler = RobustScaler()

for feature in features_to_norm:
    df[feature + ' normed'] = scaler.fit_transform(df[feature].values.reshape(-1, 1))

normed_features = [feature + ' normed' for feature in features_to_norm]

pca = PCA(n_components=3)
pca.fit(df[normed_features])
X_pca = pca.transform(df[normed_features])

# Creating & initializing columns to hold PC values

for i in range(3):
    df['Principal Component ' + str(i + 1)] = X_pca[:, i]

# Creating & initializing columns to hold t-SNE values
tsne = TSNE(random_state=42)
rackets_tsne = tsne.fit_transform(df[normed_features])

df['t-SNE 1'] = rackets_tsne[:, 0]
df['t-SNE 2'] = rackets_tsne[:, 1]


# Defining a color palette
colors = {
    'black': 'black',
    'white': '#d9d9d9',
    'purple': '#794bc4', #CC8899 #B300FF;
    'gray': '#222222', #'#3d3d3d',
    'pink': '#ff00fb',
    'blue': '#00d0ff',
    'riviera_blue': '#00a2ff'
}

# Helper function for defining lowest slider marks
def rounduptonearest(x, base=5):
    return base * math.ceil(int(x)/base)

# FILL IN URL WHERE BLANK
# def replace_blank_urls_with_ebay(inputs):
#     return inputs[0] if type(inputs[0]) == str and inputs[0] != '' else \
#         'https://www.ebay.com/sch/i.html?_nkw={}'.format(inputs[1].replace(' ','+'))
#
# df['url'] = df[['tw_url','Model']].apply(replace_blank_urls_with_ebay, axis=1)
#
#
#
#
# df['Current/Old Models'] = df['tw_url'].apply(lambda x: \
#     'Current Model' if type(x) == str and x != '' else 'Old Model')
#
# def generate_buy_link(inputs):
#     text = "TW" if inputs[0] == 'Current Model' else 'eBay'
#     link = '['+text+'](' + inputs[1] + ')'
#     return link
#
# # new column for 'link'
# df['link'] = df[['Current/Old Models','url']].apply(generate_buy_link, axis=1)

filter_histogram_layout = {'xaxis':{'showticklabels':False},
                           'yaxis':{'showticklabels':False},
                           'height':30,
                           'margin':dict(l=25, r=25, t=0, b=3),
                           'plot_bgcolor':colors['black'],
                           'paper_bgcolor':colors['black']}


app.layout = html.Div(style={'margin':'0px','padding':"0px"},children=[
    html.Div(style={'padding-left':'10px','min-width':'200px'}, children=[ # className='twelve columns',
        html.H1(className='gradient-text', children=[
            html.Img(src=app.get_asset_url('logo.png'), height=32),
            'RACKETSPACE',
        ])],
    ),

    html.Div(id='filters-div',
             className='twelve columns inner-border',
             # style={'outline': '1px solid #794bc4','outline-offset':'-10px','padding-top':'13px','padding-bottom':'13px'},
             children=[
        html.Details(children=[
            html.Summary('Filter by Specifications',style={'color':colors['purple'],'outline':'none'}),
            html.Div(style={'padding':"0px"}, children=[
                dcc.Checklist(
                    id='current-models-only-checkbox',
                    options=[
                        {'label': 'Current Models Only', 'value': 'True'},
                    ],
                    style={'margin-top': '5px', 'margin-bottom': '5px'},
                    value=[]
                ),
                html.Div(className='twelve columns',children=[
                    html.Label("Manufacturers",style={'float':'left'}),
                    html.Button('All', id='select-all-btn', n_clicks=0, style={'outline':'none','float':'left','margin-left':'5px','border':'none'}),
                    html.Button('None', id='select-none-btn', n_clicks=0, style={'outline':'none','float':'left','margin-left':'5px','border':'none'}),
                    dcc.Dropdown(
                        className='twelve columns',
                        id='mfr-dropdown',
                        options=[{'label': mfr, 'value': mfr} for mfr in sorted(list(df['Manufacturer'].unique()))],
                        multi=True,
                        style={'border-radius':'4px', 'background-color':colors['gray'], 'color':colors['black'], 'border':'none'},
                        value=sorted(list(df['Manufacturer'].unique())),
                    ),
                ]),
                html.Div(id='filter-sliders-div', className='twelve columns', children=[
                    html.Div(className='four columns filter-slider-div', children=[
                        html.Label('Head Size (in)'),
                        dcc.Graph(
                            id='hist-head-size',
                            figure={
                                'data': [
                                    {
                                        'x': df['Head Size (in)'],
                                        'nbins': 70,
                                        'name': 'Head Size',
                                        'marker':{'color':colors['blue']},
                                        'type': 'histogram'
                                    },
                                ],
                                'layout': filter_histogram_layout
                            },
                            config={'displayModeBar':False},
                        ),
                        dcc.RangeSlider(
                            id='head-size-slider',
                            className='filter-slider',
                            min=df['Head Size (in)'].min(),
                            max=df['Head Size (in)'].max(),
                            marks={i: '{}'.format(i) for i in range(rounduptonearest(df['Head Size (in)'].min(),10),
                                                                    int(df['Head Size (in)'].max()),
                                                                    10)},
                            step=0.5,
                            tooltip = { 'always_visible': False },
                            value=[df['Head Size (in)'].min(), df['Head Size (in)'].max()]
                        ),
                    ]),

                    html.Div(className='four columns filter-slider-div', children=[
                        html.Label('Strung Weight (oz)'),
                        dcc.Graph(
                            id='hist-strung-weight',
                            figure={
                                'data': [
                                    {
                                        'x': df['Strung Weight (oz)'],
                                        'nbins': 26,
                                        'name': 'Strung Weight',
                                        'marker':{'color':colors['blue']},
                                        'type': 'histogram'
                                    },
                                ],
                                'layout': filter_histogram_layout
                            },
                            config={'displayModeBar':False},
                        ),
                        dcc.RangeSlider(
                            id='strung-weight-slider',
                            className='filter-slider',
                            min=df['Strung Weight (oz)'].min(),
                            max=df['Strung Weight (oz)'].max(),
                            marks={i: '{}'.format(i) for i in range(int(df['Strung Weight (oz)'].min()), int(df['Strung Weight (oz)'].max())+1)},
                            step=0.25,
                            tooltip={'always_visible':False},
                            value=[df['Strung Weight (oz)'].min(), df['Strung Weight (oz)'].max()]
                        ),
                    ]),

                    html.Div(className='four columns filter-slider-div', children=[
                        html.Label('Balance (pts HL/HH)'),
                        dcc.Graph(
                            id='hist-balance',
                            figure={
                                'data': [
                                    {
                                        'x': df['Balance (pts)'],
                                        'nbins': 61,
                                        'name': 'Balance',
                                        'marker':{'color':colors['blue']},
                                        'type': 'histogram'
                                    },
                                ],
                                'layout': filter_histogram_layout
                            },
                            config={'displayModeBar':False},
                        ),
                        dcc.RangeSlider(
                            id='balance-slider',
                            className='filter-slider',
                            min=df['Balance (pts)'].min(),
                            max=df['Balance (pts)'].max(),
                            marks={i: '{} HL'.format(i) if i <0  else '{} HH'.format(i) for i in range(int(df['Balance (pts)'].min()), int(df['Balance (pts)'].max())+1, 5)},
                            step=1,
                            tooltip = { 'always_visible': False },
                            value=[df['Balance (pts)'].min(), df['Balance (pts)'].max()]
                        )
                    ]),

                    html.Div(className='four columns filter-slider-div', children=[
                        html.Label('Beam Width (avg. mm)'),
                        dcc.Graph(
                            id='hist-beam-width',
                            figure={
                                'data': [
                                    {
                                        'x': df['Beam Width (avg. mm)'],
                                        'nbins': 32,
                                        'name': 'Beam Width',
                                        'marker':{'color':colors['blue']},
                                        'type': 'histogram'
                                    },
                                ],
                                'layout': filter_histogram_layout
                            },
                            config={'displayModeBar':False},
                        ),
                        dcc.RangeSlider(
                            id='beam-width-slider',
                            className='filter-slider',
                            min=df['Beam Width (avg. mm)'].min(),
                            max=df['Beam Width (avg. mm)'].max(),
                            marks={i: '{}'.format(i) for i in range(int(math.ceil(df['Beam Width (avg. mm)'].min())), int(df['Beam Width (avg. mm)'].max())+1, 3)},
                            step=0.5,
                            tooltip = { 'always_visible': False },
                            value=[df['Beam Width (avg. mm)'].min(), df['Beam Width (avg. mm)'].max()]
                        )
                    ]),

                    html.Div(className='four columns filter-slider-div', children=[
                        html.Label('String Density (X/in.²)'),
                        dcc.Graph(
                            id='hist-string-density',
                            figure={
                                'data': [
                                    {
                                        'x': df['String Density (X/in2)'],
                                        'nbins': 25,
                                        'name': 'String Density',
                                        'marker':{'color':colors['blue']},
                                        'type': 'histogram'
                                    },
                                ],
                                'layout': filter_histogram_layout
                            },
                            config={'displayModeBar':False},
                        ),
                        dcc.RangeSlider(
                            id='string-density-slider',
                            className='filter-slider',
                            min=1.8,
                            max=4.2,
                            marks={2: '2', 3: '3', 4: '4'},
                            step=0.1,
                            tooltip = { 'always_visible': False },
                            value=[1.8, 4.2]
                        )
                    ]),

                    html.Div(className='four columns filter-slider-div', children=[
                        html.Label('Swingweight'),
                        dcc.Graph(
                            id='hist-swingweight',
                            figure={
                                'data': [
                                    {
                                        'x': df['Swingweight'],
                                        'nbins': 220,
                                        'name': 'Swingweight',
                                        'marker':{'color':colors['blue']},
                                        'type': 'histogram'
                                    },
                                ],
                                'layout': filter_histogram_layout
                            },
                            config={'displayModeBar':False},
                        ),
                        dcc.RangeSlider(
                            id='swingweight-slider',
                            className='filter-slider',
                            min=df['Swingweight'].min(),
                            max=df['Swingweight'].max(),
                            marks={i: '{}'.format(i) for i in range(rounduptonearest(df['Swingweight'].min(),50), int(df['Swingweight'].max())+1, 50)},
                            step=1,
                            tooltip={'always_visible':False},
                            value=[df['Swingweight'].min(), df['Swingweight'].max()]
                        ),
                    ]),

                    html.Div(className='four columns filter-slider-div', children=[
                        html.Label('Stiffness'),
                        dcc.Graph(
                            id='hist-stiffness',
                            figure={
                                'data': [
                                    {
                                        'x': df['Stiffness'],
                                        'nbins': 62,
                                        'name': 'Stiffness',
                                        'marker':{'color':colors['blue']},
                                        'type': 'histogram'
                                    },
                                ],
                                'layout': filter_histogram_layout
                            },
                            config={'displayModeBar':False},
                        ),
                        dcc.RangeSlider(
                            id='stiffness-slider',
                            # className='filter-slider',
                            min=df['Stiffness'].min(),
                            max=df['Stiffness'].max(),
                            marks={i: '{}'.format(i) for i in range(rounduptonearest(df['Stiffness'].min(),10), int(df['Stiffness'].max())+1, 10)},
                            step=1,
                            tooltip={'always_visible':False},
                            value=[df['Stiffness'].min(), df['Stiffness'].max()]
                        ),
                    ]),
                ]),
            ]),
        ]),
    ]),
    html.Div(className='three columns inner-border', id='settings', children=[
        dcc.Tabs(id="tabs", value='tab-1', parent_className='custom-tabs', className='custom-tabs-container', children=[
            dcc.Tab(label='Choose Axes', value='tab-1', className='custom-tab',
                    selected_className='custom-tab--selected',
                    children=[
                html.Label('X-Axis', style={'margin-top':'10px'}),
                dcc.Dropdown(
                    id='x-axis-dropdown',
                    className='custom-dropdown',
                    style={'border-radius':'4px', 'background-color': colors['gray'], 'color':colors['black'], 'border': 'none'},
                    options=[{'label': 'Head Size (in)', 'value': 'Head Size (in)'},
                             {'label': 'Strung Weight (oz)', 'value': 'Strung Weight (oz)'},
                             {'label': 'Beam Width (avg. mm)', 'value': 'Beam Width (avg. mm)'},
                             {'label': 'Balance (pts)', 'value': 'Balance (pts)'},
                             {'label': 'String Density (intersections / sq. in.)', 'value': 'String Density (X/in2)'},
                             {'label': 'Swingweight', 'value': 'Swingweight'},
                             {'label': 'Stiffness', 'value': 'Stiffness'},
                             {'label': 'Current/Old Models', 'value': 'Current/Old Models'}],
                    value='Head Size (in)'
                ),
                html.Label('Y-Axis', style={'margin-top':'10px'}),
                dcc.Dropdown(
                    id='y-axis-dropdown',
                    className='custom-dropdown',
                    style={'border-radius':'4px', 'background-color': colors['gray'], 'color':colors['black'], 'border': 'none'}, #
                    options=[{'label': 'Head Size (in)', 'value': 'Head Size (in)'},
                             {'label': 'Strung Weight (oz)', 'value': 'Strung Weight (oz)'},
                             {'label': 'Beam Width (avg. mm)', 'value': 'Beam Width (avg. mm)'},
                             {'label': 'Balance (pts)', 'value': 'Balance (pts)'},
                             {'label': 'String Density (intersections / sq. in.)', 'value': 'String Density (X/in2)'},
                             {'label': 'Swingweight', 'value': 'Swingweight'},
                             {'label': 'Stiffness', 'value': 'Stiffness'},
                             {'label': 'Current/Old Models', 'value': 'Current/Old Models'}],
                    value='Strung Weight (oz)'
                ),
                dcc.Checklist(
                        id='jitter-checkbox',
                        options=[{'label': 'Jitter', 'value': 'True'}],
                        value=[],
                        style={'margin-top':'10px'}
                ),
            ]),
            dcc.Tab(label='Combine Axes', value='tab-2',className='custom-tab',
                    selected_className='custom-tab--selected', children=[
                html.Label('Dimensionality Reduction Type', style={'margin-top':'10px'}),
                dcc.Dropdown(
                    id='dim-red-algo-dropdown',
                    className='custom-dropdown',
                    style={'background-color':colors['gray'], 'color':colors['black'], 'border-radius':'4px', 'border': 'none', 'margin-bottom':'10px'},
                    options=[{'label': 'PCA', 'value': 'PCA'},
                             {'label': 't-SNE', 'value': 't-SNE'}],
                    value='PCA'
                ),
            ]),

        ]),
        html.Label('Color', style={'margin-top':'10px'}),
        dcc.Dropdown(
            id='color-dropdown',
            style={'background-color':colors['gray'], 'color':colors['black'], 'border-radius':'4px', 'border': 'none'},
            className='custom-dropdown',
            options=[{'label': 'Head Size (in)', 'value': 'Head Size (in)'},
                     {'label': 'Strung Weight (oz)', 'value': 'Strung Weight (oz)'},
                     {'label': 'Beam Width (avg. mm)', 'value': 'Beam Width (avg. mm)'},
                     {'label': 'Balance (pts)', 'value': 'Balance (pts)'},
                     {'label': 'String Density (intersections / sq. in.)', 'value': 'String Density (X/in2)'},
                     {'label': 'Swingweight', 'value': 'Swingweight'},
                     {'label': 'Stiffness', 'value': 'Stiffness'},
                     {'label': 'Current/Old Models', 'value': 'Current/Old Models'},
                     {'label': 'Principal Component 3', 'value': 'Principal Component 3'}],
            value='Balance (pts)'
        ),
        # html.Label('Axes to Use for Similarity (Below) and Combined Axes Plots', style={'margin-top':'10px'}),

    ]), #style={'padding':'10px','outline': '1px solid #794bc4','outline-offset':'-10px',}
    html.Div(className='nine columns inner-border',
			 id='graph-div',
			 # style={'outline': '1px solid #794bc4','outline-offset':'-10px','padding':'20px'},
			 children=[
        dcc.Graph(
            id='racquetspace_graph',
        ),
    ]),
    html.Div(className='twelve columns inner-border',id='table-div', children=[
        html.Label('Search or click a racket in the chart to see details and similar rackets.',
                   style={'margin-top':'5px','margin-bottom':'10px'}),
        dcc.Dropdown(
            id='name-dropdown',
            options=[{'label': model, 'value': model} for model in sorted(list(df['Name'].unique()))],
            multi=False,
            style={'background-color':colors['gray'], 'color':colors['black'], 'border-radius':'4px', 'border':'none', 'margin-bottom':'10px'},
            value=None
        ),
    html.Details(children=[
            html.Summary('Axes to Consider for Similarity',style={'color':colors['purple'],'outline':'none','margin-bottom':'10px'}),
            html.Div(style={'margin-bottom':"10px"}, children=[
                dcc.Checklist(
                    id='axes-checklist',
                    options=[
                        {'label': 'Head Size (in)', 'value': 'Head Size (in)'},
                        {'label': 'Strung Weight (oz)', 'value': 'Strung Weight (oz)'},
                        {'label': 'Balance (pts)', 'value': 'Balance (pts)'},
                        {'label': 'Stiffness', 'value': 'Stiffness'},
                        {'label': 'Beam Width (avg. mm)', 'value': 'Beam Width (avg. mm)'},
                        {'label': 'Swingweight', 'value': 'Swingweight'},
                        {'label': 'String Density (intersections / sq. in.)', 'value': 'String Density (X/in2)'},
                    ],
                    value=['Head Size (in)', 'Strung Weight (oz)', 'Balance (pts)', 'Stiffness', 'Beam Width (avg. mm)',
                           'Swingweight', 'String Density (X/in2)'],
                    style={'color': 'white'}
                ),
            ]),
        ]),
        dash_table.DataTable(
            id='selected-rackets-table',
            columns=[
				{'name': 'Distance', 'id': 'Distance', 'type': 'numeric', 'format': Format(
                    scheme=Scheme.fixed,
                    precision=3,)},
				{'name': 'Manufacturer', 'id': 'Manufacturer',},
				{'name': 'Model', 'id': 'Model'},
				{'name': 'Head Size (in)', 'id': 'Head Size (in)'},
				{'name': 'Strung Weight (oz)', 'id': 'Strung Weight (oz)'},
				{'name': 'Balance (pts)', 'id': 'Balance (pts)'},
				{'name': 'Stiffness', 'id': 'Stiffness'},
				{'name': 'Beam Width (avg. mm)', 'id': 'Beam Width (avg. mm)', 'type': 'numeric', 'format': Format(
                    scheme=Scheme.fixed,
                    precision=2,)},
                {'name': 'Swingweight', 'id': 'Swingweight'},
				{'name': 'String Density (X/in2)', 'id': 'String Density (X/in2)', 'type': 'numeric', 'format': Format(
                    scheme=Scheme.fixed,
                    precision=3,)},
				{'name': 'link', 'id': 'link', 'presentation':'markdown'},
            ],
			style_header={
				'backgroundColor': colors['black'],
				'color': colors['purple'],
				# 'fontWeight': 'bold'
			},
			style_cell={
				'backgroundColor': colors['gray'],
				'color': '#BBBBBB',
                'border': '1px solid #444444'
			},
			style_cell_conditional=[
				{
					'if': {'column_id': c},
					'textAlign': 'left'
				} for c in ['Manufacturer', 'Model']
			],
		)
    ]),

    html.Div([
        html.Pre(id='click-data'),
    ], className='nine columns'),
    html.Div(id='hidden-div', style={'display':'none'}),
])
@app.callback(
    Output('mfr-dropdown', 'value'),
    [Input('select-all-btn', 'n_clicks'),
     Input('select-none-btn', 'n_clicks')])
def select_all_none_mfrs(select_all, select_none):
    ctx = dash.callback_context # see which input triggered the callback
    if ctx.triggered[0]['prop_id'].split('.')[0] == 'select-all-btn' or not ctx.triggered:
        return sorted(list(df['Manufacturer'].unique()))
    else:
        return []

# @app.callback(
#     Output('hidden-div', 'children'),
#     [Input('axes-checklist', 'value')])
# def update_reduced_dimensions(axes_checklist):
#     # types_of_norming = {}  # min-max-scaler,
#     # Update PCA columns
#     if len(axes_checklist) < 3:
#         df['Principal Component 3'] = 0
#     if len(axes_checklist) < 2:
#         df['Principal Component 2'] = 0
#     if len(axes_checklist) < 1:
#         df['Principal Component 1'] = 0
#     else:
#         scaler = RobustScaler()
#
#         for feature in axes_checklist:
#             df[feature + ' normed'] = scaler.fit_transform(df[feature].values.reshape(-1, 1))
#
#         normed_features = [feature + ' normed' for feature in axes_checklist]
#
#         pca = PCA(n_components=min(3,len(axes_checklist)))
#         pca.fit(df[normed_features])
#         X_pca = pca.transform(df[normed_features])
#
#         for i in range(X_pca.shape[1]):
#             df['Principal Component ' + str(i + 1)] = X_pca[:, i]
#
#     # Update t-SNE columns
#     rackets_tsne = tsne.fit_transform(df[axes_checklist])
#     df['t-SNE 1'] = rackets_tsne[:, 0]
#     df['t-SNE 2'] = rackets_tsne[:, 1]
#
#     return None

@app.callback(
    Output('racquetspace_graph', 'figure'),
    [Input('mfr-dropdown', 'value'),
     Input('head-size-slider', 'value'),
     Input('strung-weight-slider', 'value'),
     Input('balance-slider', 'value'),
     Input('beam-width-slider', 'value'),
     Input('string-density-slider', 'value'),
     Input('swingweight-slider', 'value'),
     Input('stiffness-slider', 'value'),
     Input('jitter-checkbox', 'value'),
     Input('current-models-only-checkbox', 'value'),
     Input('tabs', 'value'),
     Input('x-axis-dropdown', 'value'),
     Input('y-axis-dropdown', 'value'),
     Input('dim-red-algo-dropdown', 'value'),
     Input('color-dropdown', 'value'),
     Input('axes-checklist', 'value')])
def update_scatter(mfrs, head_size_range, strung_weight_range,
                  balance_range, beam_width_range, string_density_range,
                  swingweight_range, stiffness_range, jitter,
                  current_models_only, tabs, x_axis, y_axis, dim_red_algo, color, axes_checklist):

    current_model_states = ['Current Model'] if current_models_only else \
                           ['Current Model', 'Old Model']

    filtered_df = df[(df['Manufacturer'].isin(mfrs)) & \
                     (df['Current/Old Models'].isin(current_model_states)) & \
                     (df['Head Size (in)'] >= head_size_range[0]) & \
                     (df['Head Size (in)'] <= head_size_range[1]) & \
                     (df['Strung Weight (oz)'] >= strung_weight_range[0]) & \
                     (df['Strung Weight (oz)'] <= strung_weight_range[1]) & \
                     (df['Balance (pts)'] >= balance_range[0]) & \
                     (df['Balance (pts)'] <= balance_range[1]) & \
                     (df['Beam Width (avg. mm)'] >= beam_width_range[0]) & \
                     (df['Beam Width (avg. mm)'] <= beam_width_range[1]) & \
                     (df['String Density (X/in2)'] >= string_density_range[0]) & \
                     (df['String Density (X/in2)'] <= string_density_range[1]) & \
                     (df['Swingweight'] >= swingweight_range[0]) & \
                     (df['Swingweight'] <= swingweight_range[1]) & \
                     (df['Stiffness'] >= stiffness_range[0]) & \
                     (df['Stiffness'] <= stiffness_range[1])]

    # if PCA, then use those axes
    if tabs == 'tab-1':
        x_col = (x_axis + ' jittered') if jitter else x_axis
        y_col = (y_axis + ' jittered') if jitter else y_axis
    elif (tabs == 'tab-2') and (dim_red_algo == 'PCA'):
        x_col = 'Principal Component 1'
        y_col = 'Principal Component 2'
    elif (tabs == 'tab-2') and (dim_red_algo == 't-SNE'):
        x_col = 't-SNE 1'
        y_col = 't-SNE 2'
    else:
        raise Exception('invalid tab setting')

    # elif jitter, then use jittered versions
    # else use regular versions...
    

    # Calculate base PCA columns if first time calling the function

    # Calculate dynamic PCA columns if ***checkbox enabled***
    fig = px.scatter(filtered_df, 
                 x=x_col, y=y_col,
                 color=color, hover_name='Name',
                 labels={
                    x_col: x_col,
                    y_col: y_col,
                    color: color.split(' (')[0] if '(' in color else color,
                 },
                 title='(Showing: ' + str(len(filtered_df)) + ')',
				 height=400,
                 hover_data= ['ID'] + axes_checklist,
                 opacity=1.0,
                 color_continuous_scale=[colors['pink'], colors['blue']],
                 color_discrete_sequence=[colors['pink'], colors['blue']])

    fig.update_layout(
        margin=dict(l=50, r=0, t=0, b=30), # t=10
        title_font_color=colors['purple'],
    )
    fig.update_xaxes(color=colors['purple'],
                     showgrid=True, gridwidth=1, gridcolor='#333333',
                     zeroline=True, zerolinewidth=2, zerolinecolor='#444444',
                     tickfont=dict(color='gray')),
    fig.update_yaxes(color=colors['purple'],
                     showgrid=True, gridwidth=1, gridcolor='#333333',
                     zeroline=True, zerolinewidth=2, zerolinecolor='#444444',
                     tickfont=dict(color='gray')),
    fig.update_traces(marker_size=5)
    fig.update_layout(plot_bgcolor=colors['gray'],
                      paper_bgcolor=colors['black'])
    fig.update_layout(coloraxis_colorbar=dict(
        tickfont=dict(color='gray'),
        title_font_color=colors['purple'],
        title_font_size=14
    ))

    fig.update_layout(
                      font_color='gray',
                      legend_title_font_color='green',
                      title_font_size=14,
                      legend=dict(
                          title=dict(
                              font=dict(color=colors['purple'],
                                        size=14)
                          ),
                          font=dict(
                              color="gray"
                          )
                      )
    )
    fig.update_layout(
        title={
            # 'text': "Plot Title",
            'font':{'size':12,'color':'gray'},
            'y': 0.97,
            'x': .48,
            'xanchor': 'center',
            'yanchor': 'top'})
    fig.update_layout(transition_duration=1000)

    return fig

@app.callback(
    Output('name-dropdown', 'value'),
    [Input('racquetspace_graph', 'clickData')])
def click_overwrite_search(clickData):
    try:
        return clickData['points'][0]['hovertext']
    except TypeError:
        return None

def calculate_distance(inputs, targets):
    return np.sqrt(np.sum([(a-b)**2 for a, b in zip(inputs, targets)]))

@app.callback(
    Output('selected-rackets-table', 'data'),
    [Input('name-dropdown', 'value'),
     Input('axes-checklist', 'value')])
def select_racket_table(name_dropdown, axes_checklist):
    if name_dropdown:
        id = int(df[df['Name'] == name_dropdown]['ID'].values[0])
        targets = [float(df[df['ID'] == id][axis]) for axis in axes_checklist]
        df['Distance'] = df[axes_checklist].apply(
            calculate_distance, targets=targets, axis=1)
        return df.sort_values("Distance").head(10).to_dict('records')
    else:
        return None

# @app.callback(
#     Output('click-data', 'children'),
#     [Input('racquetspace_graph', 'clickData')])
# def select_racket_json(clickData):
#     return json.dumps(clickData, indent=2)


if __name__ == '__main__':
    app.run_server(debug=True)
