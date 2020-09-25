# import json
import math
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
app.title = "RktSpc"

# df = pd.read_csv('https://gist.githubusercontent.com/chriddyp/5d1ea79569ed194d432e56108a04d188/raw/a9f9e8076b837d541398e999dcbac2b2826a81f8/gdp-life-exp-2007.csv')	
raw_df = pd.read_csv('data/racket_specs.csv')

# initial data source filtering (NaNs & erroneous values)
df = raw_df[(raw_df['Head Size (in)']>=60) & \
            (raw_df['Head Size (in)']<=140) & \
            (raw_df['Strung Weight (oz)']>=5) & \
            (raw_df['Strung Weight (oz)']<=20) & \
            (raw_df['Beam Width (avg. mm)']>=5) & \
            (raw_df['Beam Width (avg. mm)']<=50) & \
            (raw_df['Balance (pts)'] >= -20) & \
            (raw_df['Balance (pts)'] <= 20) & \
            (raw_df['String Density (X/in2)'] >= 1) & \
            (raw_df['String Density (X/in2)'] <= 6) & \
            (raw_df['Swingweight'] >= 100) & \
            (raw_df['Swingweight'] <= 600) & \
            (raw_df['Stiffness'] >= 10) & \
            (raw_df['Stiffness'] <= 100)].copy()

df.rename(columns={'url': 'tw_url', 'img_url': 'tw_img_url'},
                   # 'Unnamed: 0': 'ID', 'mfr': 'Manufacturer'},
          inplace=True)

df['ID'] = df.index
df['Distance'] = 0

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

for feature in features_to_norm:
    df[feature + ' normed'] = 0

# Creating & initializing columns to hold PC values

for i in [1,2,3]:
    df['Principal Component '+str(i)] = 0

# Creating & initializing columns to hold t-SNE values
tsne = TSNE(random_state=42)
for i in [1,2]:
    df['t-SNE '+str(i)] = 0

# Defining a color palette
colors = {
    'black': 'black',
    'white': '#d9d9d9',
    'purple': '#794bc4', #B300FF;
    'gray': '#3d3d3d',
    'pink': '#ff00fb',
    'blue': '#00d0ff',
    'riviera_blue': '#00a2ff'
}

# Helper function for defining lowest slider marks
def rounduptonearest(x, base=5):
    return base * math.ceil(int(x)/base)

# FILL IN URL WHERE BLANK
def replace_blank_urls_with_ebay(s):
    return s[0] if type(s[0]) == str and s[0] != '' else \
        'https://www.ebay.com/sch/i.html?_nkw={}'.format(s[1].replace(' ','+'))

df['url'] = df[['tw_url','Model']].apply(replace_blank_urls_with_ebay, axis=1)

df['Current/Old Models'] = df['tw_url'].apply(lambda x: \
    'Current Model' if type(x) == str and x != '' else 'Old Model')

app.layout = html.Div(style={'padding':"10px"}, children=[
    html.Div(className='twelve columns', children=[
        html.H1(children=[
                            html.Img(src=app.get_asset_url('logo.png'), height=32),
                            'RACKETSPACE',
                         ],
        )],
    ),


    html.Div(id='filters-div', className='twelve columns', style={'outline': '1px solid #794bc4', 'padding':'10px'}, children=[
        html.Details(children=[
            html.Summary('Filters',style={'color':colors['purple'],'outline':'none'}),#children=[html.Label('Filters')]),

            html.Div(style={'padding':"0px"}, children=[
                html.Div(className='twelve columns', children=[
                    html.Label('Manuacturers',style={'float':'left', 'margin-bottom':'2px'}),
                    html.Button('All', id='select-all-btn', n_clicks=0, style={'float':'left','padding':'2px','margin-left':'5px', 'margin-top':'5px'}), #
                    html.Button('None', id='select-none-btn', n_clicks=0, style={'float':'left','padding':'2px','margin-left':'5px', 'margin-top':'5px'}),
                    dcc.Dropdown(
                        className='twelve columns',
                        id='mfr-dropdown',
                        options=[{'label': mfr, 'value': mfr} for mfr in sorted(list(df['Manufacturer'].unique()))],
                        multi=True,
                        style={'border-radius':'4px'}, # 'background-color':colors['white'], 'color':colors['black'],
                        value=sorted(list(df['Manufacturer'].unique()))
                    ),
                ]),
                html.Div(className='twelve columns', children=[
                    dcc.Checklist(
                        id='current-models-only-checkbox',
                        options=[
                            {'label': 'Current Models Only', 'value': 'True'},
                        ],
                        style={'margin-top': '5px', 'margin-bottom': '5px'},
                        value=[]
                    ),
                ]),
                html.Div(id='six-slider-filter-div', className='twelve columns', children=[

                    html.Div(className='four columns', style={'padding-left':'15px', 'padding-right':'15px'},
                        children=[
                            html.Label(children=['Head Size (in)',
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

                        ]
                    ),

                    html.Div(className='four columns', style={'padding-left':'15px', 'padding-right':'15px'},
                        children=[
                            html.Label(children=['Strung Weight (oz)',
                                dcc.RangeSlider(
                                    id='strung-weight-slider',
                                    className='filter-slider',
                                    min=df['Strung Weight (oz)'].min(),
                                    max=df['Strung Weight (oz)'].max(),
                                    marks={i: '{}'.format(i) for i in range(int(df['Strung Weight (oz)'].min()), int(df['Strung Weight (oz)'].max())+1)},
                                    step=0.25,
                                    tooltip = { 'always_visible': False },
                                    value=[df['Strung Weight (oz)'].min(), df['Strung Weight (oz)'].max()]
                                )
                            ]),
                        ]
                    ),

                    html.Div(className='four columns', style={'padding-left':'15px', 'padding-right':'15px'},
                        children=[
                            html.Label(children=['Balance (pts HL/HH)',
                                dcc.RangeSlider(
                                    id='balance-slider',
                                    className='filter-slider',
                                    min=df['Balance (pts)'].min(),
                                    max=df['Balance (pts)'].max(),
                                    marks={i: '{} HL'.format(i) if i <0  else '{} HH'.format(i) for i in range(int(df['Balance (pts)'].min()), int(df['Balance (pts)'].max())+1, 5)},
                                    step=0.5,
                                    tooltip = { 'always_visible': False },
                                    value=[df['Balance (pts)'].min(), df['Balance (pts)'].max()]
                                )
                            ]),
                        ]
                    ),

                    html.Div(className='four columns', style={'padding-left':'15px', 'padding-right':'15px'},
                        children=[
                            html.Label(children=['Beam Width (avg. mm)',
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
                        ]
                    ),

                    html.Div(className='four columns', style={'padding-left':'15px', 'padding-right':'15px'},
                        children=[
                            html.Label(children=['String Density (X/in.²)',
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

                        ]
                    ),

                    html.Div(className='four columns', style={'padding-left':'15px', 'padding-right':'15px'},
                        children=[
                            html.Label(children=['Swingweight',
                                dcc.RangeSlider(
                                    id='swingweight-slider',
                                    className='filter-slider',
                                    min=df['Swingweight'].min(),
                                    max=df['Swingweight'].max(),
                                    marks={i: '{}'.format(i) for i in range(rounduptonearest(df['Swingweight'].min(),50), int(df['Swingweight'].max())+1, 50)},
                                    step=1,
                                    tooltip = { 'always_visible': False },
                                    value=[df['Swingweight'].min(), df['Swingweight'].max()]
                                )
                            ]),
                        ]
                    ),

                    html.Div(className='four columns', style={'padding-left':'15px', 'padding-right':'15px'},
                        children=[
                            html.Label(children=['Stiffness',
                                dcc.RangeSlider(
                                    id='stiffness-slider',
                                    className='filter-slider',
                                    min=df['Stiffness'].min(),
                                    max=df['Stiffness'].max(),
                                    marks={i: '{}'.format(i) for i in range(rounduptonearest(df['Stiffness'].min(),10), int(df['Stiffness'].max())+1, 10)},
                                    step=1,
                                    tooltip = { 'always_visible': False },
                                    value=[df['Stiffness'].min(), df['Stiffness'].max()]
                                )
                            ]),
                        ]
                    ),

                ]),
            ]),
        ]),
    ]),
    html.Br(),
    html.Br(),
    html.Div(className='three columns', id='settings', style={'outline': '1px solid #794bc4','margin-top':'20px','padding':'10px', 'margin-right':'20px'}, children=[
        dcc.Tabs(id="tabs", value='tab-1', parent_className='custom-tabs', className='custom-tabs-container', children=[
            dcc.Tab(label='Choose Axes', value='tab-1', className='custom-tab',
                    selected_className='custom-tab--selected',
                    children=[
                html.Label('X-Axis', style={'margin-top':'10px'}),
                dcc.Dropdown(
                    id='x-axis-dropdown',
                    className='custom-dropdown',
                    style={'background-color':colors['white'], 'color':colors['black'], 'border-radius':'4px'},
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
                    style={'border-radius':'4px', 'background-color':colors['white'], 'color':colors['purple'],}, #
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
                    style={'background-color':colors['white'], 'color':colors['black'], 'border-radius':'4px'},
                    options=[{'label': 'PCA', 'value': 'PCA'},
                             {'label': 't-SNE', 'value': 't-SNE'}],
                    value='PCA'
                ),
            ]),

        ]),
        html.Label('Color', style={'margin-top':'10px'}),
        dcc.Dropdown(
            id='color-dropdown',
            style={'background-color':colors['white'], 'color':colors['black'], 'border-radius':'4px'},
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
        html.Label('Axes to Use for Similarity (Below) and Combined Axes Plots', style={'margin-top':'10px'}),
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
    html.Div(className='nine columns',
			 id='graph-div',
			 style={'outline': '1px solid #794bc4','margin-top':'20px'},
			 children=[
        dcc.Graph(
            id='racquetspace_graph',
        ),
    ]),
    html.Div(style={'outline': '1px solid #794bc4','margin-top':'20px','padding':'10px'}, children=[
        # dcc.Markdown('''**Click a racket in the graph above or search for one below to see details and most similar rackets.**'''),
        html.Label('Click a racket in the graph above or search for one below to see details and most similar rackets.', style={'margin-top':'10px'}),
        dcc.Dropdown(
            id='model-dropdown',
            options=[{'label': model, 'value': model} for model in sorted(list(df['Model'].unique()))],
            multi=False,
            style={'background-color':colors['white'], 'color':colors['black'], 'border-radius':'4px'},
            value=None
        ),
        dash_table.DataTable(
            id='selected-rackets-table',
            columns=[
				# {'name': 'ID', 'id': 'ID'},
				{'name': 'Distance', 'id': 'Distance','format': Format(precision=0,scheme=Scheme.fixed)},
				{'name': 'Manufacturer', 'id': 'Manufacturer',},
				{'name': 'Model', 'id': 'Model'},
				{'name': 'Head Size (in)', 'id': 'Head Size (in)'},
				{'name': 'Strung Weight (oz)', 'id': 'Strung Weight (oz)'},
				{'name': 'Balance (pts)', 'id': 'Balance (pts)'},
				{'name': 'Stiffness', 'id': 'Stiffness'},
				{'name': 'Beam Width (avg. mm)', 'id': 'Beam Width (avg. mm)'},
                {'name': 'Swingweight', 'id': 'Swingweight'},
				{'name': 'String Density (X/in2)', 'id': 'String Density (X/in2)'},
				{'name': 'url', 'id': 'url'},
                # {"name": i, "id": i}
                # for i in [#"ID", # TODO: hide this one
                #           "Distance",
                #           "Manufacturer",
                #           "Model",
                #           "Head Size (in)",
                #           "Strung Weight (oz)",
                #           "Balance (pts)",
                #           "Stiffness",
                #           "Beam Width (avg. mm)",
                #           "Swingweight",
                #           "String Density (X/in2)",
                #           "url"]
            ],
			style_header={
				'backgroundColor': colors['black'],
				'color': colors['purple'],
				# 'fontWeight': 'bold'
			},
			style_cell={
				'backgroundColor': colors['gray'],
				'color': colors['white'],
			},
			style_cell_conditional=[
				{
					'if': {'column_id': c},
					'textAlign': 'left'
				} for c in ['Manufacturer', 'Model']
			],
		)

		# 'format': Format(
		#                 scheme=Scheme.fixed,
		#                 precision=2,
		#                 group=Group.yes,
		#                 groups=3,
		#                 group_delimiter='.',
		#                 decimal_delimiter=',',
		#                 symbol=Symbol.yes,
		#                 symbol_prefix=u'€')
		#         )
    ], className='twelve columns'),

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

@app.callback(
    Output('hidden-div', 'children'),
    [Input('axes-checklist', 'value')])
def update_reduced_dimensions(axes_checklist):
    # types_of_norming = {}  # min-max-scaler,
    # Update PCA columns
    if len(axes_checklist) < 3:
        df['Principal Component 3'] = 0
    if len(axes_checklist) < 2:
        df['Principal Component 2'] = 0
    if len(axes_checklist) < 1:
        df['Principal Component 1'] = 0
    else:
        scaler = RobustScaler()

        for feature in axes_checklist:
            df[feature + ' normed'] = scaler.fit_transform(df[feature].values.reshape(-1, 1))

        normed_features = [feature + ' normed' for feature in axes_checklist]

        pca = PCA(n_components=min(3,len(axes_checklist)))
        pca.fit(df[normed_features])
        X_pca = pca.transform(df[normed_features])

        for i in range(X_pca.shape[1]):
            df['Principal Component ' + str(i + 1)] = X_pca[:, i]

    # Update t-SNE columns
    rackets_tsne = tsne.fit_transform(df[axes_checklist])
    df['t-SNE 1'] = rackets_tsne[:, 0]
    df['t-SNE 2'] = rackets_tsne[:, 1]

    return None

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
                 color=color, hover_name='Model',
                 labels={
                    x_col: x_col,
                    y_col: y_col,
                    color: color.split(' (')[0]
                    #color: color.replace(' (','\n\n\n\n(#')
                 },
				 height=483,
                 hover_data= ['ID'] + axes_checklist,
                 opacity=1.0,
                 color_continuous_scale=[colors['pink'], colors['blue']],
                 color_discrete_sequence=[colors['pink'], colors['blue']])#, 
                 # log_x=True, size_max=60) "Beam Width (avg. mm)"

    fig.update_traces(marker_size=5)
    fig.update_layout(plot_bgcolor=colors['gray'], paper_bgcolor=colors['black'], font_color=colors['white'])

    fig.update_layout(transition_duration=1000)

    return fig

@app.callback(
    Output('model-dropdown', 'value'),
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
    [Input('model-dropdown', 'value'),
     Input('axes-checklist', 'value')])
def select_racket_table(model_dropdown, axes_checklist):
    try:
        id = int(df[df['Model'] == model_dropdown]['ID'])
        # df['Distance'] = id
        # x = float(df[df['ID'] == id]['Principal Component 1'])
        # y = float(df[df['ID'] == id]['Principal Component 2'])
        # z = float(df[df['ID'] == id]['Principal Component 3'])
        targets = [float(df[df['ID'] == id][axis]) for axis in axes_checklist]
        df['Distance'] = df[axes_checklist].apply(
            calculate_distance, targets=targets, axis=1)
        return df.sort_values("Distance").head(10).to_dict('records')
    except TypeError:
        return None

# @app.callback(
#     Output('racquetspace_graph', 'hoverData'),
#     [Input('model-dropdown', 'value')])
# def update_chart_hover(model_dropdown):
# 	if model_dropdown:
# 		{
# 			"points": [
# 				{
# 					"curveNumber": 0,
# 					"pointNumber": int(df[df['Model'] == model_dropdown]['ID']),
# 					"pointIndex": int(df[df['Model'] == model_dropdown]['ID']),
# 					"x": int(df[df['Model'] == model_dropdown]['t-SNE 1']),
# 					"y": int(df[df['Model'] == model_dropdown]['t-SNE 2']),
# 					"customdata": [
# 					]
# 				}
# 			]
# 		}
# 	else:
# 		return {}

# @app.callback(
#     Output('click-data', 'children'),
#     [Input('racquetspace_graph', 'clickData')])
# def select_racket_json(clickData):
#     return json.dumps(clickData, indent=2)


if __name__ == '__main__':
    app.run_server(debug=True)
