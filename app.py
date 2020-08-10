import numpy as np
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# external_stylesheets = ['style.css']

app = dash.Dash(__name__)#, external_stylesheets=external_stylesheets)

# df = pd.read_csv('https://gist.githubusercontent.com/chriddyp/5d1ea79569ed194d432e56108a04d188/raw/a9f9e8076b837d541398e999dcbac2b2826a81f8/gdp-life-exp-2007.csv')	
raw_df = pd.read_csv('data/racquet_specs.csv')

# initial data source filtering (NaNs & erroneous values)
df = raw_df[(raw_df['Beam Width (avg. mm)']>5) & \
                        (raw_df['Head Size (in)']>60) & \
                         (raw_df['Strung Weight (oz)']>5) & \
                         (raw_df['Balance (pts)'] >= -15) & \
                         (raw_df['Balance (pts)'] <= 15)]

df.rename(columns={'url': 'tw_url', 'img_url': 'tw_img_url'},
		  inplace=True)

head_size_max_jitter 	  = 0.5
strung_weight_max_jitter  = 0.05
beam_width_max_jitter 	  = 0.5
balance_max_jitter 		  = 0.5
string_density_max_jitter = 0.05
swingweight_max_jitter    = 0.5
stiffness_max_jitter      = 0.5

def add_jitter(input, max_jitter):
    return input + np.random.uniform(-max_jitter, max_jitter)

df['Head Size (in) jittered'] = df['Head Size (in)'].apply(add_jitter, max_jitter=head_size_max_jitter)
df['Strung Weight (oz) jittered'] = df['Strung Weight (oz)'].apply(add_jitter, max_jitter=strung_weight_max_jitter)
df['Beam Width (avg. mm) jittered'] = df['Beam Width (avg. mm)'].apply(add_jitter, max_jitter=beam_width_max_jitter)
df['Balance (pts) jittered'] = df['Balance (pts)'].apply(add_jitter, max_jitter=balance_max_jitter)
df['String Density (intersections / sq. in.) jittered'] = df['String Density (intersections / sq. in.)'].apply(add_jitter, max_jitter=string_density_max_jitter)
df['Swingweight jittered'] = df['Swingweight'].apply(add_jitter, max_jitter=swingweight_max_jitter)
df['Stiffness jittered'] = df['Stiffness'].apply(add_jitter, max_jitter=stiffness_max_jitter)

fields_to_norm = [] # list out all 7 strings of above col names

types_of_norming = {} # min-max-scaler, 



colors = {
	'black': 'black',
	'white': '#d9d9d9',
	'purple': '#794bc4', #B300FF;
	'gray': '#3d3d3d',
	'pink': '#ff00fb',
	'blue': '#00d0ff'
}

# FILL IN URL WHERE BLANK
def replace_blank_urls_with_ebay(s):
    return s[0] if type(s[0]) == str and s[0] != '' else \
        'https://www.ebay.com/sch/i.html?_nkw={}'.format(s[1].replace(' ','+'))

df['url'] = df[['tw_url','name']].apply(replace_blank_urls_with_ebay, axis=1)

df['Current/Old Models'] = df['tw_url'].apply(lambda x: \
	'Current Model' if type(x) == str and x != '' else 'Old Model')

app.layout = html.Div(style={'padding':"20px"}, children=[
	html.Div(className='twelve columns', children=[ 
		html.H1(children=[
							'RacquetSpace', 
							html.Img(src=app.get_asset_url('logo.png'),height=50)
						 ],
		)],
	),


	html.Div(id='filters-div', className='twelve columns', style={'outline': '1px solid #794bc4', 'padding':'10px'}, children=[ 
		html.Details(children=[ 
	    	html.Summary('Filters',style={'color':colors['purple'],'outline':'none'}),#children=[html.Label('Filters')]),
	    	
	    	html.Div(style={'padding':"0px"}, children=[
				html.Div(className='twelve columns', children=[ 
					html.Label('Manufacturers',style={'float':'left', 'margin-bottom':'2px'}),
					html.Button('Select All', id='select-all-btn', n_clicks=0, style={'float':'left','padding':'2px','margin-left':'5px', 'margin-top':'5px'}), # 
					dcc.Dropdown(
						className='twelve columns',
						id='mfr-dropdown',
						options=[{'label': mfr, 'value': mfr} for mfr in sorted(list(df['mfr'].unique()))],
						multi=True,
						style={'background-color':colors['white'], 'color':colors['black'], 'border-radius':'4px'},
						value=sorted(list(df['mfr'].unique()))
					),
				]),
				html.Div(className='twelve columns', children=[
					dcc.Checklist(
					    id='current-models-only-checkbox',
					    options=[
					        {'label': 'Current Models Only', 'value': 'True'},
					    ],
					    value=[]
					),
				]),
				html.Div(id='six-slider-filter-div', className='twelve columns', children=[
					html.Div('full-div',className='twelve columns',style={'outline': '1px solid #794bc4'}),
					html.Div(className='four columns', style={'padding':'2%'},
							 #style={'width': '30%', 'padding': '10px', 'float':'left'},
							 children=[
						html.Div(
							children=[
								html.Label(children=['Head Size (in)',
									dcc.RangeSlider(
										id='head-size-slider',
										min=df['Head Size (in)'].min(),
										max=df['Head Size (in)'].max(),
										marks={i: '{}'.format(i) for i in range(int(df['Head Size (in)'].min()), int(df['Head Size (in)'].max())+1, 5)},
										step=0.5,
										tooltip = { 'always_visible': False },
										value=[df['Head Size (in)'].min(), df['Head Size (in)'].max()]
									),
								]),
								
							]
						),
						html.Br(),
						html.Div(
							children=[
								html.Label(children=['Strung Weight (oz)',
									dcc.RangeSlider(
										id='strung-weight-slider',
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
					]),

					html.Div(className='four columns', style={'padding':'2%'},
							 children=[
						html.Div(
							children=[
								html.Label(children=['Balance (pts HL/HH)',
									dcc.RangeSlider(
										id='balance-slider',
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
						html.Br(),
						html.Div(
							children=[
								html.Label(children=['Beam Width (avg. mm)',
									dcc.RangeSlider(
										id='beam-width-slider',
										min=df['Beam Width (avg. mm)'].min(),
										max=df['Beam Width (avg. mm)'].max(),
										marks={i: '{}'.format(i) for i in range(int(df['Beam Width (avg. mm)'].min()), int(df['Beam Width (avg. mm)'].max())+1, 2)},
										step=0.5,
										tooltip = { 'always_visible': False },
										value=[df['Beam Width (avg. mm)'].min(), df['Beam Width (avg. mm)'].max()]
									)
								]),
							]
						),
					]),
					
					html.Div(className='four columns', style={'padding':'2%'},
							 children=[
						html.Div(
							children=[
								html.Label(children=['String Density (X/in.²)',
									dcc.RangeSlider(
										id='string-density-slider',
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
						html.Br(),
						html.Div(
							children=[
								html.Label(children=['Swingweight',
									dcc.RangeSlider(
										id='swingweight-slider',
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
					]),	
				]),				
			]), 
		]),
	]),
	html.Br(),
	html.Br(),
	html.Div(className='three columns', id='settings', style={'outline': '1px solid #794bc4','margin-top':'20px','padding':'10px'}, children=[
		dcc.Tabs(id="tabs", value='tab-1', parent_className='custom-tabs', className='custom-tabs-container', children=[
	        dcc.Tab(label='Choose X/Y Axes', value='tab-1', className='custom-tab',
            		selected_className='custom-tab--selected',
            		children=[
	        	html.Label('X-Axis'),
				dcc.Dropdown(
					id='x-axis-dropdown',
					className='custom-dropdown',
					style={'background-color':colors['white'], 'color':colors['black'], 'border-radius':'4px'},
					options=[{'label': 'Head Size (in)', 'value': 'Head Size (in)'},
							 {'label': 'Strung Weight (oz)', 'value': 'Strung Weight (oz)'},
							 {'label': 'Beam Width (avg. mm)', 'value': 'Beam Width (avg. mm)'}],
					value='Head Size (in)'
				),
				html.Br(),  
				html.Label('Y-Axis'),
				dcc.Dropdown(
					id='y-axis-dropdown',
					className='custom-dropdown',
					style={'background-color':colors['white'], 'color':colors['black'], 'border-radius':'4px'},
					options=[{'label': 'Head Size (in)', 'value': 'Head Size (in)'},
							 {'label': 'Strung Weight (oz)', 'value': 'Strung Weight (oz)'},
							 {'label': 'Beam Width (avg. mm)', 'value': 'Beam Width (avg. mm)'}],
					value='Strung Weight (oz)'
				),
				html.Br(),  
				dcc.Checklist(
					    id='jitter-checkbox',
					    options=[{'label': 'Jitter', 'value': 'True'}],
					    value=[]
				),
	        ]),
	        dcc.Tab(label='PCA', value='tab-2',className='custom-tab',
            		selected_className='custom-tab--selected'),

	    ]),
		html.Br(),  
		html.Label('Color'),
	    dcc.Dropdown(
			id='color-dropdown',
			style={'background-color':colors['white'], 'color':colors['black'], 'border-radius':'4px'},
			className='custom-dropdown',
			options=[{'label': 'Head Size (in)', 'value': 'Head Size (in)'},
					 {'label': 'Strung Weight (oz)', 'value': 'Strung Weight (oz)'},
					 {'label': 'Beam Width (avg. mm)', 'value': 'Beam Width (avg. mm)'},
					 {'label': 'Current/Old Models', 'value': 'Current/Old Models'},
					 {'label': 'String Density (intersections / sq. in.)', 'value': 'String Density (intersections / sq. in.)'}],
			value='Beam Width (avg. mm)'
		),
	]),
	html.Div(className='nine columns', id='graph-div', style={'outline': '1px solid #794bc4','margin-top':'20px', 'margin-left':'20px'}, children=[
		dcc.Graph(
	        id='racquetspace_graph',
	    ),
	]),  
])

@app.callback(
	Output('mfr-dropdown', 'value'),
	[Input('select-all-btn', 'n_clicks')])
def select_all_mfrs(select_all):
	print(select_all)
	return sorted(list(df['mfr'].unique()))

@app.callback(
    Output('racquetspace_graph', 'figure'),
    [Input('mfr-dropdown', 'value'),
     Input('head-size-slider', 'value'),
     Input('strung-weight-slider', 'value'),
     Input('balance-slider', 'value'),
     Input('beam-width-slider', 'value'),
     Input('string-density-slider', 'value'),
     Input('jitter-checkbox', 'value'),
     Input('current-models-only-checkbox', 'value'),
     Input('x-axis-dropdown', 'value'),
     Input('y-axis-dropdown', 'value'),
     Input('color-dropdown', 'value')])
def update_figure(mfrs, head_size_range, strung_weight_range,
				  balance_range, beam_width_range, string_density_range, jitter, 
				  current_models_only, x_axis, y_axis, color):

    current_model_states = ['Current Model'] if current_models_only else \
    					   ['Current Model', 'Old Model']

    filtered_df = df[(df['mfr'].isin(mfrs)) & \
    				 (df['Current/Old Models'].isin(current_model_states)) & \
    				 (df['Head Size (in)'] >= head_size_range[0]) & \
    				 (df['Head Size (in)'] <= head_size_range[1]) & \
    				 (df['Strung Weight (oz)'] >= strung_weight_range[0]) & \
    				 (df['Strung Weight (oz)'] <= strung_weight_range[1]) & \
    				 (df['Balance (pts)'] >= balance_range[0]) & \
    				 (df['Balance (pts)'] <= balance_range[1]) & \
    				 (df['Beam Width (avg. mm)'] >= beam_width_range[0]) & \
    				 (df['Beam Width (avg. mm)'] <= beam_width_range[1]) & \
    				 (df['String Density (intersections / sq. in.)'] >= string_density_range[0]) & \
    				 (df['String Density (intersections / sq. in.)'] <= string_density_range[1])]

    # if PCA, then use those axes
    # elif jitter, then use jittered versions
    # else use regular versions...
    x_col = (x_axis + ' jittered') if jitter else x_axis
    y_col = (y_axis + ' jittered') if jitter else y_axis

    # Calculate base PCA columns if first time calling the function

    # Calculate dynamic PCA columns if ***checkbox enabled***
    fig = px.scatter(filtered_df, 
				 x=x_col, y=y_col, 
                 color=color, hover_name='name',
                 hover_data=['Head Size (in)',
                 			 'Strung Weight (oz)',
                 			 'Balance (pts)', 
                 			 'Beam Width (avg. mm)',
                 			 'url', 
                 			 ],
                 opacity=1.0,
                 color_continuous_scale=[colors['pink'], colors['blue']],
                 color_discrete_sequence=[colors['pink'], colors['blue']])#, 
                 # log_x=True, size_max=60) "Beam Width (avg. mm)"

    fig.update_traces(marker_size=5)
    fig.update_layout(plot_bgcolor=colors['gray'], paper_bgcolor=colors['black'], font_color=colors['white'])

    # fig = px.scatter(filtered_df, x="gdpPercap", y="lifeExp", 
    #                  size="pop", color="continent", hover_name="country", 
    #                  log_x=True, size_max=55)

    fig.update_layout(transition_duration=1000)

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)