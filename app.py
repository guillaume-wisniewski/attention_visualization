import pickle

from itertools import product

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

from dash.dependencies import Input, Output, ALL

import plotly.express as px


pl_density = [[0.0, 'rgb(230,240,240)'],
              [0.1, 'rgb(187,220,228)'],
              [0.2, 'rgb(149,197,226)'],
              [0.3, 'rgb(123,173,227)'],
              [0.4, 'rgb(115,144,227)'],
              [0.5, 'rgb(119,113,213)'],
              [0.6, 'rgb(120,84,186)'],
              [0.7, 'rgb(115,57,151)'],
              [0.8, 'rgb(103,35,112)'],
              [0.9, 'rgb(82,20,69)'],
              [1.0, 'rgb(54,14,36)']]


def load_data(src_fn, self_attention_fn):
    data = []
    for src, attention_scores in zip(open(src_fn),
                                     pickle.load(open(self_attention_fn, "rb"))):
        src = src.strip()
        src = src.split() + ["<eos>"]
        scores = {layer: attention[:,:len(src),:len(src)] for layer, attention in attention_scores.items()}
        data.append((src, scores))

    layers = list(attention_scores.keys())
    n_heads = attention_scores["layer_2"].shape[0]
        
    return None, data, layers, n_heads


def generate_heatmap(src_words, scores):
    return px.imshow(scores.tolist(),
                     x=src_words,
                     y=src_words)

def generate_word_selectors(src_words, mask):
    return dcc.Checklist(options=[{"label": word, "value": idx} for idx, word in enumerate(src_words)],
                         value=mask,
                         labelStyle={'display': 'inline-block'},
                         id="words_checklist")


def generate_figure(src_words, scores, mask):
    from itertools import product
    import plotly.graph_objs as go
    from bezier import RationalBezierCurve, get_b1, dim_plus_1, get_neg_b1

    n = len(src_words)
    scores = [{"src": i, "tgt": j, "value": abs(float(scores[i, j]))} for i, j in product(range(n), repeat=2)]

    edges = [(item['src'], item['tgt'])  for item in scores]
    interact_strength = [item['value'] for item in scores]

    data = []
    
    # nodes representing words
    nodes = {"type": 'scatter',
             "x": list(range(n)),
             "y": [0] * n,
             "mode": 'markers',
             "marker": {"size": 12, 
                        "color": interact_strength, 
                        "colorscale": pl_density,
                        "showscale": False,
                        "line": dict(color='rgb(50,50,50)', width=0.75)},
             # XXX need to extract the information
             "text": [f'attention({src_words[k]}, {src_words[k]}) = ' for k in range(n)],
             "hoverinfo": 'text'}
    data.append(nodes)

    # Appexes of the circular arcs
    x_apexes, y_apexes, apexes_color = [], [], []
    #list of strings to be displayed when hovering the mouse over the middle of the circle arcs
    tooltips = [] 

    X = list(range(n)) # node x-coordinates
    nr = 75 
    for i, (j, k) in enumerate(edges):

        if j not in mask:
            continue

        if j < k:
            tooltips.append(f'interactions({src_words[j]}, {src_words[k]})={interact_strength[i]}')
        else:
            tooltips.append(f'interactions({src_words[k]}, {src_words[j]})={interact_strength[i]}')
            
        b0 = [X[j], 0.0]
        b2 = [X[k], 0.0]

        b1 = get_b1(b0, b2) if j < k else get_neg_b1(b0, b2)
        
        a = dim_plus_1([b0, b1, b2], [1, 0.5, 1])
        pts = RationalBezierCurve(a, nr)
        x, y = zip(*pts)
        
        x_apexes.append(pts[nr // 2][0]) 
        y_apexes.append(pts[nr // 2][1])
        apexes_color.append(interact_strength[i])

        data.append(dict(type='scatter',
                         x=x, 
                         y=y, 
                         name='',
                         mode='lines', 
                         line={"width": interact_strength[i] * 10,
                               "color": "#6b8aca",
                               "shape": "spline"},
                         text=tooltips,
                         hoverinfo='none'
                        )
                    )

    data.append(dict(type='scatter',
                     x=x_apexes,
                     y=y_apexes,
                     name='',
                     mode='markers',
                     marker_symbol="diamond",
                     marker={"size": 6,
                             "colorscale": pl_density,
                             "color": apexes_color},
                     text=tooltips,
                     hoverinfo='text'))
    data.append(nodes)

    layout={"font": dict(size=10), 
            "width": 900,
            "height": 460,
            "showlegend": False,
            "xaxis": dict(anchor='y',
                          showline=False,  
                          zeroline=False,
                          showgrid=False,
                          tickvals=list(range(27)), 
                          ticktext=src_words,
                          tickangle=50,
            ),
            "yaxis": dict(visible=False), 
            "hovermode": 'closest',
            "margin": dict(t=80, b=110, l=10, r=10),
    }

    return go.Figure(data=data, layout=layout)



# Load data
curr_data, data, layers, n_heads = load_data("test_metier.fr", "pred_metiers.self_attention_scores.pkl")

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
app.config.suppress_callback_exceptions = True

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

nav_items =  [dbc.NavItem(dbc.NavLink("Mean", active=True, id="head_mean"))] +\
    [dbc.NavItem(dbc.NavLink(f"Head n°{h + 1}", id=f"head_{h}", active=False)) for h in range(n_heads)]
nav_clicks = [0 for _ in nav_items]
selected_head = [n.children.active for n in nav_items]

app.layout = html.Div([
    dbc.Row(
        [
            dbc.Col(html.Button('Previous', id='prev', n_clicks=0)),
            dbc.Col(html.H3(children="Phrase n°", id="sentence_id"), width=3),
            dbc.Col(html.Button('Next', id='next', n_clicks=0))
        ],
        justify="center"
    ),
    dbc.Row(
        [
            dbc.Col(html.Div([dcc.Checklist(id="words_checklist")], id="word_selector"), width="auto"),
            dcc.Dropdown(
                options=[
                    {'label': 'layer n°0', 'value': 'layer_0'},
                    {'label': 'layer n°1', 'value': 'layer_1'},
                    {'label': 'layer n°2', 'value': 'layer_2'}
                ],
                value=["layer_0"],
                multi=True
            )
        ],
        justify="between"
    ),
    html.H4(" "),
    dbc.Nav(
        nav_items,
        pills=True
    ),
    html.H4(" "),
    html.H4("Interaction between selected words:"),
    dcc.Graph(
        id="arc_graph"
    ),
    html.H4("Attention Matrix:"),
    dcc.Graph(
        id='heatmap',
    )
])


@app.callback(
    *[Output(n.children.id, "active") for n in nav_items],
    *[Input(n.children.id, "n_clicks") for n in nav_items]
    )
def update_navbar(*values):
    global nav_clicks
    global selected_head
    
    if all(v is None for v in values):
        return [n.children.active for n in nav_items]

    selected_head = [new != old for new, old in zip(values, nav_clicks)]
    nav_clicks = values
    return selected_head

    
@app.callback(
    Output("sentence_id", "children"),
    Output("word_selector", "children"),
    Output("arc_graph", "figure"),
    Output("heatmap", "figure"),
    Input("words_checklist", "value"),
    Input("next", "n_clicks"),
    Input("prev", "n_clicks"),
    *[Input(n.children.id, "n_clicks") for n in nav_items]
)
def update_sentence_id(words_checklist,
                       next_btn,
                       prev_btn,
                       *btn):
    global curr_data
    
    ctx = dash.callback_context
    caller_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if curr_data is None:
        curr_data = 0

    mask = words_checklist if words_checklist is not None else []
    if caller_id == "words_checklist":
        mask = [v for v in ctx.triggered[0]["value"]]
    elif caller_id == "prev":
        curr_data, mask = (curr_data - 1, []) if curr_data != 0 else (0, mask)
    elif caller_id == "next":
        curr_data, mask = (curr_data + 1, []) if curr_data != len(data) - 1 else (len(data) - 1, mask)
        
    src_words, scores = data[curr_data]
    if selected_head[0]:
        scores = scores["layer_0"].mean(dim=0)
    else:
        active_head = [i for i, h in enumerate(selected_head[1:]) if h][0]
        scores = scores["layer_0"][active_head,:,:]
        
    output = [html.H3(f"Phrase n°{curr_data + 1} / {len(data)}"),
              generate_word_selectors(src_words, mask),
              generate_figure(src_words, scores, mask),
              generate_heatmap(src_words, scores)
    ]

    return output
        

if __name__ == '__main__':
    app.run_server(debug=True)
