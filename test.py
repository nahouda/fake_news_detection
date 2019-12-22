import pickle
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import punkt
from nltk.corpus.reader import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
from bs4 import BeautifulSoup
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import dash_renderer
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import re
import seaborn as sns
import string
nltk.download('wordnet')
### Importing inputs
# load from disk
svc_model = pickle.load(open("svc_model.sav", 'rb'))
tfidf_vectorizer = pickle.load(open("tfidf_vectorizer.sav", 'rb'))
category_codes = {
    'Real': 0,
    'Fake': 1
}
### Web Scraping Functions
import tweepy
def get_tweets():
    #Twitter API credentials
    consumer_key = "tiVIII7j3O9bPAbRgdJWmoJWd"
    consumer_secret = "SR5KZuyzFgXkXvd0tADD8qFDfYzQ0mXBaTTvgDq0SagnHlWLEU"
    access_token = "490394724-yTbJ770PfbqAqdiejo8IfEEKkIKTTtmTNodZI0ru"
    access_token_secret = "3qouPTCCnp6t1tfV6NjcXbJ4ulQ4wJ5WQt9QZzwURuTGo"
    
    OAUTH_KEYS = {'consumer_key':consumer_key, 'consumer_secret':consumer_secret,
    'access_token_key':access_token, 'access_token_secret':access_token_secret}
    auth = tweepy.OAuthHandler(OAUTH_KEYS['consumer_key'], OAUTH_KEYS['consumer_secret'])
    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

    # Request
    search = tweepy.Cursor(api.search, q='Trump').items(60)

    # Creation des listes pour chaque tweet
    sn = []
    text = []
    timestamp =[]
    for tweet in search:
        sn.append(tweet.user.screen_name)
        text.append(tweet.text)
        
    # df_features
    df_features = pd.DataFrame(
         {'tweets': text 
        })
    # df_show_info

    df_show_info = pd.DataFrame(

        {'User Screen Name': sn
        })
    
    return (df_features,df_show_info)
### Feature Engineering
# Downloading the stop words list
nltk.download('stopwords')

# Loading the stop words in english
stopword = nltk.corpus.stopwords.words('english')

def cleaning_df(df):
    
    def remove_punct(text):
      text  = "".join([char for char in text if char not in string.punctuation])
      return text

    def clean_text(text):
      txt = re.sub("[( ' )( ')(' )]", ' ', text)
      txt=re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) | (\w +:\ / \ / \S +)", " ", txt)
      return txt.lower()

    def remove_stopwords(text):
      text  = " ".join([word for word in text.split(" ") if word not in stopword])
      return text

    df['new_tweets'] = df['tweets'].apply(lambda x: remove_punct(str(x)))
    df['new_tweets'] = df['new_tweets'].apply(lambda x: clean_text(str(x)))
    df['new_tweets'] = df['new_tweets'].apply(lambda x: remove_stopwords(str(x)))
    df.dropna()
    return df
def get_category_name(category_id):
    for category, id_ in category_codes.items():    
        if id_ == category_id:
            return category
###Prediction Functions

def predict_from_features(features):
        
    predictions_pre = svc_model.predict(features)

    predictions = []

    for cat in predictions_pre:
           predictions.append(cat)

    categories = [get_category_name(x) for x in predictions]
    
    return categories

def complete_df(df, categories):
    df['Prediction'] = categories
    return df
### Dash App
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Stylesheet

app = dash.Dash(__name__)

# Colors
colors = {
    'background': '#ECECEC',  
    'text': '#696969',
    'titles': '#599ACF',
    'blocks': '#F7F7F7',
    'graph_background': '#F7F7F7',
    'banner': '#C3DCF2'

}
#HTML. For writing blocks of text, you can use the Markdown component in the dash_core_components library.
# Markdown text
markdown_text1 = '''

This application gathers tweets, predicts their category between **Real**, **Fake** and then shows a summary.

The scraped tweets are converted into a numeric feature vector with *TF-IDF vectorization*.

Then, a *Support Vector Classifier* is applied to predict each category. 

Finally, the results are visualized in a graph and a pie chart.

Please press the **Scrape** button.

'''
markdown_text2 = '''
 Created by **SBOUI Nihed** and **ABCHA Mariem**
'''
#The layout of a Dash app describes what the app looks like. It is a hierarchical tree of components.
#The dash_html_components library provides classes for all of the HTML tags and the keyword arguments describe the HTML attributes like style, className, and id.
#html.Div([ section ]) applies CSS to section of page
app.layout = html.Div(style={'backgroundColor':colors['background']}, children=[
    #html.H1(‘text’) heading (level 1)
    # Space before title
   
    # Title 
    html.Div(
        [
            html.H1(children='Tweets Classification App',
                    style={"margin-bottom": "0px"}
                   ),
            html.H4(children='A Machine Learning based app')
        ],
        style={
            'textAlign': 'center',
            'color': colors['text'],
            #'padding': '0px',
            'backgroundColor': colors['background']
              },
        #The class key is renamed as className
        className='banner',
            ),
    

    # Space after title
    html.H1(children=' ',
            style={'padding': '1px'}),
    
    

    # Text boxes
    html.Div(
        [
            html.Div(
                [
                    html.H6(children='What does this app do?',
                            style={'color':colors['titles']}),
                    
                    html.Div(
                        #The dcc library generates higher-level components like controls and graphs.
                        #dash_core_components,
                        [dcc.Markdown(children=markdown_text1),],
                        style={'font-size': '12px',
                               'color': colors['text']}),
                                        
                    html.Div(
                        [
                            dcc.Dropdown(
                                options=[
                                    {'label': 'Tweets', 'value': 'EPE'},
                                        ],
                                value=['EPE'],
                                multi=True,
                                id='checklist'),
                            
                        ],
                        style={'font-size': '12px',
                               'margin-top': '25px'}),
              
            html.Div([

                        html.Button('Scrape', 

                                    id='submit', 

                                    type='submit', 

                                    style={'color': colors['blocks'],

                                           'background-color': colors['titles'],

                                           'border': 'None'})],

                        style={'textAlign': 'center',

                               'padding': '20px',

                               "margin-bottom": "0px",

                               'color': colors['titles']}),

            
                    #A Loading component that wraps any other component and displays a spinner until the wrapped component has rendered.
                    #children: Array that holds components to render
                    #id:The ID of this component, used to identify dash components in callbacks. 
                    #type:Property that determines which spinner to show
                    dcc.Loading(id="loading-1", children=[html.Div(id="loading-output-1")], type="circle"),]),

                   
            html.Div(
                [
                    html.H6("Graphic summary",
                            style={'color': colors['titles']}),

                    html.Div([
                         dcc.Graph(id='graph1', style={'height': '300px'})
                         ],
                         style={'backgroundColor': colors['blocks'],
                                'padding': '20px'}
                    ),
                    
                    html.Div([
                         dcc.Graph(id='graph2', style={'height': '300px'})
                         ],
                         style={'backgroundColor': colors['blocks'],
                                'padding': '20px'}
                    )
                ],
                     style={'backgroundColor': colors['blocks'],
                            'padding': '20px',
                            'border-radius': '5px',
                            'box-shadow': '1px 1px 1px #9D9D9D'},
                     className='one-half column')

        ],
        className="row flex-display",
        style={'padding': '20px',
               'margin-bottom': '0px'}
    ),
    # Space
    html.H1(id='space2', children=' '),
    # Final paragraph
    html.Div(
            [dcc.Markdown(children=markdown_text2),],
            style={'textAlign': 'center','font-size': '35px',
                   'color': colors['text']}),
    # Hidden div inside the app that stores the intermediate value
    html.Div(id='intermediate-value', style={'display': 'none'})
   

])

#Connecting Components with Callbacks
@app.callback(
     [

    Output('intermediate-value', 'children'),

    Output('loading-1', 'children')

    ],

    [Input('submit', 'n_clicks')],

    [State('checklist', 'value')])
    
    
def scrape_and_predict(n_clicks, values):

    df_features = get_tweets()[0]
    df_show_info = get_tweets()[1]
                                           
    
    # Create features
    df_features = cleaning_df(df_features)
    features = tfidf_vectorizer.transform(df_features['new_tweets']).toarray()
    # Predict
    predictions = predict_from_features(features)
    # Put into dataset
    df = complete_df(df_show_info, predictions)
  
    
    return df.to_json(date_format='iso', orient='split'), ' '


@app.callback(
    Output('graph1', 'figure'),
    [Input('intermediate-value', 'children')])
def update_barchart(jsonified_df):
    print(jsonified_df)
    df = pd.read_json(jsonified_df, orient='split')
    print(df.head())
    #df.reset_index(level=0, inplace=True)
    
     # Create a summary df
    df_sum = df['Prediction'].value_counts()
    # Create x and y arrays for the bar plot
    x = ['Real', 'Fake']
    y = [[df_sum['Real'] if 'Real' in df_sum.index else 0][0],
         [df_sum['Fake'] if 'Fake' in df_sum.index else 0][0]]
        
    
    # Create plotly figure
    figure = {
        'data': [
            {'x': x, 'y': y, 'type': 'bar', 'name': 'Tweets', 'marker': {'color':['rgb(62, 137, 195)',
                                   'rgb(167, 203, 232)',
                                   ]}},
        ],
        'layout': {
            'title': 'Tweets Classification',
            'plot_bgcolor': colors['graph_background'],
            'paper_bgcolor': colors['graph_background'],
            'font': {
                    'color': colors['text'],
                    'size': '10'
            },
            'barmode': 'stack'
            
        }   
    }

    return figure

@app.callback(
    Output('graph2', 'figure'),
    [Input('intermediate-value', 'children')])
def update_piechart(jsonified_df):
    
    df = pd.read_json(jsonified_df, orient='split')
    #df = pd.DataFrame.from_dict(jsonified_df, orient='index')
    #df.reset_index(level=0, inplace=True)
    # Create a summary df
    df_sum = df['Prediction'].value_counts()

    # Create x and y arrays for the bar plot
    x = ['Real', 'Fake']
    y = [[df_sum['Real'] if 'Real' in df_sum.index else 0][0],
         [df_sum['Fake'] if 'Fake' in df_sum.index else 0][0]]
    
    # Create plotly figure
    figure = {
        'data': [
            {'values': y,
             'labels': x, 
             'type': 'pie',
             'hole': .4,
             'name': '% of tweets',
             'marker': {'colors': ['rgb(62, 137, 195)',
                                   'rgb(167, 203, 232)',
                                   ]},

            }
        ],
        
        'layout': {
            'title': 'Tweets',
            'plot_bgcolor': colors['graph_background'],
            'paper_bgcolor': colors['graph_background'],
            'font': {
                    'color': colors['text'],
                    'size': '10'
            }
        }
        
    }
    
    return figure
    
           
# Loading CSS
app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})
app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/brPBPO.css"})
app.run_server(debug=False)