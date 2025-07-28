# import umap
import umap.umap_ as umap
import numpy as np 
import pandas as pd

import plotly.express as px

def map_dict(**args):
    dct = {key:value for key, value in args.items()}
    return dct

params_red = {
    2 : map_dict(n_components=2, min_dist = .050, n_neighbors=20, random_state=42, n_jobs = 1),
    #2 : map_dict(n_components=2, min_dist = .005, n_neighbors=10, random_state=42, n_jobs = 1),
    
    3 : map_dict(n_components=3, min_dist = 0.05, n_neighbors=15, random_state=42, n_jobs = 1),
    #3 : map_dict(n_components=3, min_dist = 0.08, n_neighbors=40, random_state=42, n_jobs = 1),
    
    #4 : map_dict(n_components=3, min_dist = 0.05, n_neighbors=15, random_state=42, n_jobs = 1),
    #...
         }

custom = px.colors.qualitative.Alphabet

def reduce_umapND(X_full, N_r = 3, custom = {}):
    if len(custom) == 0:
        reducerND = umap.UMAP(**params_red[N_r])
    else:
        reducerND = umap.UMAP(**custom[N_r])
        
    X_umapND = reducerND.fit_transform(X_full)
    return X_umapND

def build_id(OBJC_desc):
    ID = [str(i) for i in range(len(OBJC_desc))]
    return ID
    
def run_attach_simple(DF, Desc):
    # DF['code'] = (code if special else None)
    DF['id'] = build_id(Desc)
    return DF

    
def print_reduce(X_umapND, Desc, N_r = 3):
    if N_r == 2:
        DF = pd.DataFrame(X_umapND, columns = ['x','y'])
        #DF = pd.DataFrame(rad_tract(X_umap2D,1.8, 1.01), columns = ['x','y'])
        DF = run_attach_simple(DF,Desc)
        fig = px.scatter(DF, x='x', y='y',
                            opacity=.4,
                            #color='code',
                            #text  = 'id',
                            color_discrete_sequence=custom,
                            hover_data={'id':':.2f', # customize hover for column of y attribute
                                        }
                           )
        fig.update_layout(
        width=1100,  # Width in pixels
        height=600  # Height in pixels
        )
        fig.update_traces(marker_size =5)
        #fig.write_html('results/2D_dim_red.html'
        fig.show(renderer = 'colab')
    
    elif N_r ==3:
        DF = pd.DataFrame(X_umapND, columns = ['x','y','z'])
        #DF = pd.DataFrame(rad_tractN(X_umap3D,std = 1.5, p = 1.5), columns = ['x','y','z'])
        DF = run_attach_simple(DF,Desc)
        fig = px.scatter_3d(DF, x='x', y='y', z='z',
                            opacity=.50,
                            #color='code',
                            #text  = 'id',
                            color_discrete_sequence=custom,
                            hover_data={
                                        'id':':.2f', # customize hover for column of y attribute
                                        }
                           )
        fig.update_traces(marker_size = 1.5
                         )
        #fig.write_html('results/3D_dim_red.html')
        fig.show(renderer = 'colab')

