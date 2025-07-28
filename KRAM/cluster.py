import hdbscan

import numpy as np 
import pandas as pd

import plotly.express as px

def map_dict(**args):
    dct = {key:value for key, value in args.items()}
    return dct

params_clu = {
    2 : map_dict(
                min_cluster_size=9,
                cluster_selection_epsilon = 0.150,
                min_samples = 5, 
                cluster_selection_method='eom',
                prediction_data=True),
    #2 : map_dict(min_cluster_size=20, cluster_selection_epsilon = 0.07, min_samples = 12, cluster_selection_method='eom', prediction_data=True),

    3 : map_dict(min_cluster_size=6,
                 cluster_selection_epsilon = 0.06,
                 #min_samples = 6,
                 cluster_selection_method='eom',
                 prediction_data=True),
    #3 : map_dict(min_cluster_size=10, cluster_selection_epsilon = 0.08, prediction_data=True),
    #3 : map_dict(min_cluster_size=8, cluster_selection_epsilon = 0.05, min_samples = 2, cluster_selection_method='eom', prediction_data=True),

    #4 : map_dict(min_cluster_size=6, cluster_selection_epsilon = 0.06,min_samples = 6, cluster_selection_method='eom',prediction_data=True),
    #...
         }

#Z = rad_tract(X_umap2D,1.8, 3)
#Z = rad_tractN(X_umapND,std =1.6, p = 1)
def cluster_hdbscanND(X_umapND, N_c = 3, custom={} ,return_obj = False, print_prop=False):
    Z = X_umapND

    if len(custom) == 0:
        db = hdbscan.HDBSCAN(**params_clu[N_c]).fit(Z)
    else:
        params_clu = custom
        db = hdbscan.HDBSCAN(**params_clu[N_c]).fit(Z)
    
    labels = db.labels_
    prob = db.probabilities_
    
    if print_prop:    
        # Number of clusters in labels, ignoring noise if present.
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        print("Estimated number of clusters: %d" % n_clusters)
        print("Estimated number of noise points: %d" % n_noise)
        #print(np.unique(labels3D, return_counts = True))
    return (labels,prob) if not return_obj else db

def build_id(OBJC_desc):
    ID = [str(i) for i in range(len(OBJC_desc))]
    return ID
    
def run_attach(DF, lbl, Desc):
    DF['cluster'] = lbl.astype(str)
    DF['id'] = build_id(Desc)
    DF['item'] = Desc
    #DF['code'] = code
    DF['size'] = .25*np.ones(len(DF['id']))
    return DF
    
def print_cluster(X_reduceND, labels, Desc, N_c = 3):
    if N_c == 2:
        DF = pd.DataFrame(X_reduceND, columns = ['x','y'])
        DF = run_attach(DF, labels, Desc)
        fig = px.scatter(DF, x='x', y='y',
                            opacity=0.5,
                            color='cluster',
                            #text  = 'id',
                            hover_data={
                                'id':':.2f', # customize hover for column of y attribute
                                'item': True,
                                        }
                           )
        fig.update_layout(
        width=1100,  # Width in pixels
        height=600  # Height in pixels
        )
        fig.update_traces(marker_size = 4)
        fig.show(renderer = 'colab')
        
    elif N_c == 3:
        DF = pd.DataFrame(X_reduceND, columns = ['x','y','z'])
        DF = run_attach(DF, labels, Desc)
        fig = px.scatter_3d(DF, x='x', y='y', z='z',
                            opacity=1.0,
                            color='cluster',
                            #text  = 'id',
                            hover_data={#'species':False, # remove species from hover data
                                        'id':':.2f', # customize hover for column of y attribute
                                        'item': True,
                                        #'suppl_2': (':.3f', np.random.random(len(df)))
                                        }
                           )
        fig.update_traces(marker_size = 2)
        #fig.add_trace(px.scatter_3d(DF[DF['code'] == 'rct'], x='x', y='y', z='z', size = 'size',opacity=.5).data[0])
        fig.show(renderer = 'colab')
        