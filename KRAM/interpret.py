import numpy as np 
import pandas as pd

from tqdm import tqdm

from tenacity import (
    retry,
    stop_after_attempt,  # Added for limiting retry attempts
    wait_exponential,  # Changed from wait_fixed to wait_exponential
    wait_fixed
)

from .respond import Respond

# global Respond

# import sys
# sys.path.append("..") # Adds higher directory to python modules path.
# from Scripts.LLM_RAND.respond import *

def clst2id(labels, cluster, num, ID, Prob = None, r = 1.0):
    if Prob is not None:
        P_subset = (Prob[labels == cluster])**r
        Prob = P_subset/np.sum(P_subset)
    if len(ID[labels == cluster]) < num:
        num = len(ID[labels == cluster])
    if num == 'auto':
        num = int(len(ID[labels == cluster])/3)
    return list(np.random.choice(ID[labels == cluster], size=num, replace=False, p=Prob).astype(int))


def process_topics(labels, OBJC, probs, text_col = 'Description', num_inc = 10, model = 'r1:8'): 
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    values, count = np.unique(labels, return_counts = True)
    Prt = pd.DataFrame(data = np.array([values[1:],count[1:]]).T, columns = ['cluster', 'count'])
    
    interpretations = []
    titles = []
    tags = []
    for j in tqdm(range(n_clusters)):
        topic = j
        idz = clst2id(labels, topic, num_inc, OBJC.index, probs)
        summary = Summarize(OBJC[text_col][idz].to_list(), ret = True, model=model)[0]
        # title = Title(summary,T = 1.5, C = .5, N = 1, ret = True, model=model)[0]
        tag = Label(summary,T = .3, C = 1, N = 1, ret = True, model = model)[0]
        interpretations.append(summary)
        # titles.append(title)
        tags.append(tag)
        
    pd.set_option("display.max_colwidth", 1000)
    pd.set_option('display.max_rows', 140)

    Prt['count'] = count[1:]
    # Prt['Title'] = [tit.replace('"',"") for tit in titles]
    Prt['Interpretation'] = interpretations
    #Prt['keys'] = key
    Prt['tag'] = tags
    Prt = Prt[['cluster',
               'count',
               'tag',
               #'Title',
               'Interpretation']]
    
    # order = np.argsort(-Prt['count'])
    # P = Prt.iloc[order]
    # display(P)
    return Prt

#@retry(wait=wait_fixed(2))
def Summarize(items, context = '',T = .3, C = 1, N = 1, ret = True, model = '4om'): 
    #print(model)
    try:
        context0 = 'You are a helpful assistent that carefully and completely: reads, thinks through, and executes tasks.'
        #descriptions #triple
        if context == '':
            # context = 'For the following descriptions, delimited by triple backticks, summarize these into one very short paragraph highlighting commmon themes and terms.'
            context = 'For the following items that were requested, delimited by triple backticks, summarize them into one short sentence highlighting their uniquely commmon themes.'
        
        text  = '```'+'``` \n```'.join(items) +'```'
        
        request =  context
        prompt = text
    
        answer = Respond(request + '\n' + prompt, context = context0, t = T, c = C, model = model, n = N)
        
        if ret == True:
            return answer
        else:
            for a in answer:
                print(a)
                print('#############')
    except Exception as e:
        print(e)
        raise


@retry(wait=wait_fixed(2))
def Title(item, context = '',T = .3, C = 1, N = 1, ret = True, model = '4om'):
    try:
        context0 = 'You are a helpful assistent that carefully and completely: reads, thinks through, and executes tasks.'
        #descriptions #triple
        if context == '':
            context = """\
            You will be provided with a summary of documents that came from the same cluster. \
            The summary will be delimited with triple backticks. \
            Your task is to define a topic title that likley and uniquley represents the summary of the documents.\
            Do not include the word "Title:" at the beggining of your response.
            """
        
        text  = '```'+ item +'```'
        
        request =  context
        prompt = text
    
        answer = Respond(request + '\n' + prompt, context = context0, t = T, c = C, model = model, n = N)
        
        if ret == True:
            return answer
        else:
            for a in answer:
                print(a)
                print('#############')
    except Exception as e:
        print(e)
        raise

@retry(wait=wait_fixed(2))
def Label(item, context = '',T = .3, C = 1, N = 1, ret = True, model = '4om'):
    try:
        context0 = 'You are a helpful assistent that carefully and completely: reads, thinks through, and executes tasks.'
        #descriptions #triple
        if context == '':
            # context = """\
            # You will be provided with a title of a set of documents . \
            # The title will be delimited with triple backticks. \
            # Your task is to come up with a 1 to 3 word label that generally but uniquley represents the title of the documents.\
            # Do not include the word "Label:" at the beggining of your response.
            # """
            context = """\
            You will be provided with a description of a set of requested items. \
            The description sentence will be delimited by triple backticks. \
            Your task is to come up with a 1 to 4 word label or phrase that uniquley captures the description of these items.\
            Do not include the word "Label:" at the beggining of your response.
            Provide only one response, not a series of options.
            Do not add extraneous text, symbols or explainations, outisde of the required phrase.
            """
        text  = '```'+ item +'```'
        
        request =  context
        prompt = text
    
        answer = Respond(request + '\n' + prompt, context = context0, t = T, c = C, model = model, n = N)
        
        if ret == True:
            return answer
        else:
            for a in answer:
                print(a)
                print('#############') 
    except Exception as e:
        print(e)
        raise

# def Cluster(idz, dscrps, T = 2, C = .7, N = 1,ret = False, model = '4om'):
#     print('#############')
#     #Turn down c when vaiance of answers goes up
#     instruct = 'First, for the following descriptions, delimited by triple backticks, summarize these using one complete sentence by highlighting only common themes and terms, but do not create a list. Second, add a list of common terms.'
#     answer = Summarize((dscrps[idz]).to_list(),
#                           context = instruct, T = T, C = C, N = N, ret = ret, model = model)
#     for i in idz:
#         print('------------')
#         print(dscrps[i])
#     if ret == True:
#         return answer

