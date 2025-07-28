import numpy as np 
import pandas as pd
from tqdm import tqdm

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential, 
    wait_fixed
)

from .respond import Respond

def clst2id(labels, cluster, num, ID, Prob=None, r=1.0):
    """
    Selects a subset of IDs from a specific cluster.

    Args:
        labels (numpy.ndarray): Cluster labels for each data point.
        cluster (int): Index of the target cluster.
        num (int or str): Number of IDs to select or 'auto' for automatic selection.
        ID (numpy.ndarray): Array of IDs corresponding to data points.
        Prob (numpy.ndarray, optional): Probability weights for each data point. Defaults to None.
        r (float, optional): Exponent for adjusting probability weights. Defaults to 1.0.

    Returns:
        list: Selected IDs from the specified cluster.
    """
    if Prob is not None:
        # Normalize probabilities for the specified cluster
        P_subset = (Prob[labels == cluster])**r
        Prob = P_subset / np.sum(P_subset)
    if len(ID[labels == cluster]) < num:
        # Adjust selection number if cluster size is smaller
        num = len(ID[labels == cluster])
    if num == 'auto':
        # Automatically determine number of items to select
        num = int(len(ID[labels == cluster]) / 3)
    # Select IDs with or without consideration of probabilities
    return list(np.random.choice(ID[labels == cluster], size=num, replace=False, p=Prob).astype(int))


def process_topics(labels, OBJC, probs, text_col='Description', num_inc=10, model='r1:8'): 
    """
    Processes descriptive topics by summarizing and tagging based on cluster labels.

    Args:
        labels (numpy.ndarray): Cluster labels for each data point.
        OBJC (pandas.DataFrame): DataFrame containing data to process.
        probs (numpy.ndarray): Probability weights for cluster membership.
        text_col (str, optional): Column name containing descriptions. Defaults to 'Description'.
        num_inc (int, optional): Number of items to include in each cluster's summary. Defaults to 10.
        model (str, optional): Model identifier to use for summarization. Defaults to 'r1:8'.

    Returns:
        pandas.DataFrame: Processed information containing clusters, counts, tags, and interpretations.
    """
    # Determine the number of clusters (excluding noise)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    # Count the unique elements in labels
    values, count = np.unique(labels, return_counts=True)
    
    # Initialize DataFrame for storing processing results
    Prt = pd.DataFrame(data=np.array([values[1:], count[1:]]).T, columns=['cluster', 'count'])
    
    # Initialize lists for interpretations and tags
    interpretations = []
    tags = []
    
    # Iterate over each cluster
    for j in tqdm(range(n_clusters), desc="Processing Topics"):
        topic = j
        # Select example descriptions from the cluster
        idz = clst2id(labels, topic, num_inc, OBJC.index, probs)
        # Summarize the selected descriptions
        summary = Summarize(OBJC[text_col][idz].to_list(), ret=True, model=model)[0]
        # Generate a tag for the summary
        tag = Label(summary, T=0.3, C=1, N=1, ret=True, model=model)[0]
        
        # Store the summary and tag results
        interpretations.append(summary)
        tags.append(tag)
    
    # Set display options for larger DataFrame outputs
    pd.set_option("display.max_colwidth", 1000)
    pd.set_option('display.max_rows', 140)

    # Populate DataFrame with the results
    Prt['Interpretation'] = interpretations
    Prt['tag'] = tags
    Prt = Prt[['cluster', 'count', 'tag', 'Interpretation']]

    return Prt

@retry(wait=wait_fixed(2))
def Summarize(items, context='', T=0.3, C=1, N=1, ret=True, model='4om'): 
    """
    Summarizes a list of items using a language model.

    Args:
        items (list): Items to summarize.
        context (str, optional): Context for the summarization. Defaults to ''.
        T (float, optional): Temperature parameter for randomness. Defaults to 0.3.
        C (float, optional): Nucleus sampling parameter. Defaults to 1.
        N (int, optional): Number of summary outputs. Defaults to 1.
        ret (bool, optional): Whether to return the summaries. Defaults to True.
        model (str, optional): Model identifier for summarization. Defaults to '4om'.

    Returns:
        list: Summarized outputs if ret is True.
    """
    try:
        # Default context for the model
        context0 = 'You are a helpful assistant that carefully and completely reads, thinks through, and executes tasks.'
        if not context:
            context = 'Summarize the following items into one short sentence highlighting uniquely common themes.'
        
        # Format the items for input to the model
        text = '```' + '``` \n```'.join(items) + '```'
        request = context
        prompt = text
    
        # Request a summary from the Respond function
        answer = Respond(request + '\n' + prompt, context=context0, t=T, c=C, model=model, n=N)
        
        if ret:
            return answer
        else:
            for a in answer:
                print(a)
                print('#############')
    except Exception as e:
        print(e)
        raise

@retry(wait=wait_fixed(2))
def Title(item, context='', T=0.3, C=1, N=1, ret=True, model='4om'):
    """
    Generates a title for a given summary using a language model.

    Args:
        item (str): Summary text from which to derive a title.
        context (str, optional): Context for title generation. Defaults to ''.
        T (float, optional): Temperature parameter for randomness. Defaults to 0.3.
        C (float, optional): Nucleus sampling parameter. Defaults to 1.
        N (int, optional): Number of titles to generate. Defaults to 1.
        ret (bool): Whether to return the titles. Defaults to True.
        model (str): Model identifier for generation. Defaults to '4om'.

    Returns:
        list: Generated title(s) if ret is True.
    """
    try:
        # Default model context
        context0 = 'You are a helpful assistant that carefully and completely reads, thinks through, and executes tasks.'
        if not context:
            context = """\
            Given a summary of documents, define a unique topic title. \
            The summary is delimited by triple backticks and should represent the documents collectively.\
            Avoid including the word "Title:" at the beginning of the response.
            """
        
        # Prepare input text for the model
        text = '```' + item + '```'
        request = context
        prompt = text
    
        # Generate titles using the Respond function
        answer = Respond(request + '\n' + prompt, context=context0, t=T, c=C, model=model, n=N)
        
        if ret:
            return answer
        else:
            for a in answer:
                print(a)
                print('#############')
    except Exception as e:
        print(e)
        raise

@retry(wait=wait_fixed(2))
def Label(item, context='', T=0.3, C=1, N=1, ret=True, model='4om'):
    """
    Produces a concise label or phrase from a provided description using a language model.

    Args:
        item (str): Description to label.
        context (str, optional): Context for labeling. Defaults to ''.
        T (float, optional): Temperature parameter for randomness. Defaults to 0.3.
        C (float, optional): Nucleus sampling parameter. Defaults to 1.
        N (int, optional): Number of labels to produce. Defaults to 1.
        ret (bool): Whether to return the label. Defaults to True.
        model (str): Model identifier for generation. Defaults to '4om'.

    Returns:
        list: Generated label(s) if ret is True.
    """
    try:
        # Model context definition
        context0 = 'You are a helpful assistant that carefully and completely reads, thinks through, and executes tasks.'
        if not context:
            context = """\
            Given a description of requested items, create a concise label or phrase. \
            The description is delimited by triple backticks; use up to 4 words. \
            Avoid including "Label:" or options, and do not use extraneous symbols or explanations.
            """
        
        # Format the input text for the model
        text = '```' + item + '```'
        request = context
        prompt = text
    
        # Request a label from the Respond function
        answer = Respond(request + '\n' + prompt, context=context0, t=T, c=C, model=model, n=N)
        
        if ret:
            return answer
        else:
            for a in answer:
                print(a)
                print('#############') 
    except Exception as e:
        print(e)
        raise
