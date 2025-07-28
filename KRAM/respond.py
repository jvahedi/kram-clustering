import importlib, urllib.request
import requests

from tenacity import (
    retry,
    #stop_after_attempt,
    wait_exponential,
    wait_fixed
)
import importlib
import urllib.request
import json
import ssl
from tenacity import (
    retry,
    stop_after_attempt,  # Added for limiting retry attempts
    wait_exponential  # Changed from wait_fixed to wait_exponential
)
import regex as re

gpt_250 = 'fd141762ad904a91b170781fcb428b04'  # GPT-4 enabled

def endUrl(deployment, api,
           base='https://apigw.rand.org/openai/RAND/inference/deployments/', 
           method='/chat/completions?api-version='):
    return base + deployment + method + api

def sendRequest(url, hdr, data):
    data = json.dumps(data)
    context = ssl._create_unverified_context()
    req = urllib.request.Request(url, headers=hdr, data=bytes(data.encode("utf-8")))
    req.get_method = lambda: 'POST'
    
    try:
        response = urllib.request.urlopen(req, context=context, timeout=10)  # Added timeout
        content = bytes.decode(response.read(), 'utf-8')  # Return string value
        return json.loads(content)
    except urllib.error.URLError as e:  # Added specific error handling
        print(f"Network error: {e}")
        raise
    except Exception as e:  # General exception handling
        print(f"Unexpected error: {e}")
        raise

@retry(wait=wait_exponential(multiplier=1, min=2, max=7))#, stop=stop_after_attempt(5))  # Modified retry logic
def gptRespond(prompt, context='', t=1, c=1, model='4om', n=1, print_rslt=False):
    '''Makes text calls to RAND's internal GPT'''

    key = gpt_250
    try:
        api = '2024-06-01'  # Updated 10/15/24
        
        Deployment = {
            '3': 'gpt-35-turbo-v0125-base',
            '4': 'gpt-4-v0613-base',
            '4o': 'gpt-4o-2024-08-06',
            '4om': 'gpt-4o-mini-2024-07-18',
        }
    
        Model = {
            '3': 'gpt-35-turbo',
            '4': 'gpt-4',
            '4o': 'gpt-4o',
            '4om': 'gpt-4o-mini',
        }
        
        deployment = Deployment[model]
        model = Model[model]
        
        url = endUrl(deployment, api)
        
        hdr = {
            'Content-Type': 'application/json',
            'Cache-Control': 'no-cache',
            'Ocp-Apim-Subscription-Key': key,
        }
        
        data = {
            'model': model,
            'messages': [
                {'role': 'system', 'content': context},
                {'role': 'user', 'content': prompt}],
            'temperature': t,
            'top_p': c,
            'n': n,
        }
        
        res = sendRequest(url, hdr, data)
        
        Results = [res['choices'][i]['message']['content'] for i in range(n)]
        if print_rslt:
            for answer in Results:
                print('#---------------------------------#')
                print(answer)
        return Results
    except Exception as e:
        print(e)
        raise
#OLLAMA
@retry(wait=wait_exponential(multiplier=1, min=2, max=5))#, stop=stop_after_attempt(5))  # Modified retry logic
def olmRespond(prompt, context='', t=1, c=1, model='r1:8', n=1,
               url = f'http://127.0.0.1:11434/api/chat',
               print_rslt = False, filter_think = True):
    try:
        Model = {
            '3.2': 'llama3.2',
            'r1:8': 'deepseek-r1:8b'
        }
        
        hdr = {
            'Content-Type': 'application/json',
        }
        
        mdl = Model[model]
        data = {
            'model': mdl,
            'messages': [
                {'role': 'system', 'content': context},
                {'role': 'user', 'content': prompt}],
            'temperature': t,
            'top_p': c,
            #'n': n,
            'stream': False
        }
        Results = []
        for i in range(n):
            completion_response = requests.post(url,json=data)
            Result = completion_response.json()['message']['content']
            if filter_think : Result = think_masking(Result)
            Results.append(Result)
            if print_rslt:
                print(Result)
                print('#---------------------------------#')
        if print_rslt: print("DONE.")
        return Results
    except Exception as e:
        print(e)
        raise 

def think_masking(json_string):
    cleaned_string = re.sub(r'<think>.*?</think>', '', json_string, flags=re.DOTALL).strip()
    return cleaned_string
        
def Respond(*args, model='4om', **kwargs):
    # Mapping model types to response functions
    Model_Choice = {
        '3': 'gpt',
        '4': 'gpt',
        '4o': 'gpt',
        '4om': 'gpt',
        
        '3.2': 'olm',
        'r1:8': 'olm'
    }

    # Determine which function to use based on the model
    choice = Model_Choice.get(model)
    if choice == 'gpt':
        return gptRespond(*args, model=model, **kwargs)
    elif choice == 'olm':
        return olmRespond(*args, model=model, **kwargs)
    else:
        print("Model not recognized")
        raise ValueError("Invalid model choice")



# txt = '''
# '''
# ctxt = "How do I color my first dataframe, df, based on the values of a different dataframe, df2, of the same size?'"

# answers = gptRespond(txt, ctxt, t = 1, c = 1, model = '4', n = 2)
# for answer in answers:
#     print(answer)