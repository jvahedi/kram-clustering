import importlib, urllib.request
import requests
from tenacity import (
    retry,
    wait_exponential,
    stop_after_attempt,  # Added for limiting retry attempts
)
import json
import ssl
import regex as re

gpt_250 = 'XXXXXXXXXXXXXXXXXXXXXXX'  # GPT-4 enabled API key

def endUrl(deployment, api, base='https://apigw.rand.org/openai/RAND/inference/deployments/', method='/chat/completions?api-version='):
    """
    Constructs the endpoint URL for the API call.

    Args:
        deployment (str): The specific model deployment identifier.
        api (str): Version of the API to be used.
        base (str, optional): Base URL for the API. Defaults to 'https://apigw.rand.org/openai/RAND/inference/deployments/'.
        method (str, optional): Method endpoint for the API call. Defaults to '/chat/completions?api-version='.

    Returns:
        str: Full endpoint URL for the API call.
    """
    return base + deployment + method + api

def sendRequest(url, hdr, data):
    """
    Sends an HTTP POST request to the specified URL with given headers and data.

    Args:
        url (str): The URL to send the request to.
        hdr (dict): Headers to include in the request.
        data (dict): Data payload for the request.

    Returns:
        dict: The JSON response from the server.

    Raises:
        URLError: If there is a network-related error.
        Exception: For any other unexpected error.
    """
    data = json.dumps(data)  # Convert data dictionary to JSON string
    context = ssl._create_unverified_context()  # Context to allow unverified SSL certificates
    req = urllib.request.Request(url, headers=hdr, data=bytes(data.encode("utf-8")))  # Prepare the request
    req.get_method = lambda: 'POST'  # Set the HTTP method to POST
    
    try:
        # Execute the request
        response = urllib.request.urlopen(req, context=context, timeout=10)  # Set timeout for 10 seconds
        content = bytes.decode(response.read(), 'utf-8')  # Decode response content to UTF-8 string
        return json.loads(content)  # Convert string response to Python dictionary
    except urllib.error.URLError as e:  # Handle URL errors
        print(f"Network error: {e}")
        raise
    except Exception as e:  # Handle other exceptions
        print(f"Unexpected error: {e}")
        raise

@retry(wait=wait_exponential(multiplier=1, min=2, max=7))  # Retries with exponential backoff
def gptRespond(prompt, context='', t=1, c=1, model='4om', n=1, print_rslt=False):
    """
    Makes a text call to RAND's internal GPT model.

    Args:
        prompt (str): User input prompt to be processed.
        context (str, optional): The context for the prompt. Defaults to ''.
        t (float, optional): Temperature setting for response creativity. Defaults to 1.
        c (float, optional): Parameter for nucleus sampling. Defaults to 1.
        model (str, optional): Model identifier to use. Defaults to '4om'.
        n (int, optional): Number of response choices to return. Defaults to 1.
        print_rslt (bool, optional): Whether to print the result. Defaults to False.

    Returns:
        list: List of response strings from the model.
    
    Raises:
        Exception: For any errors during request processing.
    """
    key = gpt_250  # Subscription key
    try:
        # Define API version and model deployment mappings
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
        
        # Select deployment and model based on provided model identifier
        deployment = Deployment[model]
        model = Model[model]
        
        # Construct request URL
        url = endUrl(deployment, api)
        
        # Set request headers
        hdr = {
            'Content-Type': 'application/json',
            'Cache-Control': 'no-cache',
            'Ocp-Apim-Subscription-Key': key,
        }
        
        # Prepare request data payload
        data = {
            'model': model,
            'messages': [
                {'role': 'system', 'content': context},
                {'role': 'user', 'content': prompt}],
            'temperature': t,
            'top_p': c,
            'n': n,
        }
        
        # Send the request and get the response
        res = sendRequest(url, hdr, data) 
        
        # Extract the content of each response choice
        Results = [res['choices'][i]['message']['content'] for i in range(n)]
        if print_rslt:
            for answer in Results:
                print('#---------------------------------#')
                print(answer)
        return Results
    except Exception as e:
        print(e)
        raise

@retry(wait=wait_exponential(multiplier=1, min=2, max=5))  # Retry with exponential backoff
def olmRespond(prompt, context='', t=1, c=1, model='r1:8', n=1, url=f'http://127.0.0.1:11434/api/chat', print_rslt=False, filter_think=True):
    """
    Makes a call to the OLLAMA LLM model.

    Args:
        prompt (str): User input prompt to be processed.
        context (str, optional): The context for the prompt. Defaults to ''.
        t (float, optional): Temperature setting for response randomness. Defaults to 1.
        c (float, optional): Nucleus sampling parameter. Defaults to 1.
        model (str, optional): Model identifier to use. Defaults to 'r1:8'.
        n (int, optional): Number of response choices to return. Defaults to 1.
        url (str, optional): The URL endpoint for the API. Defaults to 'http://127.0.0.1:11434/api/chat'.
        print_rslt (bool, optional): Whether to print the result. Defaults to False.
        filter_think (bool, optional): Whether to filter out 'think' tags from responses. Defaults to True.

    Returns:
        list: List of response strings from the model.

    Raises:
        Exception: For any errors during request processing.
    """
    try:
        # Define model mapping
        Model = {
            '3.2': 'llama3.2',
            'r1:8': 'deepseek-r1:8b'
        }
        
        # Set request headers
        hdr = {
            'Content-Type': 'application/json',
        }
        
        # Prepare request data payload
        mdl = Model[model]
        data = {
            'model': mdl,
            'messages': [
                {'role': 'system', 'content': context},
                {'role': 'user', 'content': prompt}],
            'temperature': t,
            'top_p': c,
            'stream': False
        }
        Results = []  # Initialize results list
        for i in range(n):
            # Send the POST request and receive completion response
            completion_response = requests.post(url, json=data)
            Result = completion_response.json()['message']['content']
            if filter_think: Result = think_masking(Result)  # Apply think masking if required
            Results.append(Result)  # Append result to list
            if print_rslt:  # Print result if requested
                print(Result)
                print('#---------------------------------#')
        if print_rslt: 
            print("DONE.")
        return Results
    except Exception as e:
        print(e)
        raise 

def think_masking(json_string):
    """
    Removes <think> tags and content from a JSON-formatted string.

    Args:
        json_string (str): Input string potentially containing <think> tags.

    Returns:
        str: Cleaned string with <think> tags removed.
    """
    # Use regex to remove <think> tags and enclosed content
    cleaned_string = re.sub(r'<think>.*?</think>', '', json_string, flags=re.DOTALL).strip()
    return cleaned_string
        
def Respond(*args, model='4om', **kwargs):
    """
    Determines which response function to call based on the model type.

    Args:
        *args: Positional arguments for the response functions.
        model (str, optional): Specifies the model type. Defaults to '4om'.
        **kwargs: Additional keyword arguments for the response functions.

    Returns:
        list: The result from the chosen response function.

    Raises:
        ValueError: If an invalid model choice is provided.
    """
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
