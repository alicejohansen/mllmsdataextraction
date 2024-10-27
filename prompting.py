# imports
from openai import AzureOpenAI
import os
import pandas as pd
import io
import json
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")
load_dotenv()
import base64

client = AzureOpenAI(
    api_key=os.getenv('OPENAI_KEY'),
    api_version=os.getenv('OPENAI_API_VERSION'),
    azure_endpoint=os.getenv('OPENAI_ENDPOINT')
)


JSON_RESPONSE = {
    "x": "[1, 2, 3, 4, 5, ...] the x values to be extracted from the chart, which are integers.",
    "y": "[An equivalent quantity of integer values] to be extracted from the chart, the y values from the chart."
}

def convert_image_to_base64(image_data):
    """
    Convert an image to a base64 encoded string.
    Args:
        image_data (PIL.Image.Image): The image to be converted.
    Returns:
        str: The base64 encoded string of the image.
    """
    buffered = io.BytesIO()
    image_data.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def make_api_call(client, prompt):
    """
    Makes an API call to the chat completion endpoint using the provided client and prompt.
    Args:
        client (object): The client object used to make the API call.
        prompt (list): A list of messages to be sent as the prompt for the chat completion.
    Returns:
        dict: The response from the API call in JSON format.
    """
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=prompt,
        response_format={ "type": "json_object" }
    )
    return response


def create_prompt(img_str, quantity):
    """
    Generates a prompt for a chart-to-data assistant to extract data points from a given image.
    Args:
        img_str (str): A base64 encoded string representing the image of the chart.
        quantity (int): The number of data points to be extracted from the chart.
    Returns:
        list: A list of dictionaries representing the prompt for the assistant, including the system role, user role, and the image URL.
    """

    return [
        {"role": "system", "content": f"You are a chart-to-data assistant. Please read the chart and extract the original data points from the graph. There are exactly {quantity} data points."},
        {"role": "user", "content": [
            {"type": "text", "text": f"Provide the {quantity} data points in the following JSON format strictly adhering to the JSON specification:  {{{JSON_RESPONSE}}}. The response must only contain valid JSON. Do not include any additional text or explanations. Report the y values at ones digit precision. "},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}}
        ]}
    ]

def process_image(client, image_data, quantity):
    """
    Processes an image by converting it to a base64 string, creating a prompt, 
    and making an API call with the prompt.
    Args:
        client: The API client to use for making the API call.
        image_data: The image data to be processed.
        quantity: An integer representing the quantity to be used in the prompt.
    Returns:
        The response from the API call.
    """

    img_str = convert_image_to_base64(image_data)
    prompt = create_prompt(img_str, quantity)
    response = make_api_call(client, prompt)
    return response


def save_response_to_file(response, file_path, file_name):
    """
    Saves the content of a response message to a text file.
    Parameters:
    response (object): The response object containing the message to be saved.
    file_path (str): The directory path where the file will be saved.
    file_name (str): The name of the file (without the .png extension) where the message will be saved.
    """

    response_message = response.choices[0].message.content
    os.makedirs(file_path, exist_ok=True)
    file_name = file_name.replace(".png", "")
    save_txt_url = os.path.join(file_path, file_name)

    with open(save_txt_url, 'w') as file:
        file.write(response_message)

    print(f"Response saved to {save_txt_url}")


def save_json_to_file(json_file_path, image_file_name, response_message):
    """
    Saves a JSON response message to a file.
    Args:
        json_file_path (str): The directory path where the JSON file will be saved.
        image_file_name (str): The name of the image file, used to generate the JSON file name.
        response_message (str): The JSON response message to be saved.
    Raises:
        Exception: If there is an error parsing the JSON response message.
    Side Effects:
        - Creates the directory specified by `json_file_path` if it does not exist.
        - Writes the parsed JSON data to a file in the specified directory.
        - Prints messages indicating the success or failure of the operation.
    """
    os.makedirs(json_file_path, exist_ok=True)
    json_file_name = image_file_name.replace(".png", ".json")
    json_file_path = os.path.join(json_file_path, json_file_name)

    try:
      data = json.loads(response_message)
      print(data)
      with open(json_file_path, 'w') as json_file:
          json.dump(data, json_file, indent=4)

      print(f"JSON data saved to {json_file_path} for image {image_file_name}")

    except:
      print(f"Error parsing JSON data for image {image_file_name}")


def load_result_dfs_from_json(json_file_path):
    """
    Load result dataframes from JSON files in a specified directory.
    This function reads all JSON files in the given directory, extracts 'x' and 'y' values from each file,
    and creates a pandas DataFrame for each file. The DataFrame also includes a 'data_freq' column, which
    is derived from the filename. The function returns a dictionary where the keys are filenames and the
    values are the corresponding DataFrames.
    Args:
        json_file_path (str): The path to the directory containing the JSON files.
    Returns:
        dict: A dictionary where keys are filenames and values are pandas DataFrames containing the data
            from the JSON files.
    """
    
    results_df = {}

    for filename in os.listdir(json_file_path):
        if filename.endswith(".json"):
            filepath = os.path.join(json_file_path, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)
            x_values = data['x']
            y_values = data['y']
            data_freq = int(filename.split('_')[0])
            df = pd.DataFrame({'x': x_values, 'y': y_values})
            df['data_freq'] = data_freq
            print(df)
            results_df[filename] = df
    return results_df


def run_experiment(data_freq, image, response_file_path, title, json_file_path):
    """
    Runs an experiment by processing an image and saving the response and metadata to files.
    Args:
        data_freq (str): The frequency of the data being processed.
        image (str): The path to the image file to be processed.
        response_file_path (str): The path where the response file will be saved.
        title (str): The title to be used for naming the output files.
        json_file_path (str): The path where the JSON file will be saved.
    Returns:
        None
    """
    
    image_name = f'{data_freq}_{title}.png'
    response = process_image(client, image, data_freq)
    save_response_to_file(response, response_file_path, f'{image_name}.txt')
    save_json_to_file(json_file_path, image_name, response.choices[0].message.content, response_file_path)



def extract_json_from_response(json_file_path, image_file_name, response_message, save_txt_url):
    """
    Extracts JSON data from a response message and saves it to a specified file path.

    Args:
        json_file_path (str): The directory path where the JSON file will be saved.
        image_file_name (str): The name of the image file, used to generate the JSON file name.
        response_message (str): The response message containing JSON data as a string.
        save_txt_url (str): The URL where the text file will be saved (not used in the function).

    Returns:
        dict or None: The parsed JSON data as a dictionary if successful, otherwise None.

    Raises:
        Exception: If there is an error parsing the JSON data.
    """

    os.makedirs(json_file_path, exist_ok=True)

    json_file_name = image_file_name.replace(".png", ".json")
    json_file_path = os.path.join(json_file_path, json_file_name)

    try:
      data = json.loads(response_message)
      print(data)
      with open(json_file_path, 'w') as json_file:
          json.dump(data, json_file, indent=4)

      print(f"JSON data saved to {json_file_path} for image {image_file_name}")
      return data
    except:
      print(f"Error parsing JSON data for image {image_file_name}")
      return None
