import os
from mistralai import Mistral
from dotenv import load_dotenv
from strip_markdown import strip_markdown
import json
from tqdm import tqdm

from transformers import AutoTokenizer

from helper_functions_class import *

load_dotenv()

api_key = os.environ["MISTRAL_API_KEY"]
client = Mistral(api_key=api_key)
    # Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2')

files_directory = 'data/returned_text'

files = os.listdir(files_directory)


#for testing
#file_name = files[30]
#
#id_number = file_name.split('_')[1]
#
#with open(os.path.join(files_directory, file_name), 'r') as file:
#    text = file.read()

#
#stripped_text  = strip_markdown(text)
#truncated_text = truncate_to_n_tokens(stripped_text, tokenizer)


#iptc_result = classify_text_with_api(create_iptc_prompt(truncated_text), client, model="mistral-large-latest")
#genre_result = classify_text_with_api(create_genre_prompt(truncated_text), client, model="mistral-large-latest")


# Path to save the results dictionary
results_file = 'classification_results.json'

# Load existing results if the file exists
if os.path.exists(results_file):
    with open(results_file, 'r') as f:
        results_dict = json.load(f)
else:
    results_dict = {}


# Get the list of files
files = [f for f in os.listdir(files_directory) if f.endswith('.txt')]  # Adjust the file extension if needed

# Create a progress bar
for file_name in tqdm(files, desc="Processing files"):
    # Extract the ID number from the file name
    id_number = file_name.split('_')[1]
    
    # Skip if this file has already been processed
    if id_number in results_dict:
        continue
    
    # Read the file content
    with open(os.path.join(files_directory, file_name), 'r') as file:
        text = file.read()
    
    # Process the text
    stripped_text = strip_markdown(text)
    truncated_text = truncate_to_n_tokens(stripped_text, tokenizer)
    
    # Get classification results
    iptc_result = classify_text_with_api(create_iptc_prompt(truncated_text), client, model="mistral-large-latest")
    genre_result = classify_text_with_api(create_genre_prompt(truncated_text), client, model="mistral-large-latest")
    
    # Add results to the dictionary
    results_dict[id_number] = {
        'iptc': iptc_result,
        'genre': genre_result
    }
    
    # Save the updated dictionary after each iteration
    with open(results_file, 'w') as f:
        json.dump(results_dict, f)

print("Processing complete. Results saved to", results_file)