import pandas as pd
import io
import base64
import math
import difflib
from pdf2image import convert_from_path

import time
import os
import logging
from datetime import datetime

import logging

def create_page_dict(df):
    # Initialize an empty dictionary to store the results
    page_dict = {}

    # Iterate through the DataFrame rows
    for _, row in df.iterrows():
        page_id = row['page_id']
        article_id = row['id']

        # Convert the bounding_box string to a dictionary if it's not already
        if isinstance(row['bounding_box'], str):
            bounding_box = ast.literal_eval(row['bounding_box'])
        else:
            bounding_box = row['bounding_box']

        # If the page_id is not in the dictionary, add it
        if page_id not in page_dict:
            page_dict[page_id] = {}

        # Add the article_id and bounding_box to the page's dictionary
        page_dict[page_id][article_id] = bounding_box

    return page_dict

def scale_bbox(bbox, original_size, new_size):
    '''
    Scale the bounding box from the original image size to the new image size.

    :param bbox: List of [x1, y1, x2, y2] coordinates of the bounding box
    :param original_size: Tuple of (width, height) of the original image
    :param new_size: Tuple of (width, height) of the new image
    :return: Scaled bounding box coordinates
    '''
    original_width, original_height = original_size
    new_width, new_height = new_size

    # Calculate scale factors for width and height
    width_scale = new_width / original_width
    height_scale = new_height / original_height

    # Scale the bounding box coordinates
    x1, y1, x2, y2 = bbox
    new_x1 = int(x1 * width_scale)
    new_y1 = int(y1 * height_scale)
    new_x2 = int(x2 * width_scale)
    new_y2 = int(y2 * height_scale)

    return [new_x1, new_y1, new_x2, new_y2]


def crop_and_encode_image(page, x0, y0, x1, y1):
    cropped_image = page.crop((x0, y0, x1, y1))
    buffered = io.BytesIO()
    cropped_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def split_tall_box(page, x0, y0, x1, y1, max_height, overlap):
    height = y1 - y0
    num_boxes = math.ceil((height - overlap) / (max_height - overlap))
    
    split_images = []
    for i in range(num_boxes):
        box_y0 = y0 + i * (max_height - overlap)
        box_y1 = min(y1, box_y0 + max_height)
        split_images.append(crop_and_encode_image(page, x0, box_y0, x1, box_y1))
    
    return split_images

def process_bounding_box(page, key, coords, original_size, page_size):
    x0, y0, x1, y1 = scale_bbox([coords["x0"], coords["y0"], coords["x1"], coords["y1"]],
                                original_size, page_size)
    
    # Ensure the coordinates are within the image bounds
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(page.width, x1), min(page.height, y1)
    
    width = x1 - x0
    height = y1 - y0
    
    if height > 1.5 * width:
        max_height = int(1.5 * width)
        overlap = int(0.1 * width)
        images = split_tall_box(page, x0, y0, x1, y1, max_height, overlap)
        return {key: images}
    else:
        image = crop_and_encode_image(page, x0, y0, x1, y1)
        return {key: [image]}

def crop_and_encode_images(page, bounding_boxes, original_size, page_size):
    result = {}
    for key, coords in bounding_boxes.items():
        result.update(process_bounding_box(page, key, coords, original_size, page_size))
    return result




def knit_strings(s1: str, s2: str) -> str:
    """
    Knit two strings together based on their longest common substring.

    This function finds the longest common substring between s1 and s2,
    then combines them by appending the non-overlapping part of s2 to s1.
    If no common substring is found, it simply concatenates the two strings.


    Args:
        s1 (str): The first string.
        s2 (str): The second string.

    Returns:
        str: A new string combining s1 and s2, with the overlapping part appearing only once.

    Example:
        >>> knit_strings("Hello world", "world of Python")
        'Hello world of Python'
    """
    # Create a SequenceMatcher object to compare the two strings
    matcher = difflib.SequenceMatcher(None, s1, s2)

    # Find the longest matching substring
    # match.a: start index of match in s1
    # match.b: start index of match in s2
    # match.size: length of the match
    match = matcher.find_longest_match(0, len(s1), 0, len(s2))
    
    # If no match is found (match.size == 0), simply concatenate the strings
    if match.size == 0:
        return s1 + s2

    # Start with the entire first string
    result = s1

    # Append the part of s2 that comes after the overlapping portion
    # match.size gives us the length of the overlap, so we slice s2 from that index
    result += s2[match.size:]
    
    return result


def knit_string_list(content_list: list) -> str:
    """
    Knit a list of strings together based on their longest common substrings.

    This function iteratively applies the knit_strings function to a list of strings,
    combining them based on their longest common substrings.

    Args:
        content_list (list): A list of strings to be knitted together.

    Returns:
        str: A new string combining all strings in the list.

    Example:
        >>> knit_string_list(["Hello world", "world of Python", "Python is great"])
        'Hello world of Python is great'
    """
    if not content_list:
        return ""
    
    result = content_list[0]
    for i in range(1, len(content_list)):
        result = knit_strings(result, content_list[i])
    
    return result


def process_image_with_api(image_base64, client, model="pixtral-12b-2409"):
    try:
        chat_response = client.chat.complete(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Attached is a jpeg taken from a scan of an English 19th century newspaper. The jpeg may contain, text, lists, tables, or images, newspaper. Please extract the information from it appropriately. In the case of an image please return the alt text of a couple of sentences. Do not add any additional comment or chat."
                        },
                        {
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    ]
                }
            ]
        )

        content = chat_response.choices[0].message.content
        usage = chat_response.usage
        usage = (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens)
        return content, usage

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None, None

def process_page(row, image_drive, page_dict, client, save_folder, dataset_df, log_df):
    start_time = time.time()
    
    try:
        file_path = os.path.join(image_drive, row['folder_path'], row['file_name'])
        all_pages = convert_from_path(file_path, dpi=300)
        
        page = all_pages[row['page_number'] - 1].copy()
        bounding_boxes = page_dict[str(row['page_id'])]
        
        cropped_images = crop_and_encode_images(
            page,
            bounding_boxes,
            (row['width'], row['height']),
            page.size
        )
        
        process_articles(cropped_images, client, save_folder, dataset_df)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Log successful processing
        new_row = pd.DataFrame({
            'page_id': [row['page_id']],
            'status': ['Success'],
            'processing_time': [processing_time],
            'timestamp': [pd.Timestamp.now()]
        })
        log_df = pd.concat([log_df, new_row], ignore_index=True)
        
        print(f"Successfully processed page {row['page_id']} in {processing_time:.2f} seconds.")
        
    except Exception as e:
        end_time = time.time()
        processing_time = end_time - start_time
        
        logging.error(f"Error processing page {row['page_id']}: {str(e)}")
        print(f"Error processing page {row['page_id']}. See log for details.")
        
        # Log failed processing
        new_row = pd.DataFrame({
            'page_id': [row['page_id']],
            'status': ['Failed'],
            'processing_time': [processing_time],
            'timestamp': [pd.Timestamp.now()],
            'error_message': [str(e)]
        })
        log_df = pd.concat([log_df, new_row], ignore_index=True)
    
    return log_df


def process_articles(cropped_images, client, save_folder, dataset_df):
    for article_id, image_string in cropped_images.items():
        try:
            content_list, usage_list = extract_text_from_images(image_string, client)
            full_string = knit_string_list(content_list)
            save_article_text(article_id, full_string, save_folder, dataset_df)
            print(f"Processed article {article_id}")
        except Exception as e:
            logging.error(f"Error processing article {article_id}: {str(e)}")
            print(f"Error processing article {article_id}. See log for details.")

def extract_text_from_images(image_string, client):
    content_list = []
    usage_list = []
    for i, image in enumerate(image_string):
        try:
            content, usage = process_image_with_api(image, client, model="pixtral-12b-2409")
            if content is not None:
                content_list.append(content)
                usage_list.append(usage)
        except Exception as e:
            logging.error(f"Error processing image {i}: {str(e)}")
    return content_list, usage_list

def save_article_text(article_id, full_string, save_folder, dataset_df):
    try:
        file_name = dataset_df.loc[dataset_df['id'] == int(article_id), 'save_name'].iloc[0]
        with open(os.path.join(save_folder, file_name), "w") as file:
            file.write(full_string)
    except Exception as e:
        logging.error(f"Error saving article {article_id}: {str(e)}")