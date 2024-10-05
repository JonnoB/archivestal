import pandas as pd


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
  