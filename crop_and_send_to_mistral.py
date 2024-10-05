import marimo

__generated_with = "0.9.1"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import pandas as pd
    import os

    from helper_functions import (create_page_dict, scale_bbox, crop_and_encode_images, 
    knit_strings, process_image_with_api, knit_string_list)
    import io
    from pdf2image import convert_from_path
    from PIL import Image, ImageDraw, ImageFont
    import matplotlib.pyplot as plt
    import json
    from dotenv import load_dotenv

    from mistralai import Mistral
    load_dotenv()
    image_drive = '/media/jonno/ncse'

    input_file = 'data/page_dict.json'
    with open(input_file, 'r') as f:
        page_dict = json.load(f)

    dataset_df = pd.read_parquet('data/example_set_1858-1860.parquet')
    return (
        Image,
        ImageDraw,
        ImageFont,
        Mistral,
        convert_from_path,
        create_page_dict,
        crop_and_encode_images,
        dataset_df,
        f,
        image_drive,
        input_file,
        io,
        json,
        knit_string_list,
        knit_strings,
        load_dotenv,
        mo,
        os,
        page_dict,
        pd,
        plt,
        process_image_with_api,
        scale_bbox,
    )


@app.cell
def __(dataset_df):
    target_pages_issues = dataset_df.copy().loc[:, 
    ['issue_id', 'page_id', 'page_number', 'file_name', 'folder_path', 'width', 'height']].drop_duplicates().reset_index(drop=True)

    print(f"Number of issues to extract {len(target_pages_issues[['issue_id']].drop_duplicates())}, number of pages {len(target_pages_issues[['page_id']].drop_duplicates())},")
    return (target_pages_issues,)


@app.cell
def __():
    #load issue
    return


@app.cell
def __(convert_from_path, image_drive, os, target_pages_issues):
    #select row
    check_row_df = target_pages_issues.loc[300, :]

    _file = os.path.join(image_drive, check_row_df['folder_path'], check_row_df['file_name'])

    #load all pages from an issue
    #relatively slow
    all_pages = convert_from_path(_file, dpi = 300)
    return all_pages, check_row_df


@app.cell
def __(mo):
    mo.md(r"""## Plot 1 page from the issue""")
    return


@app.cell
def __(all_pages, check_row_df, crop_and_encode_images, page_dict):
    # Use the function
    _page = all_pages[check_row_df['page_number']-1].copy()
    _bounding_boxes = page_dict[str(check_row_df['page_id'])]

    cropped_images = crop_and_encode_images(
        _page,
        _bounding_boxes,
        (check_row_df['width'], check_row_df['height']),
        _page.size
    )

    # Now 'cropped_images' is a list of PIL Image objects, each representing a cropped region
    return (cropped_images,)


@app.cell
def __(Mistral, os):
    api_key = os.environ["MISTRAL_API_KEY"]
    model = "pixtral-12b-2409"

    client = Mistral(api_key=api_key)
    return api_key, client, model


@app.cell
def __(client, cropped_images, process_image_with_api):
    image_string = cropped_images[0]

    content_list = []
    usage_list = []
    for image in image_string:

        _content, _usage = process_image_with_api(image, client, model="pixtral-12b-2409")
        content_list.append(_content)
        usage_list.append(_usage)
    return content_list, image, image_string, usage_list


@app.cell
def __(content_list, knit_string_list):
    full_string = knit_string_list(content_list)


    with open("data/filename.txt", "w") as file:
        file.write(full_string)
    return file, full_string


@app.cell
def __(full_string):
    full_string
    return


@app.cell
def __(base64, io, math, scale_bbox):
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
    return (
        crop_and_encode_image,
        crop_and_encode_images,
        process_bounding_box,
        split_tall_box,
    )


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
