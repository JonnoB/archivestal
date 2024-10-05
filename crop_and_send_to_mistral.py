import marimo

__generated_with = "0.9.1"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import pandas as pd
    import os
    from helper_functions import create_page_dict, scale_bbox

    from pdf2image import convert_from_path
    from PIL import Image, ImageDraw
    import matplotlib.pyplot as plt
    import io
    import json

    image_drive = '/media/jonno/ncse'

    input_file = 'data/page_dict.json'
    with open(input_file, 'r') as f:
        page_dict = json.load(f)

    dataset_df = pd.read_parquet('data/example_set_1858-1860.parquet')
    return (
        Image,
        ImageDraw,
        convert_from_path,
        create_page_dict,
        dataset_df,
        f,
        image_drive,
        input_file,
        io,
        json,
        mo,
        os,
        page_dict,
        pd,
        plt,
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

    check_row_df = target_pages_issues.loc[300, :]

    _file = os.path.join(image_drive, check_row_df['folder_path'], check_row_df['file_name'])

    #load all pages from an issue
    #relatively slow
    all_pages = convert_from_path(_file, dpi = 300)
    return all_pages, check_row_df


@app.cell
def __():
    return


@app.cell
def __(mo):
    mo.md(r"""## Plot 1 page from the issue""")
    return


@app.cell
def __(
    Image,
    ImageDraw,
    all_pages,
    check_row_df,
    io,
    page_dict,
    plt,
    scale_bbox,
):
    _page = all_pages[check_row_df['page_number']-1].copy()
    _draw = ImageDraw.Draw(_page)

    #N.B the dictionary loads the issues and article ids as string 
    _bounding_boxes = page_dict[str(check_row_df['page_id'])]

    # Draw rectangles for each bounding box
    for _box_id, _coords in _bounding_boxes.items():
        _x0, _y0, _x1, _y1 = scale_bbox([_coords["x0"], _coords["y0"], _coords["x1"], _coords["y1"]],
                                         (check_row_df['width'], check_row_df['height']), _page.size)
        _draw.rectangle([_x0, _y0, _x1, _y1], outline="red", width=2)

    _buf = io.BytesIO()
    _page.save(_buf, format='PNG')
    _buf.seek(0)

    # Display the image in the notebook
    plt.figure(figsize=(15, 20))
    plt.imshow(Image.open(_buf))
    plt.axis('off')
    plt.show()
    return


@app.cell
def __(all_pages, check_row_df, page_dict, scale_bbox):

    def crop_images_from_bounding_boxes(page, bounding_boxes, original_size, page_size):
        cropped_images = []
        
        for box_id, coords in bounding_boxes.items():
            x0, y0, x1, y1 = scale_bbox([coords["x0"], coords["y0"], coords["x1"], coords["y1"]],
                                        original_size, page_size)
            
            # Ensure the coordinates are within the image bounds
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(page.width, x1), min(page.height, y1)
            
            # Crop the image
            cropped_image = page.crop((x0, y0, x1, y1))
            cropped_images.append(cropped_image)
        
        return cropped_images

    # Use the function
    _page = all_pages[check_row_df['page_number']-1].copy()
    _bounding_boxes = page_dict[str(check_row_df['page_id'])]

    cropped_images = crop_images_from_bounding_boxes(
        _page,
        _bounding_boxes,
        (check_row_df['width'], check_row_df['height']),
        _page.size
    )

    # Now 'cropped_images' is a list of PIL Image objects, each representing a cropped region
    return crop_images_from_bounding_boxes, cropped_images


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
