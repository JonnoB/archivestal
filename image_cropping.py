import marimo

__generated_with = "0.9.1"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import pandas as pd
    import os

    from pdf2image import convert_from_path
    from PIL import Image, ImageDraw
    import matplotlib.pyplot as plt
    import io

    image_drive = '/media/jonno/ncse'
    return (
        Image,
        ImageDraw,
        convert_from_path,
        image_drive,
        io,
        mo,
        os,
        pd,
        plt,
    )


@app.cell
def __(mo):
    mo.md(
        """
        # General things

        The images are a total 48gb zipped  and 60Gb unzipped.
        """
    )
    return


@app.cell
def __(mo):
    mo.md("""# Create dataframe with bounding box""")
    return


@app.cell
def __(pd):
    page_data = pd.read_parquet('data/periodicals_page.parquet')


    return (page_data,)


@app.cell
def __(os, page_data, pd):
    all_bounding_boxes = []
    for _file in os.listdir("data/new_parquet"):
        _temp = pd.read_parquet(os.path.join("data/new_parquet",_file))
        all_bounding_boxes.append(_temp.loc[:, ['id', 'bounding_box', 'article_type_id', 'issue_id', 'page_id', 'publication_id', 'page_number', 'pdf']])

    all_bounding_boxes = pd.concat(all_bounding_boxes, ignore_index=True)

    all_bounding_boxes['bounding_box'] = all_bounding_boxes['bounding_box'].apply(lambda box: {k: int(v) for k, v in box.items()})

    all_bounding_boxes['file_name']  = all_bounding_boxes['pdf'].str.extract('(\w{3}-\d{4}-\d{2}-\d{2})')

    all_bounding_boxes['file_name'] = all_bounding_boxes['file_name'] +'.pdf'

    all_bounding_boxes['date'] = pd.to_datetime(all_bounding_boxes['file_name'].str.extract(r'-(\d{4}-\d{2}-\d{2})\.pdf')[0], format='%Y-%m-%d')

    all_bounding_boxes = all_bounding_boxes.merge(
        page_data[['id', 'height', 'width']].set_index('id'),
        left_on='page_id',
        right_index=True
    )
    return (all_bounding_boxes,)


@app.cell
def __():
    return


@app.cell
def __(all_bounding_boxes):
    len(all_bounding_boxes['issue_id'].unique())
    return


@app.cell
def __(all_bounding_boxes):
    all_bounding_boxes
    return


@app.cell
def __(mo):
    mo.md(r"""# Create bounding box dictionary""")
    return


@app.cell
def __(all_bounding_boxes, ast):
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

    # Create the dictionary
    page_dict = create_page_dict(all_bounding_boxes)
    return create_page_dict, page_dict


@app.cell
def __(page_dict, pd):
    lengths = pd.DataFrame({'len':[len(value) for key, value in page_dict.items()]})

    lengths.describe()
    return (lengths,)


@app.cell
def __(page_dict):
    page_dict.keys()
    return


@app.cell
def __(page_dict):
    page_dict[96368]
    return


@app.cell
def __(all_bounding_boxes):
    page_issue_df = all_bounding_boxes.groupby(['page_id', 'page_number','issue_id', 'pdf']).size().reset_index(name = 'counts')

    page_issue_df
    return (page_issue_df,)


@app.cell
def __(all_bounding_boxes):
    all_bounding_boxes.groupby(['issue_id', 'pdf']).size().reset_index(name = 'counts')
    return


@app.cell
def __(mo):
    mo.md(
        """
        # explore a single exmaple
        vvb
        """
    )
    return


@app.cell
def __(page_issue_df):
    page_issue_df.loc[page_issue_df['issue_id']==2302]
    return


@app.cell
def __(page_dict):
    page_dict[79242]
    return


@app.cell
def __(all_bounding_boxes):
    all_bounding_boxes.groupby(['issue_id', 'pdf']).size().reset_index(name = 'counts')
    return


@app.cell
def __(pd):
    #Add the folder paths to the periodicals dataframe, this will allow us to construct paths to the PDF's

    periodicals = pd.read_parquet('data/periodicals_publication.parquet')

    periodicals['folder_path'] = [
     'Northern_Star_issue_PDF_files',
    'Leader_issue_PDF_files/Leader_issue_PDF_files',
    'Tomahawk_issue_PDF_files',
    'Publishers_Circular_issue_PDF_files',
        'English_Womans_Journal_issue_PDF_files',
    'Monthly_Repository_issue_PDF_files']


    periodicals
    return (periodicals,)


@app.cell
def __():
    return


@app.cell
def __(all_bounding_boxes):
    all_bounding_boxes.loc[(all_bounding_boxes['file_name']=='NSS-1852-10-02.pdf') & 
    (all_bounding_boxes['page_number']==14),:]
    return


@app.cell
def __(page_dict):
    page_dict[175540]
    return


@app.cell
def __():
    def scale_bbox(bbox, original_dpi, new_dpi):
        '''
        The bounding boxes of the OCR are scaled for a different image size, apparently based on dpi.
        This conversion function allows the crunching of data
        '''
        scale_factor = new_dpi / original_dpi
        return [int(coord * scale_factor) for coord in bbox]
    return (scale_bbox,)


@app.cell
def __(ImageDraw, convert_from_path, os, page_dict, scale_bbox):
    _file = 'NSS-1852-10-02.pdf'
    #online image size is
    # 1211 * 1954
    #Note images are not all the same size
    #bounding width is 
    #1127
    pages = convert_from_path(os.path.join('data/test_pdf/batch_1', _file), dpi = 300)

    page = pages[0]
    draw = ImageDraw.Draw(page)

    # Your bounding box dictionary
    bounding_boxes = page_dict[175540]
    # Draw rectangles for each bounding box
    for box_id, coords in bounding_boxes.items():
        x0, y0, x1, y1 = scale_bbox([coords["x0"], coords["y0"], coords["x1"], coords["y1"]], 72, 300)
    #    x0, y0, x1, y1 = coords["x0"], coords["y0"], coords["x1"], coords["y1"]
        draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
    return bounding_boxes, box_id, coords, draw, page, pages, x0, x1, y0, y1


@app.cell
def __(Image, io, page, plt):
    buf = io.BytesIO()
    page.save(buf, format='PNG')
    buf.seek(0)

    # Display the image in the notebook
    plt.figure(figsize=(15, 20))
    plt.imshow(Image.open(buf))
    plt.axis('off')
    plt.show()
    return (buf,)


@app.cell
def __(page):
    width, height = page.size

    print(f"Image dimensions: {width} x {height} pixels")
    return height, width


@app.cell
def __(Image):
    def get_dpi(filename):
        with Image.open(filename) as img:
            dpi = img.info.get('dpi')
            if dpi:
                return dpi
            else:
                return None

    # Usage
    image_path = 'data/science_and_art.png'
    dpi = get_dpi(image_path)
    if dpi:
        print(f"The DPI of the image is: {dpi}")
    else:
        print("DPI information not found in the image.")
    return dpi, get_dpi, image_path


@app.cell
def __(scale_bbox):
    # Example usage
    original_bbox = [300, 300, 600, 600]  # for 300 DPI
    scaled_bbox = scale_bbox(original_bbox, 300, 100)  # scaling to 100 DPI
    print(scaled_bbox)  # Should print something close to [100, 100, 200, 200]
    return original_bbox, scaled_bbox


@app.cell
def __(mo):
    mo.md(
        r"""
        # Create subset

        There is too much data. I am creating a subset of only the overlapping periods to allow a compare and contrast
        """
    )
    return


@app.cell
def __(all_bounding_boxes, periodicals):
    mask_1850_1852 = all_bounding_boxes['date'].between('1850-01-01', '1852-12-31')

    # Create a mask for dates between 1858-1860 (inclusive)
    mask_1858_1860 = all_bounding_boxes['date'].between('1858-01-01', '1860-12-31')

    # Combine the masks using the OR operator
    combined_mask = mask_1850_1852 | mask_1858_1860

    # Combine the masks using the OR operator
    combined_mask = mask_1858_1860


    # Apply the combined mask to the dataframe
    subset_df = all_bounding_boxes[combined_mask]

    subset_df = subset_df.merge(periodicals[['id','folder_path']], left_on='publication_id', right_on='id')

    print(f"Rows in dataset:{len(subset_df['page_id'].unique())}")
    subset_df
    return combined_mask, mask_1850_1852, mask_1858_1860, subset_df


@app.cell
def __(subset_df):
    target_pages_isues = subset_df.copy().loc[:, 
    ['issue_id', 'page_id', 'page_number', 'file_name', 'folder_path', 'width', 'height']].drop_duplicates().reset_index(drop=True)


    print(f"Number of issues to extract {len(target_pages_isues[['issue_id']].drop_duplicates())}, number of pages {len(target_pages_isues[['page_id']].drop_duplicates())},")
    target_pages_isues
    return (target_pages_isues,)


@app.cell
def __(convert_from_path, image_drive, os, target_pages_isues):
    check_row_df = target_pages_isues.loc[0, :]

    _file = os.path.join(image_drive, check_row_df['folder_path'], check_row_df['file_name'])

    all_pages = convert_from_path(_file, dpi = 300)
    return all_pages, check_row_df


@app.cell
def __():
    def scale_bboxes(bbox_df, target_image_size):
        # Find the maximum x1 and y1 values
        max_x1 = max(bbox['x1'] for bbox in bbox_df.values())
        max_y1 = max(bbox['y1'] for bbox in bbox_df.values())

        # Calculate scale factors
        scale_x = target_image_size[0] / max_x1
        scale_y = target_image_size[1] / max_y1

        # Create a new dictionary with scaled coordinates
        scaled_bbox_dict = {}
        for key, bbox in bbox_df.items():
            scaled_bbox_dict[key] = {
                'x0': int(bbox['x0'] * scale_x),
                'x1': int(bbox['x1'] * scale_x),
                'y0': int(bbox['y0'] * scale_y),
                'y1': int(bbox['y1'] * scale_y)
            }

        return scaled_bbox_dict
    return (scale_bboxes,)


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

    print(_page.size)

    # Your bounding box dictionary
    _bounding_boxes = page_dict[check_row_df['page_id']]

    # Draw rectangles for each bounding box
    for _box_id, _coords in _bounding_boxes.items():
        _x0, _y0, _x1, _y1 = scale_bbox([_coords["x0"], _coords["y0"], _coords["x1"], _coords["y1"]], 90, 300)
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
def __(Image, ImageDraw, all_pages, check_row_df, io, page_dict, plt):
    def scale_bbox2(bbox, original_size, new_size):
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


    _page = all_pages[check_row_df['page_number']-1].copy()
    _draw = ImageDraw.Draw(_page)

    print(_page.size)

    # Your bounding box dictionary
    _bounding_boxes = page_dict[check_row_df['page_id']]

    # Draw rectangles for each bounding box
    for _box_id, _coords in _bounding_boxes.items():
        _x0, _y0, _x1, _y1 = scale_bbox2([_coords["x0"], _coords["y0"], _coords["x1"], _coords["y1"]],
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
    return (scale_bbox2,)


if __name__ == "__main__":
    app.run()
