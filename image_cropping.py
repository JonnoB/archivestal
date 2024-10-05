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
    return Image, ImageDraw, convert_from_path, io, mo, os, pd, plt


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
def __(os, pd):
    all_bounding_boxes = []
    for _file in os.listdir("data/new_parquet"):
        _temp = pd.read_parquet(os.path.join("data/new_parquet",_file))
        all_bounding_boxes.append(_temp.loc[:, ['id', 'bounding_box', 'article_type_id', 'issue_id', 'page_id', 'publication_id', 'page_number', 'pdf']])

    all_bounding_boxes = pd.concat(all_bounding_boxes, ignore_index=True)

    all_bounding_boxes['bounding_box'] = all_bounding_boxes['bounding_box'].apply(lambda box: {k: int(v) for k, v in box.items()})

    all_bounding_boxes['file_name']  = all_bounding_boxes['pdf'].str.extract('(\w{3}-\d{4}-\d{2}-\d{2})')

    all_bounding_boxes['file_name'] = all_bounding_boxes['file_name'] +'.pdf'

    all_bounding_boxes['date'] = pd.to_datetime(all_bounding_boxes['file_name'].str.extract(r'-(\d{4}-\d{2}-\d{2})\.pdf')[0], format='%Y-%m-%d')
    return (all_bounding_boxes,)


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
    #/media/jonno/ncse
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
def __(all_bounding_boxes):
    mask_1850_1852 = all_bounding_boxes['date'].between('1850-01-01', '1852-12-31')

    # Create a mask for dates between 1858-1860 (inclusive)
    mask_1858_1860 = all_bounding_boxes['date'].between('1858-01-01', '1860-12-31')

    # Combine the masks using the OR operator
    combined_mask = mask_1850_1852 | mask_1858_1860

    # Combine the masks using the OR operator
    combined_mask = mask_1858_1860


    # Apply the combined mask to the dataframe
    subset_df = all_bounding_boxes[combined_mask]

    print(f"Rows in dataset:len(subset_df['page_id'].unique())")
    subset_df
    return combined_mask, mask_1850_1852, mask_1858_1860, subset_df


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
