import marimo

__generated_with = "0.9.1"
app = marimo.App(width="medium")


@app.cell
def __():
    import json
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Load the JSON data
    with open('data/classification_results.json', 'r') as file:
        data = json.load(file)


    def process_item(key, value):
        # Handle IPTC classes
        iptc_classes = value.get('iptc', {}).get('class', [])
        if not isinstance(iptc_classes, list):
            iptc_classes = [iptc_classes] if iptc_classes != 99 else []
        
        # Handle genre class
        genre_class = value.get('genre', {}).get('class', 99)
        if isinstance(genre_class, list):
            genre_class = genre_class[0] if genre_class else 99
        
        # Create a dictionary with the key and genre
        item_dict = {'key': key, 'genre': genre_class}
        
        # Add columns for IPTC, filling with 0 if not present
        for i in range(1, 17):
            item_dict[f'iptc_{i}'] = 1 if i in iptc_classes else 0
        
        return item_dict

    # Process all items in the JSON
    processed_data = [process_item(key, value) for key, value in data.items()]

    # Create the DataFrame
    df = pd.DataFrame(processed_data)

    # Set the 'key' column as the index
    df.set_index('key', inplace=True)

    # Print the first few rows of the DataFrame
    print(df.head())
    return data, df, file, json, pd, plt, process_item, processed_data, sns


@app.cell
def __(df, sns):
    sns.histplot(data = df.loc[df['genre']!=99], x = 'genre')
    return


if __name__ == "__main__":
    app.run()
