# TODO: create merged dataframe with column per datestamp
# TODO: Verify if datestamp.csv already exists, don't overwrite
# TODO: Make pretty plots
# TODO: Clean up (docstrings, ...)
# TODO: Create consistent data file names

import io
import pandas as pd
import requests
import string

def makeFilename(filename, replacements={}):
    """
    Strip input string down to only valid characters for a Windows filename and return.
    Replacements is an optional dictionary keys representing characters to replace by the corresponding value before stripping invalid characters.
    If the substituted value is not a valid filename character, it will be stripped anyway after the replacement. 
    """
    valid_chars = frozenset("-_.() %s%s" % (string.ascii_letters, string.digits))
    
    # TODO: do all replacements in a single iteration over filename. 
    for key, val in replacements.items():
        filename = filename.replace(key, val)

    return ''.join(c for c in filename if c in valid_chars)


def getData():
    """Scrape Corona infections per Dutch municipality from the RIVM and store data as CSV."""

    ### Collect the data
    # Download source from rivm corona page
    url = "https://www.rivm.nl/coronavirus-kaart-van-nederland"
    page_content = requests.get(url).text

    # Find start and end of csv data based on <div> tags
    tag_start, tag_end = '<div id="csvData">', '</div>'
    index_from = page_content.index(tag_start) + len(tag_start)
    index_to = page_content.index(tag_end, index_from)
    # Extract data, trim whitespace, split multi-line string to list
    csv_lines = page_content[index_from:index_to].strip().splitlines()


    ### Clean up csv formatting
    # Determine number of column headers
    num_cols = csv_lines[0].count(";") + 1
    # Remove fields without column headers
    csv_lines = [";".join(str(line).split(";")[:num_cols]) for line in csv_lines] # TODO: Just find the numCol'th ";" and splice the end off
    csv_data = "\n".join(csv_lines)


    
    # Define how to deal with NA values
    naInt = lambda x: int(x) if len(x) else 0

    # Convert csv_data to StringIO class for pd.read_table() compatibility
    csv_data = io.StringIO(csv_data)
    df = pd.read_table(csv_data, sep=";", index_col=0, dtype={"Gemeente:":str}, converters={"Gemnr": naInt, "Aantal": naInt})

    # read timestamp
    tag_start, tag_end = '<span class="content-date-edited">', "</span>"
    index_from = page_content.index(tag_start) + len(tag_start)
    index_to = page_content.index(tag_end, index_from)
    raw_timestamp = str(page_content[index_from:index_to])
    
    # Store to csv
    replacements = {":": "-"}
    df.to_csv("data/" + makeFilename(raw_timestamp, replacements) + ".csv")



# Main script execution
if __name__ == '__main__':
    getData()