import os
import urllib
import urllib.request
import argparse

from collections import defaultdict

def make_quadkey_lookup_dict(urls):
    result = defaultdict(list)
    for url in urls:
        if url.endswith("tif") or url.endswith("tiff"):
            pared_quadkey = url.split("/")[-1].split(".")[0].split("-")[0]
            result[pared_quadkey].append(url)
    return result

program_description = '''This program takes a set of tiles,
and a set of links to those tiles, and then downlaods the
json metadata from MAXAR.'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='download_maxar_tile_info', description=program_description)
    parser.add_argument('--search_geotif_folder', type=str, help="The path to the folder containing the tifs.")
    parser.add_argument('--metadata_folder', type=str, help="The path to the folder where the metadata jsons will be written.")
    parser.add_argument('--imagery_file_list', type=str, help="The path to the list of urls where the files were downloaded from.")
    args = parser.parse_args()

    with open(args.imagery_file_list, "r") as f:
        urls_in_file = f.readlines()

    clean_urls_in_file = [line.strip() for line in urls_in_file]

    quadkey_to_url = make_quadkey_lookup_dict(clean_urls_in_file)

    os.makedirs(args.metadata_folder, exist_ok=True)

    output_counts = defaultdict(lambda:0)

    for root, dirs, files in os.walk(args.search_geotif_folder):
        for file in files:
            base_file_name = file
            file_num = file.split(".")[-1]
            file_url_idx = 0
            if file_num.isnumeric():
                file_url_idx = int(file_num)
                base_file_name = ".".join(file.split(".")[:-1])
            if base_file_name.endswith("tif") or base_file_name.endswith("tiff"):
                parsed_url = None
                filename_quadkey = None
                try:
                    filename_quadkey = base_file_name.split(".")[0].split("-")[0]
                    parsed_url = quadkey_to_url[filename_quadkey][file_url_idx]
                except KeyError:
                    pass

                if parsed_url:
                    metadata_url = parsed_url.replace("-visual", "").replace(".geo", "").split(".tif")[0] + ".json"
                    output_path = os.path.join(args.metadata_folder, filename_quadkey + ".json")
                    sufix = "" if file_url_idx == 0 else "."+str(file_url_idx)
                    print("Attempting to download metata from", metadata_url)
                    try:
                        urllib.request.urlretrieve(metadata_url, output_path+sufix)
                    except urllib.error.HTTPError as e:
                        print("\tFailed to download file", e)
