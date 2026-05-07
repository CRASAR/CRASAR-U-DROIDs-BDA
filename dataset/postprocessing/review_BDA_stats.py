import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='review BDA stats', description='This program computes the number of spot checks made and the number of custom added polygons.')
    parser.add_argument('--fused_annotation_map', type=str, help='The fused annotations path map.')
    parser.add_argument('--gold_annotation_map', type=str, help="The gold annotations path map.")
    args = parser.parse_args()

    print("Loading Annotation File Path Mapping...")
    f = open(args.fused_annotation_map, "r")
    annotation_path_map = json.loads(f.read())
    f.close()

    print("Loading Annotation File Path Mapping...")
    f = open(args.gold_annotation_map, "r")
    gold_path_map = json.loads(f.read())
    f.close()

    custom_polygons_count = 0
    differing_polygons_count = 0
    for geotif_path, annotation_path in annotation_path_map.items():
        #Load the annotations
        print("Loading the BDA annotations from:", annotation_path)
        f = open(annotation_path, "r")
        annotations_data = json.loads(f.read())
        f.close()

        print("Loading the BDA annotations from:", gold_path_map[geotif_path])
        f = open(gold_path_map[geotif_path], "r")
        gold_annotations_data = json.loads(f.read())
        f.close()

        microsoft_polygons_1 = [p for p in annotations_data if p['source'] == 'Microsoft']
        microsoft_polygons_2 = [p for p in gold_annotations_data if p['source'] == 'Microsoft']

        custom_polygons_count += sum(1 for p in gold_annotations_data if p['source'] == 'custom')

        # Counting differences in Microsoft polygons
        differing_polygons_count += sum(1 for p1, p2 in zip(microsoft_polygons_1, microsoft_polygons_2) if p1["label"] != p2["label"])

    print(f"Number of differing polygons from source 'Microsoft': {differing_polygons_count}")
    print(f"Total number of polygons from source 'custom': {custom_polygons_count}")