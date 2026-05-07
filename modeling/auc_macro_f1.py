import os
import json
import csv
import argparse
from pathlib import Path

def process_metrics(directory_path, output_csv):
    results = []
    
    # Use a set to dynamically keep track of all unique sources found across all files
    all_sources = set()

    # Recursively find all files matching *metrics.json
    print(f"Scanning directory: {directory_path} ...")
    
    # Track if we actually find any files to give a better error message if we don't
    files_found = 0
    
    for filepath in Path(directory_path).rglob('*metrics.json'):
        path_str = str(filepath.resolve())
        if os.name == 'nt' and not path_str.startswith('\\\\?\\'):
            path_str = '\\\\?\\' + path_str
        files_found += 1
        with open(path_str, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse JSON in {filepath}. Skipping.")
                continue

            # Safely grab the model name (defaults to 'Unknown' if not found)
            model_name = data.get("model_name", ["Unknown"])[0]
            row = {"Model Name": model_name}

            # Traverse into the 'per_source_gsd_metrics'
            source_metrics = data.get("per_source_gsd_metrics", {})
            for source, gsds in source_metrics.items():
                f1_scores = []
                
                # Iterate through all GSD entries (e.g. '0.3087...', '0.8', etc.)
                for gsd, metrics_data in gsds.items():
                    try:
                        # Extract the nested macro F1 score
                        macro_f1 = metrics_data["metrics"]["F1"]["macro"]
                        f1_scores.append(macro_f1)
                    except KeyError:
                        # Skip if the path metrics -> F1 -> macro doesn't exist for this GSD
                        continue
                
                # Compute the average if we successfully grabbed F1 scores
                if f1_scores:
                    avg_f1 = sum(f1_scores) / len(f1_scores)
                    
                    # Create a safe column header name (e.g. "Crewed Aircraft" -> "Crewed_Aircraft_macro_f1_avg")
                    safe_source = source.replace(" ", "_")
                    col_name = f"{safe_source}_macro_f1_avg"
                    
                    row[col_name] = avg_f1
                    all_sources.add(col_name)

            results.append(row)

    if files_found == 0:
        print(f"No files matching '*metrics.json' were found in {directory_path}.")
        return

    # Prepare CSV fieldnames: 'Model Name' followed by all unique source columns sorted alphabetically
    fieldnames = ["Model Name"] + sorted(list(all_sources))

    # Write the compiled results out to a CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        # Using writerows alongside DictWriter handles missing keys dynamically by leaving the cell blank
        writer.writerows(results)

    print(f"Successfully processed {len(results)} files.")
    print(f"Output saved to {output_csv}")

if __name__ == "__main__":
    # Set up argparse for command line arguments
    parser = argparse.ArgumentParser(description="Parse ML model metrics from JSON files and compute average macro F1 scores per source.")
    
    # Positional argument for the target directory
    parser.add_argument(
        "directory", 
        type=str, 
        help="Path to the folder containing the evaluation runs."
    )
    
    # Optional flag for the output CSV file name/path
    parser.add_argument(
        "-o", "--output", 
        type=str, 
        default="aggregated_model_metrics.csv", 
        help="Path to the output CSV file (default: aggregated_model_metrics.csv)."
    )

    args = parser.parse_args()
    
    # Run the processor with the parsed arguments
    process_metrics(args.directory, args.output)