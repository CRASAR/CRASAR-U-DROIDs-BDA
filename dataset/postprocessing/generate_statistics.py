import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
generators_path = os.path.dirname(current)
dataset_path = os.path.dirname(generators_path)
sys.path.append(generators_path)
sys.path.append(dataset_path)

from dataset.constants import BDA_CATEGORY_COLOR_MAP
import re

def generate_statistics(save_path, json_path):
  f = open(json_path)

  data = json.load(f)
  damage_labels = {'no damage':[], 'minor damage':[], 'major damage':[],'destroyed':[], 'un-classified':[]}
  counts = []
  for x in  data['preds']:
    damage_labels[data['preds'][x]['label']].append(data['preds'][x]['confidence'])

  for damage_label in damage_labels:
    counts.append(len(damage_labels[damage_label]))

  # Plotting
  barlist = plt.bar(damage_labels.keys(), counts)
  plt.xlabel('Labels')
  plt.ylabel('Counts')
  plt.title('Distribution of Labels')
  plt.xticks(rotation=90)
  
  for i, damage_label in enumerate(damage_labels):
    barlist[i].set_color(np.array(BDA_CATEGORY_COLOR_MAP[damage_label])/255)

  
  plt.savefig(save_path + 'label_distribution.png', bbox_inches='tight')
  plt.show()

  for damage_label in damage_labels.keys():
    histo = plt.hist(damage_labels[damage_label])
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title(f'Confidence scores distribution for {re.sub(r"(_|-)+", " ", damage_label).title().replace(" ", "")}')
    plt.grid(alpha =  0.5)
    
    plt.savefig(save_path + f'confidence_scores_{damage_label}.png', bbox_inches='tight')
    plt.show()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot the metrics associated with the passed metrics files')
    parser.add_argument('--save_path', type=str, help='The path to where the plots should be saved.')
    parser.add_argument('--json_path', type=str, help='The path from which the json is to be loaded.')
    args = parser.parse_args()
    generate_statistics(args.save_path, args.json_path)