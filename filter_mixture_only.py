#!/usr/bin/env python3
"""
Filter JSON files to keep only mixture.wav entries
"""
import json
import sys
from pathlib import Path

def filter_mixture_only(json_path):
    """Keep only mixture.wav entries in the JSON file"""
    print(f"Processing {json_path}...")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    original_count = len(data)
    
    # Keep only entries that contain 'mixture.wav'
    filtered_data = [entry for entry in data if 'mixture.wav' in entry[0]]
    
    filtered_count = len(filtered_data)
    
    print(f"  Original entries: {original_count}")
    print(f"  Filtered entries: {filtered_count}")
    print(f"  Removed entries: {original_count - filtered_count}")
    
    # Backup original file
    backup_path = json_path.with_suffix('.json.bak')
    print(f"  Creating backup at {backup_path}")
    with open(backup_path, 'w') as f:
        json.dump(data, f, indent=4)
    
    # Write filtered data
    print(f"  Writing filtered data...")
    with open(json_path, 'w') as f:
        json.dump(filtered_data, f, indent=4)
    
    print(f"  ✓ Done!\n")

if __name__ == '__main__':
    base_path = Path('/workspace/aeromamba-pytorch2/egs/musdb')
    
    json_files = [
        base_path / 'tr' / 'hr.json',
        base_path / 'tr' / 'lr.json',
        base_path / 'tt' / 'hr.json',
        base_path / 'tt' / 'lr.json',
    ]
    
    for json_file in json_files:
        if json_file.exists():
            filter_mixture_only(json_file)
        else:
            print(f"Warning: {json_file} not found, skipping...")
    
    print("All files processed successfully!")
