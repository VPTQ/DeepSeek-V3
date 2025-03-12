import safetensors
import argparse
import os
import json
import glob
import shutil
from safetensors.torch import save_file, load_file

# Merge split safetensor files back into their original form
# This is the reverse operation of split_safetensor_folder.py
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing split safetensor shards")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save merged safetensors")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Find all subdirectories in the input directory
    # Each subdirectory corresponds to one original safetensor file
    subdirs = [d for d in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, d))]
    
    if not subdirs:
        print(f"No subdirectories found in {args.input_dir}")
        exit(1)
    
    print(f"Found {len(subdirs)} safetensor groups to merge")
    
    # Process each subdirectory
    for subdir in subdirs:
        subdir_path = os.path.join(args.input_dir, subdir)
        output_file = os.path.join(args.output_dir, f"{subdir}.safetensors")
        
        print(f"\nProcessing {subdir_path}...")
        print(f"Output will be saved to {output_file}")
        
        # Look for the index file
        index_path = os.path.join(subdir_path, "model.safetensors.index.json")
        if not os.path.exists(index_path):
            print(f"Error: Index file not found at {index_path}")
            continue
        
        # Load the index file
        with open(index_path, "r") as f:
            index = json.load(f)
        
        weight_map = index.get("weight_map", {})
        if not weight_map:
            print(f"Error: No weight map found in index file {index_path}")
            continue
        
        # Create a dictionary to store all tensors
        merged_tensors = {}
        
        # Track unique shard files to avoid loading the same file multiple times
        shard_files = set(weight_map.values())
        
        # Load each shard and extract tensors
        for shard_file in shard_files:
            shard_path = os.path.join(subdir_path, shard_file)
            if not os.path.exists(shard_path):
                print(f"Error: Shard file not found at {shard_path}")
                continue
            
            print(f"Loading shard {shard_file}...")
            try:
                shard_tensors = load_file(shard_path)
            except Exception as e:
                print(f"Error loading {shard_path}: {e}")
                continue
            
            # Add tensors from this shard to the merged dictionary
            for key, tensor in shard_tensors.items():
                merged_tensors[key] = tensor
        
        # Verify that all tensors in the weight map were loaded
        missing_tensors = set(weight_map.keys()) - set(merged_tensors.keys())
        if missing_tensors:
            print(f"Warning: {len(missing_tensors)} tensors from the weight map were not found in the shards")
            print(f"First few missing tensors: {list(missing_tensors)[:5]}")
        
        # Save the merged tensors to a single file
        print(f"Saving merged file with {len(merged_tensors)} tensors to {output_file}")
        try:
            save_file(merged_tensors, output_file)
            print(f"Successfully merged shards into {output_file}")
        except Exception as e:
            print(f"Error saving merged file {output_file}: {e}")
    
    print("\nAll files processed successfully!")
    
    # Copy all JSON configuration files from input directory to output directory
    print("\nCopying JSON configuration files...")
    json_files = glob.glob(os.path.join(args.input_dir, "*.json"))
    
    if json_files:
        for json_file in json_files:
            json_filename = os.path.basename(json_file)
            output_json_path = os.path.join(args.output_dir, json_filename)
            print(f"Copying {json_file} to {output_json_path}")
            try:
                shutil.copy2(json_file, output_json_path)
            except Exception as e:
                print(f"Error copying {json_file}: {e}")
        print(f"Successfully copied {len(json_files)} JSON configuration files")
    else:
        print("No JSON configuration files found in the input directory")
    
    print("\nAll operations completed successfully!") 