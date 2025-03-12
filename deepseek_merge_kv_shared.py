import safetensors.torch
import torch
import json
import os

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_0_path", type=str, required=True)
    parser.add_argument("--model_1_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--num_shards", type=int, required=True)
    args = parser.parse_args()
    
    os.makedirs(args.output_path, exist_ok=True)
    # number of shards
    # policy: merge w2 from model_1 to model_0
    for i in range(args.num_shards):
        print(f'Merging model {i}...')
        model_0 = safetensors.torch.load_file(f'{args.model_0_path}/model{i}-mp4.safetensors')
        model_1 = safetensors.torch.load_file(f'{args.model_1_path}/model{i}-mp4.safetensors')
        for key in model_1.keys():
            if 'w2' in key or 'wq' in key or 'wk' in key or 'wo' in key or 'shared' in key:
                print(f'Merging {key}...')
                model_0[key] = model_1[key]
        safetensors.torch.save_file(model_0, f'{args.output_path}/model{i}-mp4.safetensors')
    
        model_0_config = json.load(open(f'{args.model_0_path}/config.json'))
        model_1_config = json.load(open(f'{args.model_1_path}/config.json'))
        for key in model_1_config['quantization_config']['config_for_layers'].keys():
            if 'w2' in key or 'wq' in key or 'wk' in key or 'wo' in key or 'shared' in key:
                model_0_config['quantization_config']['config_for_layers'][key] = model_1_config['quantization_config']['config_for_layers'][key]
        with open(f'{args.output_path}/config.json', 'w') as f:
            json.dump(model_0_config, f)
