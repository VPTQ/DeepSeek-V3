import os
import safetensors.torch
import argparse
import json
import math

import torch
from vptq.utils.pack import dtype_convert, pack_index, unpack_index_tensor

reverse_mapping = {
    "embed": 0,
    "attn_norm": None,
    "ffn_norm": None,
    "wq": 0,
    "wq_a": None,
    "q_norm": None,
    "wq_b": 0,
    "wkv_a": None,
    "kv_norm": None,
    "wkv_b": 0,
    "wo": 1,
    "gate": None,
    "w1": 0,
    "w2": 1,
    "w3": 0,
    "norm": None,
    "head": 0,
    "scale": None,
}

def get_mapping_key(key, filter_key=reverse_mapping.keys()):
    parts = key.split('.')
    # for all parts, if part in filter_key, return part
    for part in parts:
        if part in filter_key:
            return part 
    return None

def set_dim(dim):
    if dim == 0:
        return 1
    elif dim == 1:
        return 0

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Convert DeepSeek model layer names and save to a new safetensors file')
    parser.add_argument('--input-model', type=str, default="/home/aiscuser/yangwang/repo_packed_output_model/model0-mp1.safetensors",
                        help='Path to the input safetensors file')
    parser.add_argument('--input-model-config', type=str, default="/home/aiscuser/yangwang/repo_packed_output_model/repo_config.json",
                        help='Path to the input model config file')
    parser.add_argument('--output-path', type=str, default="/home/aiscuser/yangwang/repo_packed_output_model_reshard",
                        help='Path to save the output safetensors file')
    parser.add_argument('--world-size', type=int, default=4, help='Number of world size')
    parser.add_argument('--norm-dim', type=int, default=0, help='Normalize dimension')
    parser.add_argument('--vecort-dim', type=int, default=0, help='Weight bias dimension')
    parser.add_argument('--weight-scale-dim', type=int, default=1, help='Weight bias dimension')
    parser.add_argument('--num-experts', type=int, default=256, help='Number of experts')
    args = parser.parse_args()
    
    os.makedirs(args.output_path, exist_ok=True)
    
    print(f"\nLoading model from: {args.input_model}")
    
    print(f"Loading model config from: {args.input_model_config}")
    with open(args.input_model_config, 'r') as f:
        config = json.load(f)
        config = config['quantization_config']['config_for_layers']
     
    for rank in range(args.world_size):
        print(f'processing rank: {rank}')
        # Load the safetensors file
        model_data = safetensors.torch.load_file(args.input_model) 
         
        n_local_experts = args.num_experts // args.world_size
        print(f'model_data keys: {model_data.keys()}')
        keys = list(model_data.keys())
        for key in keys:
            # print(f'processing {key}')
            mapping_key = get_mapping_key(key)
            if mapping_key is None:
                continue
            dim = reverse_mapping[mapping_key]
            model_data[key] = model_data[key].to('cuda')
            ## expert parallelism
            if '.experts.' in key:
                idx = int(key.split('.')[4])
                if idx < rank * n_local_experts or idx >= (rank + 1) * n_local_experts:
                    del model_data[key]
                    print(f'remove {key}')
                    continue
                else:
                    print(f'keep {key}')
                    continue
            ## handle weight_bias and weight_scale
            elif dim is not None and 'centroids' not in key:
                # weight_bias and weight_scale are 1D tensors
                if 'weight_bias' in key or 'weight_scale' in key:
                    print(f'{key}, mapping_key: {mapping_key}, dim: {dim}, model_data[key].shape: {model_data[key].shape}')
                    shape = model_data[key].shape
                    if dim == args.weight_scale_dim:
                        assert shape[0] % args.world_size == 0
                        slice_size = shape[0] // args.world_size
                        start_idx = rank * slice_size
                        end_idx = (rank + 1) * slice_size if rank < args.world_size - 1 else shape[0]
                        model_data[key] = model_data[key][start_idx:end_idx]
                        print(f'reshard {key} from {shape} to {model_data[key].shape}, rank: {rank}, dim: {dim}')
                    else:
                        # do nothing
                        pass
                # handle tensor except indices, head, embed
                elif 'indices' not in key and 'weight_bias' not in key and 'weight_scale' not in key:
                    print(f'{key}, mapping_key: {mapping_key}, dim: {dim}, model_data[key].shape: {model_data[key].shape}')
                    shape = model_data[key].shape
                    assert shape[dim] % args.world_size == 0
                    slice_size = shape[dim] // args.world_size
                    start_idx = rank * slice_size
                    end_idx = (rank + 1) * slice_size if rank < args.world_size - 1 else shape[dim]
                    
                    if dim == 0:
                        model_data[key] = model_data[key][start_idx:end_idx, :]
                    elif dim == 1:
                        model_data[key] = model_data[key][:, start_idx:end_idx]
                    
                    model_data[key] = model_data[key] / args.world_size
                    print(f'reshard {key} from {shape} to {model_data[key].shape}, rank: {rank}, dim: {dim}')
                # packed indices
                elif 'indices' in key:
                    print(f'{key}, mapping_key: {mapping_key}, dim: {dim}, model_data[key].shape: {model_data[key].shape}')
                    indices_config = config[key.replace('.indices', '')]
                    if indices_config['is_indice_packed']:
                        indices = model_data[key]
                        # print(f'indices_config: {indices_config}')
                        num_centroids = indices_config['num_centroids'][1]
                        num_res_centroids = indices_config['num_res_centroids'][1]
                        if num_centroids > 0:
                            index_bits = math.ceil(math.log2(num_centroids))
                        else:
                            index_bits = 0
                        if num_res_centroids > 0:
                            res_bits = math.ceil(math.log2(num_res_centroids))
                        else:
                            res_bits = 0
                        group_size = indices_config['group_size']
                        unpacked_indices, unpacked_res_indices = unpack_index_tensor(
                                                        pack_tensor=indices, 
                                                        index_bits=index_bits, 
                                                        num_elements=group_size,
                                                        res_bits=res_bits,
                                                        num_res_elements=group_size,
                                                        index_dtype=torch.uint16)
                        print(f'unpacked_indices: {unpacked_indices.shape}, dtype: {unpacked_indices.dtype}, index_bits: {index_bits}, res_bits: {res_bits}')
                        unpacked_indices_shape = unpacked_indices.shape
                        assert unpacked_indices_shape[dim] % args.world_size == 0
                        slice_size = unpacked_indices_shape[dim] // args.world_size
                        start_idx = rank * slice_size
                        end_idx = (rank + 1) * slice_size if rank < args.world_size - 1 else unpacked_indices_shape[dim]
                        if dim == 0:
                            unpacked_indices = unpacked_indices[start_idx:end_idx, :]
                        elif dim == 1:
                            unpacked_indices = unpacked_indices[:, start_idx:end_idx]
                        unpacked_indices = unpacked_indices.to(torch.uint16) 
                        print(f'reshard {key} from {unpacked_indices_shape} to {unpacked_indices.shape}, rank: {rank}, dim: {dim}')
                        
                        if num_res_centroids > 0:
                            print(f'unpacked_res_indices: {unpacked_res_indices.shape}, dtype: {unpacked_res_indices.dtype}')
                            res_indices_shape = unpacked_res_indices.shape
                            assert res_indices_shape[dim] % args.world_size == 0
                            slice_size = res_indices_shape[dim] // args.world_size
                            start_idx = rank * slice_size
                            end_idx = (rank + 1) * slice_size if rank < args.world_size - 1 else res_indices_shape[dim]
                            if dim == 0:
                                unpacked_res_indices = unpacked_res_indices[start_idx:end_idx, :]
                            elif dim == 1:
                                unpacked_res_indices = unpacked_res_indices[:, start_idx:end_idx]
                            unpacked_res_indices = unpacked_res_indices.to(torch.uint16) 
                            print(f'reshard {key} from {res_indices_shape} to {unpacked_res_indices.shape}, rank: {rank}, dim: {dim}')
                        else:
                            unpacked_res_indices = None
                    else:
                        assert False
                    
                    packed_indices = pack_index(
                        indice=unpacked_indices,
                        index_bits=index_bits,
                        res_indice=unpacked_res_indices if num_res_centroids > 0 else None,
                        res_bits=res_bits if num_res_centroids > 0 else 0,
                        index_dtype=torch.uint16
                    )
                    model_data[key] = packed_indices
                    print(f'packed_indices: {packed_indices.shape}, dtype: {packed_indices.dtype}')
            model_data[key] = model_data[key].to('cpu')
            print('--------------------------------')    
        # end of rank, save the model
        safetensors.torch.save_file(model_data, f'{args.output_path}/model{rank}-mp{args.world_size}.safetensors')
        print(f'saved model{rank}-mp{args.world_size}.safetensors')
