from email import generator
import os
import json
from argparse import ArgumentParser
from sys import argv
from typing import List, Generator
import copy

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from safetensors.torch import load_model, safe_open, load_file
from deepseek.model import Transformer, ModelArgs
from vptq.layers.vqlinear import VQuantLinear
from deepseek.model import ColumnParallelLinear, ColumnParallelVQLinear, RowParallelLinear, RowParallelVQLinear, Linear
from torch import nn
from tqdm import tqdm

def find_layers(module, target_layers=[nn.Linear], name=''):
    """Find all layers of specified types in a module using an iterative approach.
    
    Args:
        module: The PyTorch module to search
        target_layers: List of layer types to find
        name: Prefix for the layer names
        
    Returns:
        Dictionary mapping layer names to layer instances
    """
    from collections import deque
    
    result = {}
    # Use a deque for more efficient queue operations
    queue = deque([(name, module)])
    
    while queue:
        current_name, current_module = queue.popleft()  # More efficient than pop(0)
        
        # Check if current module is a target layer
        if type(current_module) in target_layers:
            result[current_name] = current_module
            
        # Add all children to the queue - only if they might contain target layers
        for child_name, child_module in current_module.named_children():
            next_name = child_name if current_name == '' else f'{current_name}.{child_name}'
            queue.append((next_name, child_module))
            
    return result

# Optimized version that directly accesses modules by path
def find_layers_by_type(module, target_types):
    """Find all layers of specified types in a module using a direct approach.
    
    Args:
        module: The PyTorch module to search
        target_types: List of layer types to find
        
    Returns:
        Dictionary mapping layer paths to layer instances
    """
    result = {}
    
    # Helper function to recursively find layers
    def _find_layers(m, path=''):
        if type(m) in target_types:
            result[path] = m
            
        for name, child in m.named_children():
            child_path = name if path == '' else f'{path}.{name}'
            _find_layers(child, child_path)
    
    _find_layers(module)
    return result

def build_module_path_map(module, prefix=''):
    """Build a direct mapping from full path names to (parent_module, attribute_name) pairs.
    
    This allows direct access to any module without traversing the hierarchy each time.
    
    Args:
        module: The PyTorch module to map
        prefix: Prefix for the module names
        
    Returns:
        Dictionary mapping full path names to (parent_module, attribute_name) pairs
    """
    from collections import deque
    
    path_map = {}
    queue = deque([(prefix, module, None)])
    
    while queue:
        current_prefix, current_module, parent_info = queue.popleft()
        
        if parent_info is not None:
            parent_module, attr_name = parent_info
            path_map[current_prefix] = (parent_module, attr_name)
        
        for child_name, child_module in current_module.named_children():
            child_prefix = child_name if current_prefix == '' else f'{current_prefix}.{child_name}'
            queue.append((child_prefix, child_module, (current_module, child_name)))
            
    return path_map

def batch_replace_layers(module, replacements):
    """Replace multiple layers in a single traversal.
    
    Args:
        module: The PyTorch module to modify
        replacements: Dictionary mapping layer names to new layers
        
    Returns:
        Number of successful replacements
    """
    if not replacements:
        return 0
        
    # Build a direct path map for efficient access
    path_map = build_module_path_map(module)
    
    # Perform all replacements directly
    successful = 0
    for target_name, new_layer in replacements.items():
        if target_name in path_map:
            parent_module, attr_name = path_map[target_name]
            setattr(parent_module, attr_name, new_layer)
            successful += 1
            
    return successful

def replace_layer(module, target_name, layer, module_name=None):
    """Replace a layer in a module using an iterative approach.
    
    Args:
        module: The PyTorch module to modify
        target_name: The name of the layer to replace
        layer: The new layer to insert
        module_name: Current module name prefix
        
    Returns:
        Boolean indicating if replacement was successful
    """
    from collections import deque
    
    # Use a deque for more efficient queue operations
    queue = deque([(module, None, module_name)])
    
    while queue:
        current_module, parent_name, current_name = queue.popleft()  # More efficient than pop(0)
        
        # Check all children of the current module
        for child_name, child_module in current_module.named_children():
            full_name = child_name if current_name is None else f'{current_name}.{child_name}'
            
            # If this is the target layer, replace it
            if full_name == target_name:
                setattr(current_module, child_name, layer)
                return True
                
            # Otherwise add it to the queue for further processing
            queue.append((child_module, child_name, full_name))
            
    return False

def convert_str_to_dtypes(obj):
    """Recursively convert string representations of torch.dtype objects back to torch.dtype in nested dictionaries and lists."""
    if isinstance(obj, dict):
        return {k: convert_str_to_dtypes(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_str_to_dtypes(item) for item in obj]
    elif isinstance(obj, str) and obj.startswith('torch.'):
        try:
            # Extract the dtype part from the string, create the dtype object using getattr
            return getattr(torch, obj.split('.')[-1])
        except AttributeError:
            # If the dtype string is not a valid torch dtype, return the string as is
            return obj
    else:
        return obj

# load layer config
def get_layer_config(quant_config: str) -> list[tuple[int, str, str]]:
    with open(quant_config, 'r') as f:
        quant_config = json.load(f)
    quant_config = quant_config['quantization_config']['config_for_layers']
    return quant_config

# 将嵌套函数移到外部，使其可以被pickle
def _process_single_layer(args):
    """Process a single layer for quantization.
    
    This function must be defined at module level (not nested) to be picklable for multiprocessing.
    
    Args:
        args: Tuple containing (layer_idx, layer_config_dict, target_layer_types)
        
    Returns:
        Tuple of (layer_idx, replacements_dict)
    """
    import torch
    from vptq.layers.vqlinear import VQuantLinear
    from deepseek.model import ColumnParallelLinear, ColumnParallelVQLinear, RowParallelLinear, RowParallelVQLinear, Linear
    
    layer_idx, layer_config_dict, target_layer_types = args
    layer_replacements = {}
    
    # Process each config directly without accessing the actual layer
    # This reduces the amount of data that needs to be pickled
    for op_name, op_config in layer_config_dict.items():
        try:
            # Extract the layer type from the config
            layer_type = op_config.pop('layer_type', None)
            if layer_type == 'ColumnParallelLinear':
                vqlinear = ColumnParallelVQLinear(**op_config)
            elif layer_type == 'RowParallelLinear':
                vqlinear = RowParallelVQLinear(**op_config)
            elif layer_type == 'Linear':
                vqlinear = VQuantLinear(**op_config)
            else:
                continue
                
            layer_replacements[op_name] = vqlinear
        except Exception as e:
            print(f"Warning: Failed to create quantized layer for layers.{layer_idx}.{op_name}: {str(e)}")
            continue
    
    return layer_idx, layer_replacements

def get_quantized_deepseek(model, ckpt_path, quant_config, 
                           world_size: int=1, rank: int=0,
                           dtype=torch.bfloat16,
                           num_load_processes: int = 1):
    import time
    from tqdm import tqdm
    import multiprocessing as mp
    from functools import partial
    import os
    import gc
    
    start_time = time.time()
    
    num_layers = len(model.layers)
    layers = model.layers
    quant_config = get_layer_config(quant_config)
    target_layers = [ColumnParallelLinear, RowParallelLinear, Linear]
    
    # OPTIMIZATION 1: Pre-compute all layer configurations and prepare them for multiprocessing
    # This reduces the amount of data that needs to be transferred between processes
    layer_configs_for_mp = {}
    
    # First find all target layers and their configurations
    for layer_idx in range(num_layers):
        layer = layers[layer_idx]
        ops = find_layers(layer, target_layers)
        
        layer_dict = {}
        for op_name, op in ops.items():
            full_op_name = f'layers.{layer_idx}.{op_name}'
            if full_op_name not in quant_config:
                continue
                
            # Get the configuration and add layer type information
            op_args = quant_config[full_op_name]
            op_args = convert_str_to_dtypes(op_args)
            op_args.pop('norm_dim', None)
            
            # Add layer type information to the config
            if type(op) == ColumnParallelLinear:
                op_args['layer_type'] = 'ColumnParallelLinear'
            elif type(op) == RowParallelLinear:
                op_args['layer_type'] = 'RowParallelLinear'
            elif type(op) == Linear:
                op_args['layer_type'] = 'Linear'
            else:
                continue
                
            layer_dict[op_name] = op_args
            
        if layer_dict:
            layer_configs_for_mp[layer_idx] = layer_dict
    
    # OPTIMIZATION 2: Process all layers at once with a global path map
    all_replacements = {}
    
    # OPTIMIZATION 3: Use multiprocessing for parallel layer processing
    # We can use multiprocessing even in distributed mode with careful resource management
    # Each distributed rank will use a subset of available CPU cores
    use_mp = num_layers > 16 and os.environ.get('DISABLE_MP', '0') != '1' and num_load_processes > 1
    
    # Calculate how many processes each rank should use - be more conservative
    total_cpu_count = mp.cpu_count()
    if world_size > 1:
        # In distributed mode, use fewer processes to avoid resource contention
        processes_per_rank = min(16, max(1, (total_cpu_count - 2) // world_size))
        processes_per_rank = min(processes_per_rank, num_load_processes)
    else:
        # In non-distributed mode, still be conservative
        processes_per_rank = min(64, max(1, total_cpu_count - 2))
        processes_per_rank = min(processes_per_rank, num_load_processes)

    if rank == 0:
        print(f"Rank {rank}: Using {processes_per_rank} processes for layer processing (out of {total_cpu_count} CPUs)")
    
    # Initialize multiprocessing safely
    if use_mp:
        try:
            # Check if multiprocessing is already initialized
            if mp.get_start_method(allow_none=True) is None:
                # Try to use 'spawn' method which is safer with CUDA
                try:
                    mp.set_start_method('spawn')
                except RuntimeError:
                    # If 'spawn' fails, try 'fork' as fallback
                    try:
                        mp.set_start_method('fork')
                    except RuntimeError:
                        use_mp = False
                        if rank == 0:
                            print("Warning: Could not initialize multiprocessing, falling back to sequential processing")
        except Exception as e:
            use_mp = False
            if rank == 0:
                print(f"Warning: Multiprocessing initialization failed: {e}, falling back to sequential processing")
    
    chunk_size = len(layer_configs_for_mp)
    
    process_args = [(idx, config_dict, target_layers) for idx, config_dict in layer_configs_for_mp.items()]
    
    if use_mp and processes_per_rank > 1 and process_args:
        try:
            ctx = mp.get_context('spawn')
            
            with ctx.Pool(processes=processes_per_rank) as pool:
                try:
                    results = list(tqdm(
                        pool.imap_unordered(_process_single_layer, process_args),
                        total=len(process_args),
                        desc=f"Rank {rank}: Preparing layers"
                    ))
                    
                    for layer_idx, replacements in results:
                        for op_name, vqlinear in replacements.items():
                            all_replacements[f'layers.{layer_idx}.{op_name}'] = (layer_idx, op_name, vqlinear)
                    
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    if rank == 0:
                        print(f"Warning: Processing failed: {str(e)}")
        except Exception as e:
            if rank == 0:
                print(f"Warning: Multiprocessing failed: {str(e)}, using sequential processing")
            use_mp = False
            
    if not use_mp or processes_per_rank <= 1 or not all_replacements:
        for args in tqdm(process_args, desc=f"Rank {rank}: Sequential processing"):
            try:
                layer_idx, replacements = _process_single_layer(args)
                for op_name, vqlinear in replacements.items():
                    all_replacements[f'layers.{layer_idx}.{op_name}'] = (layer_idx, op_name, vqlinear)
            except Exception as e:
                print(f"Warning: Failed to process layer {args[0]}")
                continue
    
    if rank == 0:
        print(f"Applying {len(all_replacements)} replacements")
    
    layer_grouped_replacements = {}
    for full_name, (layer_idx, op_name, vqlinear) in all_replacements.items():
        if layer_idx not in layer_grouped_replacements:
            layer_grouped_replacements[layer_idx] = {}
        layer_grouped_replacements[layer_idx][op_name] = vqlinear
    
    for layer_idx, replacements in tqdm(layer_grouped_replacements.items(), desc=f"Rank {rank}: Applying"):
        try:
            batch_replace_layers(layers[layer_idx], replacements)
        except Exception as e:
            print(f"Warning: Failed to apply replacements for layer {layer_idx}")
            continue
    
    elapsed = time.time() - start_time
    if rank == 0:
        print(f'Completed in {elapsed:.2f}s')
    
    # load state dict
    model_state_dict = load_file(os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors"))
    model.to('cuda')
    model.load_state_dict(model_state_dict)
    return model


def sample(logits, temperature: float = 1.0):
    """
    Samples a token from the logits using temperature scaling.

    Args:
        logits (torch.Tensor): The logits tensor for token predictions.
        temperature (float, optional): Temperature for scaling logits. Defaults to 1.0.

    Returns:
        torch.Tensor: The sampled token.
    """
    logits = logits / max(temperature, 1e-5)
    probs = torch.softmax(logits, dim=-1)
    return probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)


@torch.inference_mode()
def _interactive_generate(
    model: Transformer,
    prompt_tokens: List[List[int]],
    max_new_tokens: int,
    eos_id: int,
    temperature: float = 1.0,
    stream: bool = False
) -> List[List[int]] | Generator[int, None, None]:
    """
    Generates new tokens based on the given prompt tokens using the specified model.

    Args:
        model (Transformer): The transformer model used for token generation.
        prompt_tokens (List[List[int]]): A list of lists containing the prompt tokens for each sequence.
        max_new_tokens (int): The maximum number of new tokens to generate.
        eos_id (int): The end-of-sequence token ID.
        temperature (float, optional): The temperature value for sampling. Defaults to 1.0.

    Returns:
        List[List[int]]: A list of lists containing the generated tokens for each sequence.
    """
    prompt_lens = [len(t) for t in prompt_tokens]
    assert max(prompt_lens) <= model.max_seq_len, f"Prompt length exceeds model maximum sequence length (max_seq_len={model.max_seq_len})"
    total_len = min(model.max_seq_len, max_new_tokens + max(prompt_lens))
    tokens = torch.full((len(prompt_tokens), total_len), -1, dtype=torch.long, device="cuda")
    for i, t in enumerate(prompt_tokens):
        tokens[i, :len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
    prev_pos = 0
    finished = torch.tensor([False] * len(prompt_tokens), device="cuda")
    prompt_mask = tokens != -1
    for cur_pos in range(min(prompt_lens), total_len):
        logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
        if temperature > 0:
            next_token = sample(logits, temperature)
        else:
            next_token = logits.argmax(dim=-1)
        next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        tokens[:, cur_pos] = next_token
        finished |= torch.logical_and(~prompt_mask[:, cur_pos], next_token == eos_id)
        prev_pos = cur_pos
        
        if stream:
            yield next_token.item()
            
        if finished.all():
            break

    if not stream:
        completion_tokens = []
        for i, toks in enumerate(tokens.tolist()):
            toks = toks[prompt_lens[i]:prompt_lens[i]+max_new_tokens]
            if eos_id in toks:
                toks = toks[:toks.index(eos_id)]
            completion_tokens.append(toks)
        return completion_tokens


def interactive_generate(
    model: Transformer,
    tokenizer,
    max_new_tokens: int,
    temperature: float,
    world_size: int,
    rank: int
) -> None:
    """
    Handles interactive text generation mode with streaming output.
    """
    messages = []
    while True:
        if world_size == 1:
            prompt = input(">>> ")
        elif rank == 0:
            prompt = input(">>> ")
            objects = [prompt]
            dist.broadcast_object_list(objects, 0)
        else:
            objects = [None]
            dist.broadcast_object_list(objects, 0)
            prompt = objects[0]
        
        if prompt == "/exit":
            break
        elif prompt == "/clear":
            messages.clear()
            continue
            
        messages.append({"role": "user", "content": prompt})
        prompt_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        
        token_buffer = []
        for token in _interactive_generate(model, [prompt_tokens], max_new_tokens, 
                            tokenizer.eos_token_id, temperature, stream=True):
            token_buffer.append(token)
            if len(token_buffer) >= 1:
                text = tokenizer.decode(token_buffer, skip_special_tokens=True)
                print(text, end="", flush=True)
                token_buffer = []
                
        if token_buffer:
            text = tokenizer.decode(token_buffer, skip_special_tokens=True)
            print(text, end="", flush=True)
        print() 
        
        completion = tokenizer.decode(token_buffer, skip_special_tokens=True)
        messages.append({"role": "assistant", "content": completion})

@torch.inference_mode()
def generate(
    model: Transformer,
    prompt_tokens: List[List[int]],
    max_new_tokens: int,
    eos_id: int,
    temperature: float = 1.0
) -> List[List[int]]:
    """
    Generates new tokens based on the given prompt tokens using the specified model.

    Args:
        model (Transformer): The transformer model used for token generation.
        prompt_tokens (List[List[int]]): A list of lists containing the prompt tokens for each sequence.
        max_new_tokens (int): The maximum number of new tokens to generate.
        eos_id (int): The end-of-sequence token ID.
        temperature (float, optional): The temperature value for sampling. Defaults to 1.0.

    Returns:
        List[List[int]]: A list of lists containing the generated tokens for each sequence.
    """
    prompt_lens = [len(t) for t in prompt_tokens]
    assert max(prompt_lens) <= model.max_seq_len, f"Prompt length exceeds model maximum sequence length (max_seq_len={model.max_seq_len})"
    total_len = min(model.max_seq_len, max_new_tokens + max(prompt_lens))
    tokens = torch.full((len(prompt_tokens), total_len), -1, dtype=torch.long, device="cuda")
    for i, t in enumerate(prompt_tokens):
        tokens[i, :len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
    prev_pos = 0
    finished = torch.tensor([False] * len(prompt_tokens), device="cuda")
    prompt_mask = tokens != -1
    for cur_pos in range(min(prompt_lens), total_len):
        logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
        if temperature > 0:
            next_token = sample(logits, temperature)
        else:
            next_token = logits.argmax(dim=-1)
        next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        tokens[:, cur_pos] = next_token
        finished |= torch.logical_and(~prompt_mask[:, cur_pos], next_token == eos_id)
        prev_pos = cur_pos
        if finished.all():
            break
    completion_tokens = []
    for i, toks in enumerate(tokens.tolist()):
        toks = toks[prompt_lens[i]:prompt_lens[i]+max_new_tokens]
        if eos_id in toks:
            toks = toks[:toks.index(eos_id)]
        completion_tokens.append(toks)
    return completion_tokens


def batch_generate(
    model: Transformer,
    tokenizer,
    input_file: str,
    max_new_tokens: int,
    temperature: float,
    max_batch_size: int
) -> None:
    with open(input_file) as f:
        prompts = [line.strip() for line in f.readlines()]
    assert len(prompts) <= max_batch_size, f"Number of prompts exceeds maximum batch size ({max_batch_size})"
    prompt_tokens = [tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True) for prompt in prompts]
    completion_tokens = generate(model, prompt_tokens, max_new_tokens, tokenizer.eos_token_id, temperature)
    completions = tokenizer.batch_decode(completion_tokens, skip_special_tokens=True)
    for prompt, completion in zip(prompts, completions):
        print("Prompt:", prompt)
        print("Completion:", completion)
        print()
    with open("output.txt", "w") as f:
        for prompt, completion in zip(prompts, completions):
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Completion: {completion}\n")
            f.write("\n")

def main(
    ckpt_path: str,
    config: str,
    tokenizer_path: str = None,
    input_file: str = "",
    interactive: bool = True,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    quantize: bool = False,
    quant_config: str = "",
    num_load_processes: int = 1
) -> None:
    """
    Main function to load the model and perform interactive or batch text generation.

    Args:
        ckpt_path (str): Path to the model checkpoint directory.
        config (str): Path to the model configuration file.
        input_file (str, optional): Path to a file containing input prompts. Defaults to "".
        interactive (bool, optional): Whether to run in interactive mode. Defaults to True.
        max_new_tokens (int, optional): Maximum number of new tokens to generate. Defaults to 100.
        temperature (float, optional): Temperature for sampling. Defaults to 1.0.
    """
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
    global print
    if rank != 0:
        print = lambda *_, **__: None
    torch.cuda.set_device(local_rank)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.manual_seed(965)
    with open(config) as f:
        args = ModelArgs(**json.load(f))
    print(args)

    # load tokenizer
    if tokenizer_path is None:
        tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    else:
        print(f'load tokenizer from {tokenizer_path}')
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # load model
    model = Transformer(args)
    if quantize is False:
        # Always load distributed model files
        load_model(model, os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors"))
    else:
        model = get_quantized_deepseek(model, ckpt_path, quant_config, 
                                       world_size=world_size, rank=rank, 
                                       dtype=torch.bfloat16,
                                       num_load_processes=num_load_processes)
    
    if interactive:
        interactive_generate(model, tokenizer, max_new_tokens, temperature, world_size, rank)
    else:
        batch_generate(model, tokenizer, input_file, max_new_tokens, temperature, args.max_batch_size)
    
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    """
    Command-line interface for distributed text generation.

    Arguments:
        --ckpt-path (str): Path to the model checkpoint directory.
        --config (str): Path to the model configuration file.
        --input-file (str, optional): File containing prompts for batch processing.
        --interactive (bool, optional): Enable interactive mode for generating text.
        --max-new-tokens (int, optional): Maximum number of new tokens to generate. Defaults to 200.
        --temperature (float, optional): Temperature for sampling. Defaults to 0.2.

    Raises:
        AssertionError: If neither input-file nor interactive mode is specified.
    """
    parser = ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--tokenizer-path", type=str, default=None)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input-file", type=str, default="")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=65536)
    parser.add_argument("--temperature", type=float, default=0.15)
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--quant-config", type=str, default="")
    parser.add_argument("--num-load-processes", type=int, default=1)
    args = parser.parse_args()
     
    assert args.input_file or args.interactive
    
    os.environ['NCCL_DEBUG'] = 'OFF'
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'OFF'
    os.environ['TORCH_DISTRIBUTED_TIMEOUT'] = '3600'
    os.environ['NCCL_TIMEOUT'] = '3600'
    os.environ['NCCL_SOCKET_TIMEOUT'] = '360000'
    
    main(ckpt_path=args.ckpt_path, 
         tokenizer_path=args.tokenizer_path, 
         config=args.config, 
         input_file=args.input_file, 
         interactive=args.interactive, 
         max_new_tokens=args.max_new_tokens, 
         temperature=args.temperature, 
         quantize=args.quantize,
         quant_config=args.quant_config,
         num_load_processes=args.num_load_processes)
