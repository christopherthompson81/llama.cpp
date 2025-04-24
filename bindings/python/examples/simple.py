#!/usr/bin/env python3
import argparse
import os
import sys

# Add the parent directory to the path so we can import the llama_cpp package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from llama_cpp import LlamaModel

def sample_token(logits, temperature=0.8, top_p=0.95, top_k=40):
    """Simple token sampling function"""
    # Apply temperature
    if temperature > 0:
        logits = logits / temperature
    
    # Convert to probabilities
    probs = np.exp(logits - np.max(logits))
    probs = probs / np.sum(probs)
    
    # Apply top-k
    if top_k > 0:
        indices = np.argsort(-probs)[:top_k]
        probs_topk = probs[indices]
        probs_topk = probs_topk / np.sum(probs_topk)
        idx = np.random.choice(indices, p=probs_topk)
    else:
        idx = np.random.choice(len(probs), p=probs)
    
    return idx

def main():
    parser = argparse.ArgumentParser(description="Simple example of using llama-cpp-python")
    parser.add_argument("--model", type=str, required=True, help="Path to the model file")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Prompt to start generation")
    parser.add_argument("--max-tokens", type=int, default=100, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k sampling")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print(f"Loading model from {args.model}...")
    model = LlamaModel(args.model)
    
    print(f"Creating context...")
    context = model.create_context(n_ctx=2048, seed=args.seed)
    
    print(f"Tokenizing prompt: {args.prompt}")
    tokens = model.tokenize(args.prompt)
    n_prompt_tokens = len(tokens)

    # Create a batch for processing
    # Use n_batch from context if available, otherwise default (e.g., 512)
    # n_batch = context.n_batch if hasattr(context, 'n_batch') else 512 # Check LlamaContext API
    n_batch = 512 # Assuming default batch size if not directly accessible
    batch = model.create_batch(n_batch)
    
    print(f"Evaluating prompt...")
    output_tokens = []
    batch.n_tokens = 0 # Ensure batch is clear initially

    # Add prompt tokens to the batch
    for i, token in enumerate(tokens):
        # batch.add_sequence([token], seq_id=0, pos=i) # Preferred way if available
        # Manual population if add_sequence is not available/suitable:
        batch.tokens[batch.n_tokens] = token
        batch.pos[batch.n_tokens] = i
        batch.n_seq_id[batch.n_tokens] = 1
        batch.seq_id[batch.n_tokens][0] = 0 # Sequence ID 0
        batch.logits[batch.n_tokens] = 0 # Logits are ignored for input
        batch.n_tokens += 1
        output_tokens.append(token) # Keep track of all tokens

    # Decode the prompt batch
    context.decode(batch)

    print(f"Generating {args.max_tokens} tokens...")

    # Generation loop
    for i in range(args.max_tokens):
        # Get logits for the next token prediction
        logits = context.get_logits()
        # Sample the next token
        next_token = sample_token(
            logits,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k
        )

        # Add the new token to our output
        output_tokens.append(next_token)

        # Stop if we generate EOS
        if next_token == model.vocab.eos_token:
            break

        # Prepare the batch for the *next* token
        batch.n_tokens = 0 # Clear the batch for the next token
        current_pos = n_prompt_tokens + i # Position of the token we are about to decode

        # Add the single next token to the batch
        # batch.add_sequence([next_token], seq_id=0, pos=current_pos) # Preferred way
        # Manual population:
        batch.tokens[batch.n_tokens] = next_token
        batch.pos[batch.n_tokens] = current_pos
        batch.n_seq_id[batch.n_tokens] = 1
        batch.seq_id[batch.n_tokens][0] = 0 # Sequence ID 0
        batch.logits[batch.n_tokens] = 1 # Enable logits for this token
        batch.n_tokens += 1

        # Decode the single-token batch to update the context
        context.decode(batch)

    # Detokenize only the generated part
    # Convert list to numpy array for slicing if needed, or detokenize directly
    generated_tokens = output_tokens[n_prompt_tokens:]
    generated_text = model.detokenize(generated_tokens)

    print("\nGenerated text:")
    print(f"{args.prompt}{generated_text}")
        
    
    # Print some model information
    print("\nModel information:")
    print(f"  Embedding size: {model.n_embd}")
    print(f"  Context length: {model.n_ctx_train}")
    print(f"  Num layers: {model.n_layer}")
    print(f"  Num heads: {model.n_head}")
    print(f"  Vocab size: {model.vocab_size}")

if __name__ == "__main__":
    main()
