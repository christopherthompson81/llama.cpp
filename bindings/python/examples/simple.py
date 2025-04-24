#!/usr/bin/env python3
import argparse
import os
import sys

# Add the parent directory to the path so we can import the llama_cpp package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llama_cpp import LlamaModel

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
    
    print(f"Generating {args.max_tokens} tokens...")
    output_tokens = context.generate(
        tokens, 
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k
    )
    
    # Get just the generated text (excluding the prompt)
    generated_text = model.detokenize(output_tokens[len(tokens):])
    
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
