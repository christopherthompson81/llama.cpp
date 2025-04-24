#!/usr/bin/env python3
import argparse
import os
import sys

# Add the parent directory to the path so we can import the llama_cpp package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llama_cpp import LlamaModel

def main():
    parser = argparse.ArgumentParser(description="Chat example using llama-cpp-python")
    parser.add_argument("--model", type=str, required=True, help="Path to the model file")
    parser.add_argument("--system", type=str, default="You are a helpful assistant.", help="System message")
    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k sampling")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--custom-template", type=str, default=None, help="Custom chat template")
    args = parser.parse_args()

    print(f"Loading model from {args.model}...")
    model = LlamaModel(args.model)
    
    print(f"Creating context...")
    context = model.create_context(n_ctx=4096, seed=args.seed)
    
    # Initialize chat history
    messages = [{"role": "system", "content": args.system}]
    
    print("\nChat initialized. Type 'exit' to quit.")
    print(f"System: {args.system}")
    
    while True:
        # Get user input
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            break
            
        # Add user message to history
        messages.append({"role": "user", "content": user_input})
        
        # Apply chat template
        prompt = model.apply_chat_template(messages, args.custom_template)
        
        # Tokenize the prompt
        tokens = model.tokenize(prompt)
        
        # Generate response
        output_tokens = context.generate(
            tokens, 
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k
        )
        
        # Get just the generated text
        generated_text = model.detokenize(output_tokens[len(tokens):])
        
        # Find the end of the assistant's message (if there's a user or system token)
        end_markers = [
            "\n<|user|>", "\n<|system|>", "<|user|>", "<|system|>",
            "\nuser:", "\nsystem:", "user:", "system:",
            "\nUser:", "\nSystem:", "User:", "System:"
        ]
        
        for marker in end_markers:
            if marker in generated_text:
                generated_text = generated_text.split(marker)[0]
                break
        
        # Clean up any remaining artifacts
        generated_text = generated_text.strip()
        
        # Print the assistant's response
        print(f"\nAssistant: {generated_text}")
        
        # Add assistant message to history
        messages.append({"role": "assistant", "content": generated_text})

if __name__ == "__main__":
    main()
