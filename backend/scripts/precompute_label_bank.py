import os
import sys
import torch

# Ensure we can import from the backend directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.encoders.clap_encoder import CLAPEncoder
from utils.semantic_interrogation import SemanticInterrogator

def main():
    print("Initializing CLAPEncoder...")
    encoder = CLAPEncoder()
    # Force model load
    encoder.load_model()
    
    # Read labels from the text file
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    labels_file = os.path.join(data_dir, "semantic_transitions.txt")
    
    if not os.path.exists(labels_file):
        print(f"Error: Could not find labels file at {labels_file}")
        sys.exit(1)
        
    with open(labels_file, "r", encoding="utf-8") as f:
        # Read lines, strip whitespace, ignore empty lines
        labels = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(labels)} labels from {labels_file}.")
    print("Generating text embeddings...")
    
    # laion_clap can handle lists of strings. 
    # Let's batch them just to be safe with memory, although this list is small enough to fit.
    batch_size = 32
    all_embeddings = []
    
    for i in range(0, len(labels), batch_size):
        batch = labels[i : i + batch_size]
        print(f"Processing batch {i//batch_size + 1}...")
        
        # Get embeddings. Shape: (batch_size, embed_dim)
        batch_embeddings = encoder.get_text_embedding(batch)
        all_embeddings.append(batch_embeddings)
        
    # Concatenate all batches
    final_embeddings = torch.cat(all_embeddings, dim=0)
    print(f"Generated embeddings tensor of shape: {final_embeddings.shape}")
    
    # Load into interrogator and save
    print("Saving label bank to disk...")
    interrogator = SemanticInterrogator(device="cpu")
    interrogator.load_label_bank(labels, final_embeddings)
    
    # Save to the data folder
    output_path = os.path.join(data_dir, "semantic_transitions_bank.pt")
    
    interrogator.save_bank(output_path)
    print(f"Successfully saved label bank to {output_path}")

if __name__ == "__main__":
    main()
