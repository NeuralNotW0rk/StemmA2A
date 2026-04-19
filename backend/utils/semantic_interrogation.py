"""
Utility for Semantic Interrogation in StemmA2A.
Maps latent vectors to semantic labels using CLAP embeddings.
"""

import os
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional

class SemanticInterrogator:
    """
    Utility for mapping vectors to semantic text labels.
    Uses CLAP embeddings for unified audio-text similarity.
    """
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.labels: List[str] = []
        self.embeddings: Optional[torch.Tensor] = None

    def load_label_bank(self, labels: List[str], embeddings: torch.Tensor):
        """
        Loads pre-calculated text embeddings into the bank.
        Expects embeddings to be 2D tensors (num_labels, embed_dim).
        """
        self.labels = labels
        # Ensure 2D shape (N, D)
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)
        self.embeddings = embeddings.to(self.device)

    def add_to_bank(self, labels: List[str], embeddings: torch.Tensor):
        """Appends new labels and embeddings to the existing bank."""
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)
        embeddings = embeddings.to(self.device)
        
        if self.embeddings is None:
            self.labels = labels
            self.embeddings = embeddings
        else:
            self.labels.extend(labels)
            self.embeddings = torch.cat([self.embeddings, embeddings], dim=0)

    def save_bank(self, filepath: str):
        """Saves the label bank to disk for fast loading."""
        if self.embeddings is None:
            raise ValueError("No embeddings to save.")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        torch.save({
            "labels": self.labels,
            "embeddings": self.embeddings.cpu()
        }, filepath)

    def load_bank_from_disk(self, filepath: str):
        """Loads a saved label bank from disk."""
        data = torch.load(filepath, map_location=self.device)
        self.labels = data["labels"]
        self.embeddings = data["embeddings"].to(self.device)

    def interrogate(self, query: torch.Tensor, k: int = 3) -> List[Tuple[str, float]]:
        """
        Finds the top K most similar text labels for a given query vector.
        
        Args:
            query: Tensor representing the query vector. 
                      Should be a 1D vector (embed_dim) or 2D (1, embed_dim).
            k: Number of top labels to return.
            
        Returns:
            List of tuples: [(label_name, similarity_score), ...]
        """
        if self.embeddings is None:
            raise ValueError("Label bank is empty. Load embeddings first.")
            
        # Ensure query is on the correct device
        query = query.to(self.device)
        
        # Ensure 2D shape (1, D)
        if query.dim() == 1:
            query = query.unsqueeze(0)
            
        # Normalize vectors for cosine similarity
        query_norm = F.normalize(query, p=2, dim=-1)
        labels_norm = F.normalize(self.embeddings, p=2, dim=-1)
        
        # Compute cosine similarities via dot product
        # Shape: (1, num_labels)
        similarities = torch.mm(query_norm, labels_norm.t()).squeeze(0)
        
        # Get top K indices and values
        top_k_values, top_k_indices = torch.topk(similarities, k=min(k, len(self.labels)))
        
        results = []
        for val, idx in zip(top_k_values, top_k_indices):
            results.append((self.labels[idx.item()], val.item()))
            
        return results