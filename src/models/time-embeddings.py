import torch
import torch.nn as nn

# =============================================================================
# SINUSOIDAL POSITIONAL EMBEDDING
# =============================================================================

class SinusoidalPositionalEmbedding(nn.Module):
    """
    Transforms a scalar timestep t into a high-dimensional embedding using 
    sinusoidal functions (sine and cosine).
    
    Formula: PE(pos, 2i) = sin(pos / 10000^(2i/d))
             PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
    
    Args:
        dim (int): Embedding dimension (typically 128, 256, etc.)
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t (torch.Tensor): Timestep tensor, shape (batch_size,)
        
        Returns:
            torch.Tensor: Positional embedding, shape (batch_size, dim)
        """
        sinusoid_inp = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)
        return emb


class TimeEmbeddingMLP(nn.Module):
    """
    Simple MLP to process the sinusoidal embeddings.
    
    Args:
        embedding_dim (int): Dimension of input sinusoidal embedding
        output_dim (int): Dimension of output representation
    """
    
    def __init__(self, embedding_dim: int, output_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, output_dim * 2),
            nn.SiLU(),
            nn.Linear(output_dim * 2, output_dim)
        )
    
    def forward(self, t_emb: torch.Tensor) -> torch.Tensor:
        return self.mlp(t_emb)