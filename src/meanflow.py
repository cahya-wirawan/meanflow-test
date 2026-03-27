import torch
import torch.nn as nn
import torch.nn.functional as F

class MeanFlowLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_heads=12, num_layers=12, max_seq_len=256):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # 1. The Continuous Bridge (Embedding & Positional Encoding)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, d_model))
        
        # 2. The Time-Conditioned Transformer Backbone
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Neural network needs to know "where" it is in the flow time [0, 1]
        self.time_embed = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )
        
        # 3. Output Head (Predicts the continuous vector)
        self.output_head = nn.Linear(d_model, d_model)
        
        # 4. The "Rounding" Head (Maps continuous vector back to text probabilities)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # Often, we tie the weights of the embedding and LM head for efficiency:
        self.lm_head.weight = self.embedding.weight 

    def forward_net(self, x_t, t):
        """
        Takes the noisy latent state x_t and the current time t, 
        and predicts the final clean state x_1.
        """
        # Embed the time step t
        t_emb = self.time_embed(t).unsqueeze(1) # [batch, 1, d_model]

        if x_t.size(1) > self.max_seq_len:
            raise ValueError(
                f"Input seq_len={x_t.size(1)} exceeds max_seq_len={self.max_seq_len}."
            )
        
        # Add position and time information to the continuous latent state
        x = x_t + self.pos_encoding[:, :x_t.size(1), :] + t_emb
        
        # Process the entire sequence simultaneously in parallel
        hidden = self.transformer(x)
        
        # Predict the final continuous state x_1
        pred_x1 = self.output_head(hidden)
        return pred_x1

    def compute_loss(self, input_ids, pad_token_id=None):
        """
        The training objective: teaches the model the Mean Flow trajectory.
        Includes masking to ignore padding tokens.
        """
        batch_size, seq_len = input_ids.shape
        
        # Step A: Map discrete text to the target continuous state (x_1)
        x_1 = self.embedding(input_ids) 
        
        # Step B: Sample the starting pure noise state (x_0)
        x_0 = torch.randn_like(x_1) 
        
        # Step C: Pick a random time t between 0 and 1
        t = torch.rand(batch_size, 1, device=x_1.device)
        t_expanded = t.unsqueeze(-1) # Match dimensions [batch, 1, 1]
        
        # Step D: Construct the straight flow path (Linear Interpolation)
        x_t = t_expanded * x_1 + (1 - t_expanded) * x_0
        
        # Step E: Predict the target
        pred_x1 = self.forward_net(x_t, t)
        
        # --- Masking Logic ---
        # If pad_token_id is not provided, train on all tokens.
        if pad_token_id is None:
            mask = torch.ones_like(input_ids, dtype=torch.float)
        else:
            mask = (input_ids != pad_token_id).float()
        mask_expanded = mask.unsqueeze(-1) # [batch, seq_len, 1]
        
        # Step F: The Mean Flow Continuous Loss (Mean Squared Error)
        # We manually compute the masked MSE
        diff = (pred_x1 - x_1) * mask_expanded
        mse_loss = (diff**2).sum() / (mask.sum() * self.d_model + 1e-6)
        
        # Step G: Auxiliary Classification Loss
        logits = self.lm_head(pred_x1)
        if pad_token_id is None:
            ce_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                input_ids.view(-1),
            )
        else:
            ce_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                input_ids.view(-1),
                ignore_index=pad_token_id,
            )
        
        # Combine the losses (Increased CE weight to 0.5 to help early convergence)
        return mse_loss + 0.5 * ce_loss

    @torch.no_grad()
    def generate_1_step(self, batch_size, seq_len, device="cpu"):
        """
        Inference: Generates an entire block of text simultaneously in 1 step.
        """
        # 1. Start with a block of pure Gaussian noise
        x_0 = torch.randn(batch_size, seq_len, self.d_model, device=device)
        
        # 2. We are starting at time t = 0
        t = torch.zeros(batch_size, 1, device=device)
        
        # 3. Predict the final continuous state (The massive 1-step Mean Flow jump)
        pred_x1 = self.forward_net(x_0, t)
        
        # 4. Round the continuous state back to discrete text tokens
        logits = self.lm_head(pred_x1)
        tokens = torch.argmax(logits, dim=-1)
        
        return tokens