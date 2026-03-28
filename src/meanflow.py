import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def sinusoidal_embedding(timesteps, dim):
    """Sinusoidal positional embedding for scalar timesteps (diffusion-style).

    Converts a batch of scalar values into rich frequency representations,
    giving the network access to multiple frequency bands of the input signal.
    """
    half_dim = dim // 2
    emb = math.log(10000.0) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -emb)
    emb = timesteps.float() * emb
    emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


def _init_sinusoidal_positions(max_len, d_model):
    """Create a sinusoidal positional encoding table (non-learnable init)."""
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # [1, max_len, d_model]


class TimeConditionedTransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, ff_mult=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * d_model, d_model),
            nn.Dropout(dropout),
        )

        # Per-block FiLM conditioning from time embedding.
        self.time_to_attn_scale_shift = nn.Linear(d_model, 2 * d_model)
        self.time_to_ff_scale_shift = nn.Linear(d_model, 2 * d_model)

    def _apply_time_film(self, h, t_emb, proj):
        scale, shift = proj(t_emb).chunk(2, dim=-1)
        return h * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(self, x, t_emb):
        h = self.norm1(x)
        h = self._apply_time_film(h, t_emb, self.time_to_attn_scale_shift)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.attn_dropout(attn_out)

        h = self.norm2(x)
        h = self._apply_time_film(h, t_emb, self.time_to_ff_scale_shift)
        x = x + self.ff(h)
        return x

class MeanFlowLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=768,
        num_heads=12,
        num_layers=12,
        max_seq_len=256,
        prediction_target="x1",
        dropout=0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        if prediction_target not in {"x1", "v"}:
            raise ValueError("prediction_target must be one of {'x1', 'v'}")
        self.prediction_target = prediction_target
        
        # 1. The Continuous Bridge (Embedding & Positional Encoding)
        self.embedding = nn.Embedding(vocab_size, d_model)
        # Initialize positional encoding with sinusoidal values (learnable, but well-initialized).
        self.pos_encoding = nn.Parameter(
            _init_sinusoidal_positions(max_seq_len, d_model)
        )
        self.embed_dropout = nn.Dropout(dropout)
        
        # 2. The Time-Conditioned Transformer Backbone
        self.transformer = nn.ModuleList(
            [TimeConditionedTransformerBlock(d_model, num_heads, dropout=dropout) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)
        
        # Sinusoidal time embedding followed by MLP projection.
        # This gives the network rich multi-frequency access to the flow time,
        # rather than relying on a raw scalar input.
        self.time_embed = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        
        # 3. Output Head (Predicts the continuous vector)
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        
        # 4. The "Rounding" Head (Maps continuous vector back to text probabilities)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # Often, we tie the weights of the embedding and LM head for efficiency:
        self.lm_head.weight = self.embedding.weight 
        # Learnable logit temperature for cosine logits; helps avoid exploding CE.
        self.logit_scale = nn.Parameter(torch.tensor(math.log(math.sqrt(d_model))))

    def _get_time_embedding(self, t):
        """Convert scalar timesteps to sinusoidal embeddings and project."""
        # t: [batch, 1] -> sinusoidal_embedding expects [batch, 1]
        sin_emb = sinusoidal_embedding(t, self.d_model)  # [batch, d_model]
        return self.time_embed(sin_emb)  # [batch, d_model]

    def lm_logits(self, hidden_states, cosine=True):
        """Compute vocabulary logits from continuous hidden states.

        Args:
            hidden_states: [batch, seq_len, d_model]
            cosine: If True, use cosine-similarity logits with learnable
                temperature (better for generation).  If False, use standard
                dot-product logits (better for CE training signal).
        """
        if cosine:
            hidden = F.normalize(hidden_states, dim=-1)
            weight = F.normalize(self.lm_head.weight, dim=-1)
            scale = self.logit_scale.exp().clamp(min=1.0, max=100.0)
            return F.linear(hidden, weight) * scale
        return F.linear(hidden_states, self.lm_head.weight) / math.sqrt(self.d_model)

    def _forward_target(self, x_t, t):
        t_emb = self._get_time_embedding(t)  # [batch, d_model]

        if x_t.size(1) > self.max_seq_len:
            raise ValueError(
                f"Input seq_len={x_t.size(1)} exceeds max_seq_len={self.max_seq_len}."
            )

        # Positional encoding only — time conditioning is handled exclusively
        # by FiLM layers in each transformer block, avoiding redundant additive
        # injection that can interfere with the learned FiLM modulation.
        x = x_t + self.pos_encoding[:, :x_t.size(1), :]
        x = self.embed_dropout(x)
        for block in self.transformer:
            x = block(x, t_emb)
        x = self.final_norm(x)
        return self.output_head(x)

    def _target_to_x1(self, pred_target, x_t, t):
        if self.prediction_target == "x1":
            return pred_target
        t_expanded = t.unsqueeze(-1)
        return x_t + (1.0 - t_expanded) * pred_target

    def predict_velocity(self, x_t, t, eps=1e-4):
        pred_target = self._forward_target(x_t, t)
        if self.prediction_target == "v":
            return pred_target
        denom = (1.0 - t.unsqueeze(-1)).clamp_min(eps)
        return (pred_target - x_t) / denom

    def forward_net(self, x_t, t):
        """
        Takes the noisy latent state x_t and the current time t, 
        and predicts the final clean state x_1.
        """
        pred_target = self._forward_target(x_t, t)
        pred_x1 = self._target_to_x1(pred_target, x_t, t)
        return pred_x1

    def compute_loss(self, input_ids, pad_token_id=None, ce_weight=0.5, t_sample_power=1.0):
        """
        The training objective: teaches the model the Mean Flow trajectory.
        Includes masking to ignore padding tokens.
        """
        batch_size, seq_len = input_ids.shape
        
        # Step A: Map discrete text to the target continuous state (x_1)
        x_1 = self.embedding(input_ids) 
        
        # Step B: Sample the starting pure noise state (x_0)
        x_0 = torch.randn_like(x_1) 
        
        # Step C: Pick a random time t between 0 and 1.
        # Using t_sample_power > 1 biases sampling toward t ~ 0,
        # which better matches 1-step inference conditions.
        if t_sample_power <= 0:
            raise ValueError("t_sample_power must be > 0")
        t = torch.rand(batch_size, 1, device=x_1.device).pow(t_sample_power)
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
        # Use dot-product logits for training to provide stronger CE gradients;
        # cosine-similarity logits are used at inference time.
        logits = self.lm_logits(pred_x1, cosine=False)
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
        
        # Combine the losses with configurable CE weighting.
        return mse_loss + ce_weight * ce_loss

    @torch.no_grad()
    def generate_1_step(
        self,
        batch_size,
        seq_len,
        device="cpu",
        sample=False,
        temperature=1.0,
        top_k=0,
    ):
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
        logits = self.lm_logits(pred_x1)
        if not sample:
            tokens = torch.argmax(logits, dim=-1)
            return tokens

        if temperature <= 0:
            raise ValueError("temperature must be > 0")

        logits = logits / temperature

        if top_k > 0:
            k = min(top_k, logits.size(-1))
            topk_logits, topk_indices = torch.topk(logits, k=k, dim=-1)
            probs = F.softmax(topk_logits, dim=-1)
            sampled = torch.multinomial(probs.reshape(-1, k), num_samples=1)
            sampled = sampled.view(batch_size, seq_len, 1)
            tokens = topk_indices.gather(-1, sampled).squeeze(-1)
        else:
            probs = F.softmax(logits, dim=-1)
            sampled = torch.multinomial(probs.reshape(-1, probs.size(-1)), num_samples=1)
            tokens = sampled.view(batch_size, seq_len)
        
        return tokens
