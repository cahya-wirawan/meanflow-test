import torch
import torch.optim as optim

from meanflow import MeanFlowLanguageModel


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = pick_device()
    print(f"Smoke test device: {device}")

    vocab_size = 128
    seq_len = 16
    batch_size = 4

    model = MeanFlowLanguageModel(
        vocab_size=vocab_size,
        d_model=64,
        num_heads=4,
        num_layers=2,
        max_seq_len=seq_len,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # One training batch using synthetic token IDs.
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    model.train()
    optimizer.zero_grad()
    loss = model.compute_loss(input_ids, pad_token_id=None)
    loss.backward()
    optimizer.step()

    # One inference pass to validate generation path.
    model.eval()
    with torch.no_grad():
        generated = model.generate_1_step(batch_size=2, seq_len=seq_len, device=device)

    assert generated.shape == (2, seq_len), (
        f"Unexpected generated shape: {tuple(generated.shape)}"
    )

    print(f"Train loss: {loss.item():.4f}")
    print(f"Generated shape: {tuple(generated.shape)}")
    print("Smoke test passed")


if __name__ == "__main__":
    main()
