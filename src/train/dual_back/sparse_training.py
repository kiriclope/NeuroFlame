def prune_and_grow_vectorized(W, mask, K):
    new_mask = mask.clone()

    # --- PRUNE ---
    big_num = 1e9
    # Set inactive weights to inf (so they're not selected for pruning)
    to_prune = W.abs().clone()
    to_prune[new_mask == 0] = big_num

    # Find indices of K smallest per-column
    prune_vals, prune_idx = torch.topk(-to_prune, K, dim=0)  # negative for smallest values

    # Vectorized column/row indices for scatter
    cols = torch.arange(N).view(1, -1).expand(K, -1)   # shape (K, N)
    new_mask[prune_idx, cols] = 0.0

    # --- GROW ---
    # Regrow: randomly select K locations per column where mask == 0
    # Make a mask for regrow-eligible entries
    grow_candidates = (new_mask == 0).float()
    # Random scores for each eligible position, -inf for non-candidates
    grow_scores = torch.rand_like(W) * grow_candidates + (1 - grow_candidates) * (-big_num)
    grow_vals, grow_idx = torch.topk(grow_scores, K, dim=0)
    new_mask[grow_idx, cols] = 1.0

    return new_mask
