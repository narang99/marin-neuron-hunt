import pytest
import torch

from marin_sae.dictionary.batchtopk import BatchTopKSae
from marin_sae.dictionary.loss import auxk_loss, fvu, reconstruction_loss

D_IN, N_LATENTS, K, POOL = 16, 64, 4, 12


def make_sae(seed=0):
    generator = torch.Generator().manual_seed(seed)
    return BatchTopKSae(D_IN, N_LATENTS, K, POOL, torch.zeros(D_IN), generator)


def dense_reference(sae, code, edge=None):
    """Scatter the sparse code into a dense matrix and multiply, the slow obvious way."""
    dense = torch.zeros(code.indices.shape[0], N_LATENTS)
    values = code.values.clone()
    if edge is not None:
        values = values * (code.indices < edge)
    dense.scatter_(1, code.indices, values)
    return dense @ sae.W_dec + sae.b_dec


def test_select_keeps_exactly_batch_times_k_activations():
    sae = make_sae()
    x = torch.randn(32, D_IN)
    code = sae.select(sae.pre_activations(x))
    assert int((code.values > 0).sum()) == 32 * K


def test_sparse_decode_matches_the_dense_reference():
    sae = make_sae()
    code = sae.select(sae.pre_activations(torch.randn(8, D_IN)))
    (full,) = sae.decode(code, [N_LATENTS])
    torch.testing.assert_close(full, dense_reference(sae, code), atol=1e-5, rtol=1e-5)


def test_each_nested_reconstruction_uses_only_its_prefix_and_the_last_is_full():
    sae = make_sae()
    code = sae.select(sae.pre_activations(torch.randn(8, D_IN)))
    edges = [8, 24, N_LATENTS]
    parts = sae.decode(code, edges)
    for part, edge in zip(parts, edges):
        torch.testing.assert_close(
            part, dense_reference(sae, code, edge), atol=1e-5, rtol=1e-5
        )
    torch.testing.assert_close(
        parts[-1], dense_reference(sae, code), atol=1e-5, rtol=1e-5
    )


def test_normalising_the_decoder_gives_unit_rows():
    sae = make_sae()
    sae.W_dec.data *= torch.rand(N_LATENTS, 1) * 5 + 0.1
    sae.normalise_decoder()
    torch.testing.assert_close(sae.W_dec.norm(dim=-1), torch.ones(N_LATENTS))


def test_the_projected_gradient_is_orthogonal_to_each_decoder_row():
    sae = make_sae()
    x = torch.randn(8, D_IN)
    pre = sae.pre_activations(x)
    (x_hat,) = sae.decode(sae.select(pre), [N_LATENTS])
    fvu(x, x_hat).backward()
    sae.project_decoder_gradient()
    assert (sae.W_dec.grad * sae.W_dec).sum(dim=-1).abs().max() < 1e-5


def test_the_threshold_tracks_the_batch_wide_cut_during_training():
    sae = make_sae().train()
    for _ in range(300):
        sae.select(sae.pre_activations(torch.randn(64, D_IN)))
    assert sae.threshold.item() > 0


def test_the_threshold_path_uses_about_k_latents_once_calibrated():
    sae = make_sae().train()
    for _ in range(500):
        sae.select(sae.pre_activations(torch.randn(64, D_IN)))
    _, l0 = sae.reconstruct_at_threshold(torch.randn(256, D_IN))
    assert l0.item() == pytest.approx(K, rel=0.35)


def test_auxk_gradient_reaches_only_dead_latents():
    sae = make_sae()
    x = torch.randn(8, D_IN)
    pre = sae.pre_activations(x)
    (x_hat,) = sae.decode(sae.select(pre), [N_LATENTS])
    dead = torch.zeros(N_LATENTS, dtype=torch.bool)
    dead[:10] = True
    auxk_loss(sae, x, x_hat, pre, dead, pool=3).backward()
    assert sae.W_dec.grad[10:].abs().sum() == 0
    assert sae.W_dec.grad[:10].abs().sum() > 0


def test_auxk_is_zero_without_dead_latents():
    sae = make_sae()
    x = torch.randn(8, D_IN)
    pre = sae.pre_activations(x)
    (x_hat,) = sae.decode(sae.select(pre), [N_LATENTS])
    assert (
        auxk_loss(sae, x, x_hat, pre, torch.zeros(N_LATENTS, dtype=torch.bool), 3) == 0
    )


def sparse_data(n, generator, true_features):
    """Each row is a sum of two random true features, so a good SAE can find them."""
    pick = torch.randint(0, len(true_features), (n, 2), generator=generator)
    return true_features[pick].sum(dim=1)


@pytest.mark.parametrize("edges", [[N_LATENTS], [8, 24, N_LATENTS]])
def test_training_lowers_fvu_for_both_variants(edges):
    generator = torch.Generator().manual_seed(1)
    features = torch.nn.functional.normalize(
        torch.randn(24, D_IN, generator=generator), dim=-1
    )
    sae = make_sae().train()
    optimiser = torch.optim.Adam(sae.parameters(), lr=3e-3)
    first, last = None, None
    for step in range(400):
        x = sparse_data(128, generator, features)
        pre = sae.pre_activations(x)
        recon = sae.decode(sae.select(pre), edges)
        loss = reconstruction_loss(x, recon)
        optimiser.zero_grad()
        loss.backward()
        sae.project_decoder_gradient()
        optimiser.step()
        sae.normalise_decoder()
        first = fvu(x, recon[-1]).item() if step == 0 else first
        last = fvu(x, recon[-1]).item()
    assert last < 0.5 * first
