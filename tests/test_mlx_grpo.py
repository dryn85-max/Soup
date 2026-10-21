"""Native GRPO on MLX (#1976), Step 1: a dense base.

The objective is the one Soup's transformers path trains on TRL 0.29.1, whose
``GRPOConfig`` defaults Soup never overrides: ``loss_type="dapo"``,
``scale_rewards="group"``, ``epsilon=0.2``, ``num_iterations=1``,
``importance_sampling_level="token"``. Layers:

- units, prompt order, advantages, completion masks, reward kwargs and the data
  contract: pure Python, checked against independent references (no MLX);
- the loss and its gradients against a float64 NumPy oracle on fixed logits with
  synthetic old / reference log-probs (needs MLX), and the oracle against TRL's
  own ``GRPOTrainer._compute_loss`` (needs torch + trl, no MLX);
- generation, the adapter-off reference and training through the real wrapper
  on a tiny random Llama (needs MLX and mlx-lm). No downloads.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

# --------------------------------------------------------------------------
# A digit tokenizer with HF's surface. Ids: 0 pad, 1 BOS, 2 EOS, 3.. chars.
# Small enough that a tiny random model's completions often state a number.
# --------------------------------------------------------------------------

PAD, BOS, EOS = 0, 1, 2
_CHARS = "0123456789 :?<>/abcdefghijklmnopqrstuvwxyz|"
_IDS = {c: i + 3 for i, c in enumerate(_CHARS)}
VOCAB = len(_CHARS) + 3


class DigitTokenizer:
    bos_token_id = BOS
    eos_token_id = EOS
    pad_token_id = PAD
    eos_token = "$"
    chat_template = "<digit template>"

    def __init__(self, chars=_CHARS):
        self.chars = chars
        self.ids = {c: i + 3 for i, c in enumerate(chars)}

    def encode(self, text, add_special_tokens=True):
        return [self.ids[c] for c in text if c in self.ids]

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, **_):
        text = "".join(f"<{m['role'][0]}>{m['content']}|" for m in messages)
        text += "<a>" if add_generation_prompt else ""
        return self.encode(text) if tokenize else text

    def decode(self, ids, skip_special_tokens=False):
        out = []
        for i in ids:
            if i in (PAD, BOS, EOS):
                if not skip_special_tokens:
                    out.append({PAD: "_", BOS: "^", EOS: "$"}[i])
            else:
                out.append(self.chars[i - 3])
        return "".join(out)

    def batch_decode(self, sequences, skip_special_tokens=False):
        return [self.decode(s, skip_special_tokens=skip_special_tokens) for s in sequences]


# --------------------------------------------------------------------------
# Independent references
# --------------------------------------------------------------------------


def _np_advantages(rewards, group):
    r = np.asarray(rewards, dtype=np.float64).reshape(-1, group)
    mean = r.mean(axis=1, keepdims=True)
    std = r.std(axis=1, ddof=1, keepdims=True)
    return ((r - mean) / (std + 1e-4)).reshape(-1)


def _np_logps(logits, targets):
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    log_z = np.log(np.exp(shifted).sum(axis=-1))
    picked = np.take_along_axis(shifted, np.asarray(targets)[..., None], axis=-1)[..., 0]
    return picked - log_z


def _np_loss(logps, old, ref, adv, mask, *, beta, num_items, eps_low=0.2, eps_high=0.2):
    """TRL 0.29.1 ``_compute_loss`` for loss_type='dapo', token-level IS, in float64."""
    logps, mask = np.asarray(logps, np.float64), np.asarray(mask, np.float64)
    old = np.asarray(old, np.float64)
    adv = np.asarray(adv, np.float64)[:, None]
    coef_1 = np.exp(logps - old)
    coef_2 = np.clip(coef_1, 1 - eps_low, 1 + eps_high)
    per_token = -np.minimum(coef_1 * adv, coef_2 * adv)
    if beta != 0.0:
        d = np.asarray(ref, np.float64) - logps
        per_token = per_token + beta * (np.exp(d) - d - 1)
    return float((per_token * mask).sum() / num_items)


# --------------------------------------------------------------------------
# Units and prompt order
# --------------------------------------------------------------------------


class TestUnits:
    def test_batch_size_counts_completions_like_trl(self):
        from soup_cli.trainer.mlx_grpo import resolve_units

        units = resolve_units(batch_size=4, accum=2, num_generations=4)
        assert (units.completions_per_step, units.accum) == (4, 2)
        assert (units.prompts_per_update, units.completions_per_update) == (2, 8)
        assert not units.bumped

    def test_a_batch_smaller_than_one_group_is_raised_to_it(self):
        """grpo.py raises per_device_train_batch_size to num_generations."""
        from soup_cli.trainer.mlx_grpo import resolve_units

        units = resolve_units(batch_size=2, accum=1, num_generations=4)
        assert units.bumped and units.completions_per_step == 4
        assert units.prompts_per_update == 1

    @pytest.mark.parametrize("batch_size, accum, g", [(6, 1, 4), (5, 3, 4), (4, 1, 3)])
    def test_whole_groups_per_update_are_required(self, batch_size, accum, g):
        from soup_cli.trainer.mlx_grpo import resolve_units

        total = batch_size * accum
        with pytest.raises(ValueError, match=rf"\({total}\) must be divisible by "
                                             rf"num_generations \({g}\)"):
            resolve_units(batch_size=batch_size, accum=accum, num_generations=g)

    def test_odd_micro_batches_that_sum_to_whole_groups_are_accepted(self):
        from soup_cli.trainer.mlx_grpo import resolve_units

        units = resolve_units(batch_size=6, accum=2, num_generations=4)
        assert (units.prompts_per_update, units.completions_per_update) == (3, 12)

    def test_updates_per_epoch_drop_the_trailing_partial_chunk(self):
        from soup_cli.trainer.mlx_grpo import plan_prompt_updates

        plan = plan_prompt_updates(n_prompts=7, prompts_per_update=3, epochs=2, seed=0)
        assert len(plan) == 2 * (7 // 3)
        assert all(len(chunk) == 3 for chunk in plan)
        for epoch in (plan[:2], plan[2:]):
            seen = [i for chunk in epoch for i in chunk]
            assert len(set(seen)) == 6 and set(seen) <= set(range(7))

    def test_order_is_seeded_and_reshuffled_each_epoch(self):
        from soup_cli.trainer.mlx_grpo import plan_prompt_updates

        a = plan_prompt_updates(n_prompts=12, prompts_per_update=2, epochs=2, seed=3)
        b = plan_prompt_updates(n_prompts=12, prompts_per_update=2, epochs=2, seed=3)
        c = plan_prompt_updates(n_prompts=12, prompts_per_update=2, epochs=2, seed=4)
        assert a == b and a != c
        assert a[:6] != a[6:]

    def test_fewer_prompts_than_one_update_are_refused(self):
        from soup_cli.trainer.mlx_grpo import plan_prompt_updates

        with pytest.raises(ValueError, match="2 prompt"):
            plan_prompt_updates(n_prompts=2, prompts_per_update=3, epochs=1, seed=0)


# --------------------------------------------------------------------------
# Advantages
# --------------------------------------------------------------------------


class TestAdvantages:
    @pytest.mark.parametrize("rewards, group", [
        ([1.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.5, 0.0], 4),
        ([0.0, 1.0, 1.0, 0.0, 0.0, 0.0], 3),
        ([3.0, -2.0, 7.5, 0.25], 2),
        ([1e6, 1e6 + 3.0, -1e6, 5e5], 2),
        ([1e-7, 2e-7, 0.0, 3e-7], 4),
    ])
    def test_match_an_independent_reference(self, rewards, group):
        from soup_cli.trainer.mlx_grpo import group_advantages

        np.testing.assert_allclose(group_advantages(rewards, group),
                                   _np_advantages(rewards, group), rtol=1e-12, atol=1e-12)

    def test_equal_rewards_give_zero_advantage(self):
        from soup_cli.trainer.mlx_grpo import group_advantages

        assert group_advantages([0.5] * 4 + [1.0] * 4, 4) == [0.0] * 8

    def test_groups_are_normalised_separately(self):
        """Shifting one group's rewards must not move another group's advantages."""
        from soup_cli.trainer.mlx_grpo import group_advantages

        a = group_advantages([1.0, 0.0, 0.0, 0.0, 1.0, 2.0], 3)
        b = group_advantages([1.0, 0.0, 0.0, 100.0, 101.0, 102.0], 3)
        assert a[:3] == b[:3]
        np.testing.assert_allclose(a[3:], b[3:], rtol=1e-9)

    def test_sample_std_with_the_epsilon_outside_the_root(self):
        from soup_cli.trainer.mlx_grpo import group_advantages

        # Group (0, 1): mean 0.5, sample std sqrt(0.5) (population std would be 0.5).
        adv = group_advantages([0.0, 1.0], 2)
        assert adv[1] == pytest.approx(0.5 / (math.sqrt(0.5) + 1e-4), rel=1e-12)
        # Tiny spread: std 1e-6, so the epsilon dominates -- inside the root it would not.
        tiny = group_advantages([0.0, math.sqrt(2) * 1e-6], 2)
        assert tiny[1] == pytest.approx((math.sqrt(2) * 0.5e-6) / (1e-6 + 1e-4), rel=1e-9)

    @pytest.mark.parametrize("rewards, match", [
        ([1.0, 0.0, 1.0], "3 reward"),
        ([1.0, float("nan"), 0.0, 1.0], "non-finite"),
        ([1.0, float("inf"), 0.0, 1.0], "non-finite"),
        ([True, 0.0, 0.0, 1.0], "boolean"),
        ([None, 0.0, 0.0, 1.0], "non-numeric"),
    ])
    def test_bad_rewards_are_refused(self, rewards, match):
        from soup_cli.trainer.mlx_grpo import group_advantages

        with pytest.raises(ValueError, match=match):
            group_advantages(rewards, 2)

    def test_reward_statistics_match_trl_logging(self):
        from soup_cli.trainer.mlx_grpo import reward_statistics

        rewards = [1.0, 0.0, 1.0, 0.0, 0.5, 0.5, 0.5, 0.5]
        stats = reward_statistics(rewards, 4)
        assert stats["reward"] == pytest.approx(np.mean(rewards))
        assert stats["reward_std"] == pytest.approx(np.std(rewards, ddof=1))
        assert stats["frac_reward_zero_std"] == pytest.approx(0.5)


# --------------------------------------------------------------------------
# Completions and masks (pure)
# --------------------------------------------------------------------------


class TestCompletionMasks:
    def test_kept_through_the_first_eos(self):
        from soup_cli.trainer.mlx_grpo import cut_at_eos

        assert cut_at_eos([5, 6, EOS, 7, EOS], EOS) == ([5, 6, EOS], True)
        assert cut_at_eos([EOS, 5], EOS) == ([EOS], True)

    def test_truncated_completion_keeps_every_token(self):
        from soup_cli.trainer.mlx_grpo import cut_at_eos

        assert cut_at_eos([5, 6, 7], EOS) == ([5, 6, 7], False)

    def test_eos_equal_to_the_pad_id_is_still_a_completion_token(self):
        from soup_cli.trainer.mlx_grpo import collate_completions, cut_at_eos

        kept, done = cut_at_eos([5, 0, 6], 0)
        assert (kept, done) == ([5, 0], True)
        tokens, mask, start = collate_completions([([9, 9, 9], kept), ([9], [7])])
        # Row 0: the pad-valued EOS at position 4 is supervised; row 1's padding is not.
        assert start == 0
        assert mask[0] == [0, 0, 1, 1]
        assert mask[1] == [1, 0, 0, 0]
        assert tokens[0] == [9, 9, 9, 5, 0] and tokens[1] == [9, 7, 0, 0, 0]

    def test_mask_starts_at_each_rows_own_prompt_end(self):
        from soup_cli.trainer.mlx_grpo import collate_completions

        tokens, mask, start = collate_completions([([1, 2, 3, 4], [5, EOS]), ([1, 2], [6, 7, 8])])
        # Logit positions start at the shortest prompt's last token.
        assert start == 1
        width = len(tokens[0]) - 1 - start
        assert all(len(row) == width for row in mask)
        # Column j scores sequence position start + j + 1, i.e. positions 2..5.
        assert width == 4
        assert mask[0] == [0, 0, 1, 1]
        assert mask[1] == [1, 1, 1, 0]
        assert sum(map(sum, mask)) == 5


# --------------------------------------------------------------------------
# Rewards: TRL's kwargs shape, gold kept with its prompt
# --------------------------------------------------------------------------


class TestRewardKwargs:
    def test_trl_shape_and_every_non_prompt_column(self):
        from soup_cli.trainer.mlx_grpo import score_completions

        seen = {}

        def reward(completions, **kwargs):
            seen.update(kwargs, completions=completions)
            return [0.0] * len(completions)

        rows = [{"prompt": [{"role": "user", "content": "a"}], "answer": "1", "tag": "x"},
                {"prompt": [{"role": "user", "content": "b"}], "answer": "2", "tag": "y"}]
        score_completions(reward, rows, ["c1", "c2", "c3", "c4"], [[3], [4], [5], [6]], 2)
        assert seen["completions"] == [[{"role": "assistant", "content": c}]
                                       for c in ("c1", "c2", "c3", "c4")]
        assert seen["answer"] == ["1", "1", "2", "2"]
        assert seen["tag"] == ["x", "x", "y", "y"]
        assert seen["prompts"] == [rows[0]["prompt"]] * 2 + [rows[1]["prompt"]] * 2
        assert seen["completion_ids"] == [[3], [4], [5], [6]]

    def test_accuracy_scores_each_completion_against_its_own_gold(self):
        """Shuffled prompts, two of them identical with different golds."""
        from soup_cli.trainer.mlx_grpo import score_completions
        from soup_cli.trainer.rewards import accuracy_reward, validate_reward_funcs

        rows = [{"prompt": [{"role": "user", "content": "q"}], "answer": str(g)}
                for g in (3, 5, 7, 9)]
        rows.append({"prompt": [{"role": "user", "content": "q"}], "answer": "4"})
        rng = random.Random(0)
        order = list(range(len(rows)))
        rng.shuffle(order)
        chunk = [rows[i] for i in order]
        texts = [f"#### {row['answer']}" if k % 2 == 0 else "#### 0"
                 for row in chunk for k in range(2)]
        scores = score_completions(validate_reward_funcs(accuracy_reward), chunk, texts,
                                   [[3]] * len(texts), 2)
        assert scores == [1.0, 0.0] * len(chunk)

    def test_a_wrong_count_or_nonfinite_score_is_refused(self):
        from soup_cli.trainer.mlx_grpo import score_completions
        from soup_cli.trainer.rewards import validate_reward_funcs

        rows = [{"prompt": [{"role": "user", "content": "q"}]}]

        def short(completions, **_):
            return [1.0]

        def nan(completions, **_):
            return [float("nan")] * len(completions)

        with pytest.raises(ValueError, match="returned 1 scores for 2 completions"):
            score_completions(validate_reward_funcs(short), rows, ["a", "b"], [[3], [4]], 2)
        with pytest.raises(ValueError, match="non-finite"):
            score_completions(validate_reward_funcs(nan), rows, ["a", "b"], [[3], [4]], 2)


# --------------------------------------------------------------------------
# The loss: a fixed-logits fixture with synthetic old / reference log-probs
# --------------------------------------------------------------------------

#: (prompt length, completion length) per row: unequal lengths, one length-1.
_ROWS = [(3, 3), (2, 4), (3, 1), (2, 2)]
#: Advantages: positive, negative, zero, negative.
_ADV = [1.3, -0.7, 0.0, -1.1]
#: Ratio new/old per row: inside the trust region, above it, below it, above it.
#: With A>0 a ratio of 1.5 clips high; with A<0 a ratio of 0.5 clips low.
_RATIO = [1.1, 1.5, 0.5, 0.6]


def _fixture(seed=0, vocab=6):
    """Tokens, the collated mask, fixed logits and per-row synthetic logps."""
    from soup_cli.trainer.mlx_grpo import collate_completions

    rng = np.random.default_rng(seed)
    samples = [(list(rng.integers(3, vocab, p)), list(rng.integers(3, vocab, c)))
               for p, c in _ROWS]
    tokens, mask, start = collate_completions(samples)
    tokens, mask = np.array(tokens), np.array(mask, dtype=np.float64)
    width = tokens.shape[1] - 1
    logits = rng.normal(0.0, 1.5, (len(samples), width, vocab))
    return tokens, mask, start, logits


def _synthetic(logits, tokens, start):
    logps = _np_logps(logits[:, start:], tokens[:, 1:][:, start:])
    old = logps - np.log(np.array(_RATIO))[:, None]
    ref = logps + np.random.default_rng(9).normal(0.0, 0.3, logps.shape)
    return logps, old, ref


class _FixedLogits:
    def __new__(cls, logits):
        nn = pytest.importorskip("mlx.nn")
        mx = pytest.importorskip("mlx.core")

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.logits = mx.array(logits.astype(np.float32))

            def __call__(self, x):
                return self.logits

        return Model()


def _mlx_loss(model, tokens, mask, start, *, beta, old=None, ref=None, num_items=None):
    mx = pytest.importorskip("mlx.core")
    from soup_cli.trainer.mlx_grpo import completion_token_logps, grpo_token_loss

    logps = completion_token_logps(model, mx.array(tokens), start)
    return grpo_token_loss(
        logps, mx.array(np.array(_ADV, np.float32)), mx.array(mask.astype(np.float32)),
        num_items=float(mask.sum() if num_items is None else num_items), beta=beta,
        ref_logps=None if ref is None else mx.array(ref.astype(np.float32)),
        old_logps=None if old is None else mx.array(old.astype(np.float32)),
    )


def _fd_grad(f, logits, eps=1e-6):
    grad = np.zeros_like(logits)
    for idx in np.ndindex(*logits.shape):
        up, down = logits.copy(), logits.copy()
        up[idx] += eps
        down[idx] -= eps
        grad[idx] = (f(up) - f(down)) / (2 * eps)
    return grad


class TestTheLossMatchesTheOracle:
    @pytest.mark.parametrize("beta", [0.0, 0.1])
    def test_value_with_synthetic_old_and_reference(self, beta):
        tokens, mask, start, logits = _fixture()
        logps, old, ref = _synthetic(logits, tokens, start)
        loss, _ = _mlx_loss(_FixedLogits(logits), tokens, mask, start, beta=beta, old=old, ref=ref)
        expected = _np_loss(logps, old, ref, _ADV, mask, beta=beta, num_items=mask.sum())
        assert loss.item() == pytest.approx(expected, rel=1e-5, abs=1e-6)

    @pytest.mark.parametrize("beta", [0.0, 0.1])
    @pytest.mark.parametrize("synthetic_old", [True, False])
    def test_gradients_match_finite_differences(self, beta, synthetic_old):
        """Both clip sides, A of either sign and zero, unequal lengths, masked tokens.

        Without a synthetic old policy the ratio is 1 (old = new detached), so the
        oracle holds old fixed at the current log-probs while differentiating.
        """
        mx = pytest.importorskip("mlx.core")
        nn = pytest.importorskip("mlx.nn")
        tokens, mask, start, logits = _fixture(seed=1)
        logps, old, ref = _synthetic(logits, tokens, start)
        if not synthetic_old:
            old = logps
        model = _FixedLogits(logits)

        def f(m):
            return _mlx_loss(m, tokens, mask, start, beta=beta,
                             old=old if synthetic_old else None, ref=ref)[0]

        _, grads = nn.value_and_grad(model, f)(model)
        targets = tokens[:, 1:][:, start:]

        def oracle(z):
            return _np_loss(_np_logps(z[:, start:], targets), old, ref, _ADV, mask,
                            beta=beta, num_items=mask.sum())

        expected = _fd_grad(oracle, logits)
        np.testing.assert_allclose(np.array(grads["logits"]), expected, atol=2e-5, rtol=1e-3)
        assert np.abs(expected).max() > 1e-3
        mx.eval(grads)

    def test_clipped_tokens_carry_no_policy_gradient(self):
        mx = pytest.importorskip("mlx.core")
        nn = pytest.importorskip("mlx.nn")
        tokens, mask, start, logits = _fixture()
        logps, old, _ = _synthetic(logits, tokens, start)
        model = _FixedLogits(logits)
        _, grads = nn.value_and_grad(
            model, lambda m: _mlx_loss(m, tokens, mask, start, beta=0.0, old=old)[0]
        )(model)
        g = np.array(grads["logits"])
        # Row 1 (A<0, ratio 1.5) is unclipped; rows 2 (A=0) and 3 (A<0, ratio 0.6 < 0.8,
        # clipped low) give nothing, and row 0 (ratio 1.1) is inside the region.
        assert np.abs(g[0]).max() > 0 and np.abs(g[1]).max() > 0
        assert np.abs(g[2]).max() == 0 and np.abs(g[3]).max() == 0
        mx.eval(grads)

    def test_clip_metrics_count_each_side(self):
        tokens, mask, start, logits = _fixture()
        _, old, _ = _synthetic(logits, tokens, start)
        _, sums = _mlx_loss(_FixedLogits(logits), tokens, mask, start, beta=0.0, old=old)
        # Row 3: A<0 and ratio 0.6 < 0.8 -> low; row 1: A<0, ratio 1.5 -> neither side.
        assert sums["clip_low"].item() == _ROWS[3][1]
        assert sums["clip_high"].item() == 0
        assert sums["clip_region"].item() == _ROWS[3][1]
        assert sums["tokens"].item() == sum(c for _, c in _ROWS)

    def test_beta_zero_needs_no_reference_and_beta_adds_the_k3_kl(self):
        tokens, mask, start, logits = _fixture()
        logps, old, ref = _synthetic(logits, tokens, start)
        model = _FixedLogits(logits)
        plain, _ = _mlx_loss(model, tokens, mask, start, beta=0.0, old=old)
        with_kl, sums = _mlx_loss(model, tokens, mask, start, beta=0.25, old=old, ref=ref)
        d = ref - logps
        kl = ((np.exp(d) - d - 1) * mask).sum()
        assert with_kl.item() - plain.item() == pytest.approx(0.25 * kl / mask.sum(), rel=1e-4)
        assert sums["kl"].item() == pytest.approx(kl, rel=1e-4)
        with pytest.raises(ValueError, match="reference"):
            _mlx_loss(model, tokens, mask, start, beta=0.1)

    def test_the_normaliser_is_the_given_token_count_not_a_row_mean(self):
        """DAPO sums tokens; TRL's 'grpo' would average per row first."""
        tokens, mask, start, logits = _fixture()
        logps, old, ref = _synthetic(logits, tokens, start)
        loss, _ = _mlx_loss(_FixedLogits(logits), tokens, mask, start, beta=0.0, old=old,
                            num_items=100.0)
        assert loss.item() == pytest.approx(
            _np_loss(logps, old, ref, _ADV, mask, beta=0.0, num_items=100.0), rel=1e-5
        )
        per_row = -np.minimum(np.exp(logps - old) * np.array(_ADV)[:, None],
                              np.clip(np.exp(logps - old), 0.8, 1.2) * np.array(_ADV)[:, None])
        grpo_style = ((per_row * mask).sum(1) / mask.sum(1)).mean()
        assert abs(loss.item() * 100.0 / mask.sum() - grpo_style) > 1e-3

    def test_prompt_padding_and_post_completion_logits_move_nothing(self):
        mx = pytest.importorskip("mlx.core")
        nn = pytest.importorskip("mlx.nn")
        tokens, mask, start, logits = _fixture()
        _, old, ref = _synthetic(logits, tokens, start)
        model = _FixedLogits(logits)
        base, _ = _mlx_loss(model, tokens, mask, start, beta=0.1, old=old, ref=ref)
        # Every logit position whose target is not a completion token: the prompt
        # (before `start` and inside longer prompts) and the right padding.
        unused = np.ones(logits.shape[:2], dtype=bool)
        unused[:, start:] = mask == 0
        noisy = logits.copy()
        noisy[unused] += np.random.default_rng(5).normal(0, 50.0, noisy[unused].shape)
        moved, _ = _mlx_loss(_FixedLogits(noisy), tokens, mask, start, beta=0.1, old=old, ref=ref)
        assert moved.item() == pytest.approx(base.item(), abs=1e-6)
        _, grads = nn.value_and_grad(
            model, lambda m: _mlx_loss(m, tokens, mask, start, beta=0.1, old=old, ref=ref)[0]
        )(model)
        assert np.abs(np.array(grads["logits"])[unused]).max() == 0
        mx.eval(grads)


class TestTheOracleIsTrl:
    """The NumPy oracle against TRL 0.29.1's own ``GRPOTrainer._compute_loss``,
    bound to a stub that supplies the per-token log-probs (torch + trl, no MLX)."""

    @pytest.mark.parametrize("beta", [0.0, 0.1])
    def test_loss_and_gradient(self, beta):
        torch = pytest.importorskip("torch")
        pytest.importorskip("trl")
        from collections import defaultdict
        from types import SimpleNamespace

        from trl.trainer.grpo_trainer import GRPOTrainer

        tokens, mask, start, logits = _fixture(seed=2)
        logps, old, ref = _synthetic(logits, tokens, start)
        leaf = torch.tensor(logps, dtype=torch.float64, requires_grad=True)
        stub = SimpleNamespace(
            _get_per_token_logps_and_entropies=lambda *a, **k: (leaf, torch.zeros_like(leaf)),
            top_entropy_quantile=1.0, off_policy_mask_threshold=None,
            importance_sampling_level="token", beta=beta, loss_type="dapo",
            epsilon_low=0.2, epsilon_high=0.2, use_vllm=False,
            vllm_importance_sampling_correction=True,
            args=SimpleNamespace(use_bias_correction_kl=False, delta=None),
            model=SimpleNamespace(training=True),
            accelerator=SimpleNamespace(num_processes=1, gather=lambda t: t),
            _metrics={"train": defaultdict(list), "eval": defaultdict(list)},
        )
        width = mask.shape[1]
        inputs = {
            "prompt_ids": torch.zeros((len(_ROWS), 1), dtype=torch.long),
            "prompt_mask": torch.ones((len(_ROWS), 1), dtype=torch.long),
            "completion_ids": torch.zeros((len(_ROWS), width), dtype=torch.long),
            "completion_mask": torch.tensor(mask, dtype=torch.float64),
            "advantages": torch.tensor(_ADV, dtype=torch.float64),
            "old_per_token_logps": torch.tensor(old, dtype=torch.float64),
            "ref_per_token_logps": torch.tensor(ref, dtype=torch.float64),
            "num_items_in_batch": torch.tensor(float(mask.sum()) + 3.0, dtype=torch.float64),
        }
        loss = GRPOTrainer._compute_loss(stub, None, inputs)
        loss.backward()
        num_items = float(mask.sum()) + 3.0
        assert loss.item() == pytest.approx(
            _np_loss(logps, old, ref, _ADV, mask, beta=beta, num_items=num_items), rel=1e-12
        )

        def oracle(z):
            return _np_loss(z, old, ref, _ADV, mask, beta=beta, num_items=num_items)

        np.testing.assert_allclose(leaf.grad.numpy(), _fd_grad(oracle, logps), atol=1e-7)
        if beta:
            d = ref - logps
            assert stub._metrics["train"]["kl"][0] == pytest.approx(
                ((np.exp(d) - d - 1) * mask).sum() / mask.sum(), rel=1e-9
            )


# --------------------------------------------------------------------------
# A tiny random Llama
# --------------------------------------------------------------------------


def _tiny_llama(seed=0, vocab=VOCAB):
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    from mlx_lm.models import llama

    mx.random.seed(seed)
    return llama.Model(llama.ModelArgs(
        model_type="llama", hidden_size=32, num_hidden_layers=2, intermediate_size=64,
        num_attention_heads=4, num_key_value_heads=2, rms_norm_eps=1e-5, vocab_size=vocab,
        tie_word_embeddings=True,
    ))


def _attach_lora(model, rank=4, scale=2.0, dropout=0.0):
    from mlx_lm.tuner.utils import linear_to_lora_layers

    model.freeze()
    linear_to_lora_layers(model, len(model.layers), {
        "rank": rank, "scale": scale, "dropout": dropout,
        "keys": {"self_attn.q_proj", "self_attn.v_proj"},
    })
    return model


def _flat(params):
    from mlx.utils import tree_flatten

    return {k: np.array(v) for k, v in tree_flatten(params)}


class _Scripted:
    """A sampler that emits a fixed token per row per step and counts its calls."""

    def __init__(self, script):
        self.script, self.calls = script, 0

    def __call__(self, logprobs):
        mx = pytest.importorskip("mlx.core")
        step = self.calls
        self.calls += 1
        return mx.array([row[step] if step < len(row) else 9 for row in self.script])


class TestGeneration:
    def test_a_group_per_prompt_cut_at_the_first_eos(self):
        from soup_cli.trainer.mlx_grpo import generate_group

        script = _Scripted([[5, EOS, 7, 7], [5, 6, 7, 8], [EOS, 9, 9, 9]])
        out = generate_group(_tiny_llama(), [3, 4, 5], num_generations=3, max_tokens=4,
                             eos_token_id=EOS, sampler=script)
        assert out == [([5, EOS], True), ([5, 6, 7, 8], False), ([EOS], True)]
        assert script.calls == 4

    def test_stops_once_every_row_has_ended(self):
        from soup_cli.trainer.mlx_grpo import generate_group

        script = _Scripted([[5, EOS], [EOS, 3]])
        out = generate_group(_tiny_llama(), [3, 4], num_generations=2, max_tokens=50,
                             eos_token_id=EOS, sampler=script)
        assert out == [([5, EOS], True), ([EOS], True)] and script.calls == 2

    def test_seeded_rows_reproduce_and_differ_within_a_group(self):
        mx = pytest.importorskip("mlx.core")
        from soup_cli.trainer.mlx_grpo import generate_group

        model = _tiny_llama()

        def run(seed):
            mx.random.seed(seed)
            return generate_group(model, [3, 4, 5], num_generations=4, max_tokens=12,
                                  eos_token_id=EOS)

        first = run(0)
        assert len(first) == 4 and first == run(0) and first != run(1)
        assert len({tuple(t) for t, _ in first}) > 1

    def test_the_kv_cache_loop_matches_a_full_forward(self):
        """Each sampled token is drawn from the model's distribution given the prefix."""
        mx = pytest.importorskip("mlx.core")
        from soup_cli.trainer.mlx_grpo import generate_group

        model = _tiny_llama()
        model.eval()
        seen = []

        def greedy(logprobs):
            seen.append(np.array(logprobs))
            return mx.argmax(logprobs, axis=-1)

        prompt = [3, 4, 5, 6]
        (tokens, _), = generate_group(model, prompt, num_generations=1, max_tokens=5,
                                      eos_token_id=EOS, sampler=greedy)
        full = model(mx.array([prompt + tokens]))
        full = full - mx.logsumexp(full, axis=-1, keepdims=True)
        for k, step in enumerate(seen):
            np.testing.assert_allclose(step[0], np.array(full[0, len(prompt) - 1 + k]),
                                       atol=1e-4)


class TestReference:
    def _model(self):
        mx = pytest.importorskip("mlx.core")
        model = _tiny_llama()
        x = mx.array([[3, 4, 5, 6, 7]])
        base = np.array(model(x))
        _attach_lora(model)
        for _, m in model.named_modules():
            if hasattr(m, "lora_b"):
                m.lora_b = mx.random.normal(m.lora_b.shape) * 0.5
        return model, x, base

    def test_adapter_off_is_the_base_model_exactly(self):
        from soup_cli.trainer.mlx_grpo import adapter_disabled

        model, x, base = self._model()
        assert np.abs(np.array(model(x)) - base).max() > 1e-4
        with adapter_disabled(model) as count:
            np.testing.assert_array_equal(np.array(model(x)), base)
        assert count == 2 * len(model.layers)
        assert np.abs(np.array(model(x)) - base).max() > 1e-4

    def test_scales_come_back_after_an_exception(self):
        from soup_cli.trainer.mlx_grpo import _lora_modules, adapter_disabled

        model, _, _ = self._model()
        before = [m.scale for m in _lora_modules(model)]
        with pytest.raises(KeyError):
            with adapter_disabled(model):
                raise KeyError("boom")
        assert [m.scale for m in _lora_modules(model)] == before == [2.0] * len(before)

    def test_parameters_and_optimizer_state_are_untouched(self):
        mx = pytest.importorskip("mlx.core")
        nn = pytest.importorskip("mlx.nn")
        import mlx.optimizers as optim
        from mlx.utils import tree_flatten

        from soup_cli.trainer.mlx_grpo import adapter_disabled

        model, x, _ = self._model()
        opt = optim.AdamW(learning_rate=1e-2)
        _, grads = nn.value_and_grad(model, lambda m: m(x).sum())(model)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state)
        params = _flat(model.parameters())
        state = {k: np.array(v) for k, v in tree_flatten(opt.state)}
        trainable = set(_flat(model.trainable_parameters()))
        with adapter_disabled(model):
            mx.eval(model(x))
        assert set(_flat(model.trainable_parameters())) == trainable
        for k, v in _flat(model.parameters()).items():
            np.testing.assert_array_equal(v, params[k])
        for k, v in tree_flatten(opt.state):
            np.testing.assert_array_equal(np.array(v), state[k])

    def test_a_model_without_an_adapter_is_refused(self):
        from soup_cli.trainer.mlx_grpo import adapter_disabled

        with pytest.raises(RuntimeError, match="no LoRA module"):
            with adapter_disabled(_tiny_llama()):
                pass
