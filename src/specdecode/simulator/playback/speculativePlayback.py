"""Playback simulation classes for speculative decoding."""

import time
from typing import Any, List, Optional

import torch

from specdecode.interface.abstractDrafter import AbstractDrafter
from specdecode.interface.abstractPlayback import AbstractPlayback
from specdecode.interface.abstractTensorDrafter import AbstractTensorDrafter
from specdecode.interface.abstractTensorVerifier import AbstractTensorVerifier
from specdecode.interface.abstractVerifier import AbstractVerifier
from specdecode.simulator.drafter.tensorNGramDrafter import PAD_ID
from specdecode.simulator.metrics.playbackMetrics import PlaybackMetrics


class SpeculativePlayback(AbstractPlayback):
    """
    Concrete implementation of AbstractPlayback that runs token-by-token
    speculative decoding simulations against ground truth.
    """

    def __init__(
        self,
        tokenizer: Any,
        drafter: AbstractDrafter,
        verifier: AbstractVerifier,
        metrics: Optional[PlaybackMetrics] = None,
    ) -> None:
        super().__init__(tokenizer, drafter, verifier, metrics)
        self.metrics: Optional[PlaybackMetrics] = metrics

    def run_playback(self, input_data: str, use_drafter: bool = True) -> str:  # noqa: C901
        complete_tokens = self.tokenizer.encode(input_data)
        if not complete_tokens:
            return ""

        if self.metrics:
            self.metrics.normal_steps = len(complete_tokens) - 1

        playback_start_ns = time.perf_counter_ns() if self.metrics else 0
        drafter_time_start_ns = self.metrics.drafter_wall_time_ns if self.metrics else 0

        current_prefix = [complete_tokens[0]]
        step_idx = 0

        while len(current_prefix) < len(complete_tokens):
            if use_drafter and self.drafter is not None:
                prefix_snapshot = list(current_prefix)
                drafter_start_ns = time.perf_counter_ns()
                draft_tokens = self.drafter.generate_draft(current_prefix)
                if self.metrics:
                    self.metrics.record_drafter_time(time.perf_counter_ns() - drafter_start_ns)
                n_used = getattr(self.drafter, "last_n_used", 0)

                verification_result = self.verifier.verify(
                    draft_tokens, current_prefix, complete_tokens
                )
                accepted_tokens = verification_result["accepted_tokens"]
                accepted_count = verification_result["accepted_count"]
                rejected_count = verification_result["rejected_count"]

                if not accepted_tokens:
                    next_gt_idx = len(current_prefix)
                    if next_gt_idx < len(complete_tokens):
                        current_prefix.append(complete_tokens[next_gt_idx])
                    if self.metrics:
                        self.metrics.record_step(
                            0,
                            0,
                            draft_size=len(draft_tokens),
                            step_idx=step_idx,
                            context_ids=prefix_snapshot,
                            draft_ids=draft_tokens,
                            complete_tokens=complete_tokens,
                            n_used=n_used,
                        )
                else:
                    current_prefix.extend(accepted_tokens)  # type: ignore[arg-type]
                    if self.metrics:
                        self.metrics.record_step(
                            accepted_count,  # type: ignore[arg-type]
                            rejected_count,  # type: ignore[arg-type]
                            draft_size=len(draft_tokens),
                            step_idx=step_idx,
                            context_ids=prefix_snapshot,
                            draft_ids=draft_tokens,
                            complete_tokens=complete_tokens,
                            n_used=n_used,
                        )
            else:
                next_gt_idx = len(current_prefix)
                if next_gt_idx < len(complete_tokens):
                    current_prefix.append(complete_tokens[next_gt_idx])
                if self.metrics:
                    self.metrics.speculative_steps += 1

            step_idx += 1

        decoded_string = self.tokenizer.decode(current_prefix)
        if self.metrics:
            self.metrics.total_tokens_generated = len(current_prefix)
            drafter_elapsed_ns = self.metrics.drafter_wall_time_ns - drafter_time_start_ns
            self.metrics.record_playback_time(
                time.perf_counter_ns() - playback_start_ns,
                excluded_ns=drafter_elapsed_ns,
            )

        return decoded_string


class TensorSpeculativePlayback(AbstractPlayback):
    """
    Playback loop driven by a tensor drafter (``[S, T]``) + tensor verifier.

    Mirrors SpeculativePlayback but the verifier returns the best candidate row;
    metrics reuse PlaybackMetrics with ``draft_size`` = real (non-PAD) length of the
    chosen row, so step-type classification and the mismatch log behave exactly as in
    the single-sequence path.
    """

    def __init__(
        self,
        tokenizer: Any,
        drafter: AbstractTensorDrafter,
        verifier: AbstractTensorVerifier,
        metrics: Optional[PlaybackMetrics] = None,
    ) -> None:
        super().__init__(tokenizer, drafter, verifier, metrics)  # type: ignore[arg-type]
        self.metrics: Optional[PlaybackMetrics] = metrics
        # Typed references that shadow the parent's AbstractDrafter/Verifier attributes
        # so pyright can verify the tensor-specific call signatures below.
        self._tensor_drafter: AbstractTensorDrafter = drafter
        self._tensor_verifier: AbstractTensorVerifier = verifier

    def run_playback(self, input_data: str, use_drafter: bool = True) -> str:
        complete_tokens = self.tokenizer.encode(input_data)
        reconstructed = self.run_tokens(complete_tokens, use_drafter=use_drafter)
        return self.tokenizer.decode(reconstructed)

    def run_tokens(self, complete_tokens: List[int], use_drafter: bool = True) -> List[int]:
        """Token-level driver (skips encode/decode) — used by the analysis harness."""
        if not complete_tokens:
            return []

        if self.metrics:
            self.metrics.normal_steps = len(complete_tokens) - 1

        playback_start_ns = time.perf_counter_ns() if self.metrics else 0
        drafter_time_start_ns = self.metrics.drafter_wall_time_ns if self.metrics else 0

        current_prefix = [complete_tokens[0]]
        step_idx = 0

        while len(current_prefix) < len(complete_tokens):
            if use_drafter and self.drafter is not None:
                prefix_snapshot = list(current_prefix)
                drafter_start_ns = time.perf_counter_ns()
                draft_tokens: torch.Tensor = self._tensor_drafter.generate_draft(current_prefix)
                if self.metrics:
                    self.metrics.record_drafter_time(time.perf_counter_ns() - drafter_start_ns)
                n_used = getattr(self._tensor_drafter, "last_n_used", 0)

                result = self._tensor_verifier.verify(draft_tokens, current_prefix, complete_tokens)
                accepted_tokens = result["accepted_tokens"]
                accepted_count = result["accepted_count"]
                rejected_count = result["rejected_count"]
                chosen = result.get("chosen_sequence", -1)
                real_len = accepted_count + rejected_count  # type: ignore[operator]

                chosen_ids: List[int] = []
                if self.metrics and chosen >= 0 and draft_tokens.numel() > 0:  # type: ignore[operator]
                    chosen_ids = [
                        int(t)
                        for t in draft_tokens[chosen].tolist()  # type: ignore[index]
                        if t != PAD_ID
                    ]

                current_prefix.extend(accepted_tokens)  # type: ignore[arg-type]
                if self.metrics:
                    self.metrics.record_step(
                        accepted_count,  # type: ignore[arg-type]
                        rejected_count,  # type: ignore[arg-type]
                        draft_size=real_len,  # type: ignore[arg-type]
                        step_idx=step_idx,
                        context_ids=prefix_snapshot,
                        draft_ids=chosen_ids,
                        complete_tokens=complete_tokens,
                        n_used=n_used,
                    )
            else:
                next_gt_idx = len(current_prefix)
                if next_gt_idx < len(complete_tokens):
                    current_prefix.append(complete_tokens[next_gt_idx])
                if self.metrics:
                    self.metrics.speculative_steps += 1

            step_idx += 1

        if self.metrics:
            self.metrics.total_tokens_generated = len(current_prefix)
            drafter_elapsed_ns = self.metrics.drafter_wall_time_ns - drafter_time_start_ns
            self.metrics.record_playback_time(
                time.perf_counter_ns() - playback_start_ns,
                excluded_ns=drafter_elapsed_ns,
            )

        return current_prefix
