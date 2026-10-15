"""The result of one request, and the prompt checks every engine makes."""

from dataclasses import dataclass


@dataclass
class Generation:
    token_ids: list[int]
    elapsed_s: float
    prefill_s: float
    denoise_steps: list[int]

    @property
    def request_tps(self):
        return len(self.token_ids) / self.elapsed_s

    @property
    def commit_forwards(self):
        return len(self.denoise_steps)


def check_prompt(prompt_ids, max_new_tokens, vocab_size):
    """The prompt as a list of ints, after the checks every engine makes before any GPU work."""
    ids = list(prompt_ids)
    if not ids or max_new_tokens <= 0:
        raise ValueError("A nonempty prompt and positive token budget are required")
    if any(type(x) is not int or not 0 <= x < vocab_size for x in ids):
        raise ValueError("Prompt contains an invalid token ID")
    return ids
