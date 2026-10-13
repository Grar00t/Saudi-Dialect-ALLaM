"""Dependency-free completion boundaries and prompt-plus-output budget checks."""

def validate_generation_budget(prompt_length: int, max_new_tokens: int, context_limit: int) -> None:
    for name, value in (("prompt_length", prompt_length), ("max_new_tokens", max_new_tokens), ("context_limit", context_limit)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if prompt_length + max_new_tokens > context_limit:
        raise ValueError(
            f"Context budget exceeded: prompt={prompt_length}, requested_new={max_new_tokens}, "
            f"context_limit={context_limit}. Shorten the prompt or reduce max_new_tokens."
        )


def decode_new_tokens(tokenizer, sequence, prompt_length: int) -> str:
    """Decode one decoder-only sequence after the input-token boundary.

    Accepts a 1D token tensor or a list. Delimiters in user text are never
    interpreted as a boundary. No model weights or tokenizer state are changed.
    """
    if isinstance(prompt_length, bool) or not isinstance(prompt_length, int) or prompt_length < 0:
        raise ValueError("prompt_length must be a nonnegative integer")
    if prompt_length > len(sequence):
        raise ValueError("Generated sequence is shorter than the input boundary")
    return tokenizer.decode(sequence[prompt_length:], skip_special_tokens=True).strip()
