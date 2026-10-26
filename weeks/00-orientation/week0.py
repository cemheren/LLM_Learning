"""Inspect next-token predictions and generate text without model.generate()."""

import argparse
import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def sample_next_token(logits, temperature, generator=None):
    """Return one token ID per batch row from [batch, vocabulary] logits."""
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and non-negative")
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)
    probabilities = torch.softmax(logits.float() / temperature, dim=-1)
    return torch.multinomial(probabilities, num_samples=1, generator=generator)


@torch.inference_mode()
def generate_tokens(model, input_ids, max_new_tokens, temperature, generator=None,
                    eos_token_id=None, debug=False):
    """Recompute the full sequence each step; KV caching comes in Stage 2."""
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
        raise ValueError("input_ids must contain exactly one non-empty prompt")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and non-negative")
    if input_ids.shape[1] + max_new_tokens > model.config.max_position_embeddings:
        raise ValueError("prompt plus requested output exceeds the model context window")

    token_ids = input_ids.clone()
    generated = 0
    if debug:
        breakpoint()
    while generated < max_new_tokens:
        output = model(input_ids=token_ids, use_cache=False)
        next_logits = output.logits[:, -1, :]
        next_token = sample_next_token(next_logits, temperature, generator)
        token_ids = torch.cat((token_ids, next_token), dim=1)
        generated += 1
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break
    return token_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="openai-community/gpt2")
    parser.add_argument("--revision", default="607a30d783dfa663caf39e06633721c8d4cfcd7e",
                        help="Model revision; use a commit hash for reproducibility")
    parser.add_argument("--prompt", default="The best way to learn something new is")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="0 = greedy; positive values = sampling")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--debug", action="store_true",
                        help="Pause in the generation loop for interactive stepping")
    args = parser.parse_args()
    if not args.prompt.strip():
        parser.error("--prompt must not be empty")
    if args.max_new_tokens < 0:
        parser.error("--max-new-tokens must be non-negative")
    if not math.isfinite(args.temperature) or args.temperature < 0:
        parser.error("--temperature must be finite and non-negative")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; omit --device cuda to run on CPU")

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, use_safetensors=True,
    ).to(args.device).eval()
    input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(args.device)
    if input_ids.shape[1] + args.max_new_tokens > model.config.max_position_embeddings:
        parser.error("prompt plus requested output exceeds the model context window")
    generator = torch.Generator(device=args.device).manual_seed(args.seed)

    print(f"Model: {args.model}")
    print(f"Resolved revision: {getattr(model.config, '_commit_hash', 'unknown')}")
    print(f"Device: {args.device}; torch: {torch.__version__}")
    print(f"Prompt: {args.prompt!r}")
    print(f"Token IDs shape: {tuple(input_ids.shape)} = (batch, prompt tokens)")
    print("Tokens (spaces are significant):")
    for token_id in input_ids[0].tolist():
        print(f"  {token_id:5d}  {tokenizer.decode([token_id])!r}")

    with torch.inference_mode():
        logits = model(input_ids=input_ids, use_cache=False).logits
    print(f"Logits shape: {tuple(logits.shape)} = (batch, prompt tokens, vocabulary)")
    probabilities = torch.softmax(logits[0, -1, :].float(), dim=-1)
    top_probabilities, top_ids = probabilities.topk(5)
    print("Top 5 next tokens (unmodified distribution, temperature=1):")
    for probability, token_id in zip(top_probabilities.tolist(), top_ids.tolist()):
        print(f"  {tokenizer.decode([token_id])!r}: {probability:.4%}")

    output_ids = generate_tokens(
        model, input_ids, args.max_new_tokens, args.temperature, generator,
        eos_token_id=tokenizer.eos_token_id, debug=args.debug,
    )
    print(f"\nTemperature: {args.temperature}; seed: {args.seed}")
    print(f"Generated {output_ids.shape[1] - input_ids.shape[1]} new tokens")
    print("--- Full text (prompt + continuation) ---")
    print(tokenizer.decode(output_ids[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
