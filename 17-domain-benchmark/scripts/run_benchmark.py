import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.runner import run
from benchmark.models.anthropic_model import AnthropicModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="anthropic/claude-sonnet-4-6")
    args = parser.parse_args()

    provider, model_id = args.model.split("/", 1) if "/" in args.model else ("anthropic", args.model)

    if provider == "anthropic":
        model = AnthropicModel(model=model_id)
    else:
        print(f"Unknown provider: {provider}")
        sys.exit(1)

    print(f"Running benchmark with {args.model}...")
    result = run(model, args.model)

    print(f"\nResults for {args.model}")
    print(f"  Overall:              {result['overall_score']:.4f}")
    print(f"  ROUGE-L:              {result['avg_rouge_l']:.4f}")
    print(f"  Semantic Similarity:  {result['avg_semantic_similarity']:.4f}")
    print(f"\nFull results saved to results/")


if __name__ == "__main__":
    main()
