"""CLI alias for building trainer-ready tokenized SFT artifacts."""

from .precompute_lengths import main


if __name__ == "__main__":
    raise SystemExit(main())
