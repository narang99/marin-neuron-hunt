"""Chooses which source supplies the next document so token shares, not document shares, hit the target."""


class ShareScheduler:
    """Always draws from the source furthest below its target share.

    - Documents differ a lot in length (a paper versus a flan prompt), so a draw by document count
      would not give the weights in the config.
    - Ties go to the source listed first, which keeps the order deterministic.
    """

    def __init__(self, shares: dict[str, float]) -> None:
        if not shares:
            raise ValueError("need at least one source")
        self.shares = dict(shares)
        self.tokens: dict[str, int] = dict.fromkeys(shares, 0)

    def next_source(self) -> str:
        return min(self.shares, key=lambda name: self.tokens[name] / self.shares[name])

    def record(self, name: str, n_tokens: int) -> None:
        self.tokens[name] += n_tokens

    @property
    def realised_shares(self) -> dict[str, float]:
        total = sum(self.tokens.values())
        return {
            name: count / total if total else 0.0 for name, count in self.tokens.items()
        }
