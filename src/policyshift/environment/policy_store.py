"""In-memory versioned policy store with effective-date resolution."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from policyshift.schemas import Domain, PolicyClause, PolicyDocument
from policyshift.utils.io import read_json


class PolicyStore:
    """Load and query versioned synthetic policies."""

    def __init__(self, policies: list[PolicyDocument] | None = None) -> None:
        self._policies: dict[str, PolicyDocument] = {}
        if policies:
            for policy in policies:
                self.add(policy)

    def add(self, policy: PolicyDocument) -> None:
        self._policies[policy.version_key] = policy

    @classmethod
    def from_directory(cls, root: str | Path) -> PolicyStore:
        store = cls()
        root_path = Path(root)
        for path in sorted(root_path.rglob("*.json")):
            if path.name.endswith(".schema.json") or "schemas" in path.parts:
                continue
            raw = read_json(path)
            if isinstance(raw, list):
                for item in raw:
                    store.add(PolicyDocument.model_validate(item))
            else:
                store.add(PolicyDocument.model_validate(raw))
        return store

    @classmethod
    def from_builtin(cls) -> PolicyStore:
        from policyshift.data_generation.policies import build_all_policies

        return cls(build_all_policies())

    def get(self, policy_id: str, version: str) -> PolicyDocument | None:
        return self._policies.get(f"{policy_id}@{version}")

    def all(self) -> list[PolicyDocument]:
        return sorted(self._policies.values(), key=lambda p: (p.domain.value, p.effective_at, p.version))

    def list_for_domain(self, domain: Domain | str) -> list[PolicyDocument]:
        domain_value = domain.value if isinstance(domain, Domain) else domain
        return [p for p in self.all() if p.domain.value == domain_value]

    def list_available(self, domain: Domain | str, occurred_at: datetime) -> list[PolicyDocument]:
        """Return policies whose effective window covers occurred_at."""
        return [p for p in self.list_for_domain(domain) if p.is_effective_at(occurred_at)]

    def resolve_active(self, domain: Domain | str, occurred_at: datetime) -> PolicyDocument | None:
        available = self.list_available(domain, occurred_at)
        if not available:
            return None
        # Prefer latest effective_at among those covering the event time.
        return sorted(available, key=lambda p: p.effective_at)[-1]

    def is_stale(self, policy_id: str, version: str, occurred_at: datetime) -> bool:
        policy = self.get(policy_id, version)
        if policy is None:
            return True
        active = self.resolve_active(policy.domain, occurred_at)
        if active is None:
            return not policy.is_effective_at(occurred_at)
        return active.version_key != policy.version_key

    def search_clauses(
        self,
        query: str,
        domain: Domain | str,
        occurred_at: datetime,
        *,
        include_stale: bool = False,
    ) -> list[tuple[PolicyDocument, PolicyClause, float]]:
        """Naive keyword search over clauses; returns (policy, clause, score)."""
        query_l = query.lower().strip()
        tokens = [t for t in query_l.split() if t]
        results: list[tuple[PolicyDocument, PolicyClause, float]] = []
        candidates = (
            self.list_for_domain(domain) if include_stale else self.list_available(domain, occurred_at)
        )
        for policy in candidates:
            for clause in policy.clauses:
                hay = f"{clause.clause_id} {clause.text} {' '.join(clause.tags)}".lower()
                if not tokens:
                    score = 0.0
                else:
                    score = sum(1.0 for token in tokens if token in hay) / len(tokens)
                if score > 0 or not tokens:
                    results.append((policy, clause, score))
        results.sort(key=lambda item: item[2], reverse=True)
        return results
