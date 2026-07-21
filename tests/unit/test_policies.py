"""Phase 1 unit tests for versioned policies."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from policyshift.data_generation.policies import build_all_policies
from policyshift.environment.policy_store import PolicyStore
from policyshift.schemas import Domain

pytestmark = pytest.mark.phase1


def test_three_domains_three_versions() -> None:
    policies = build_all_policies()
    assert len(policies) == 9
    domains = {p.domain for p in policies}
    assert domains == {Domain.MATERIALS, Domain.LABORATORY, Domain.AI_GOVERNANCE}
    for domain in domains:
        versions = sorted(p.version for p in policies if p.domain == domain)
        assert versions == ["1.0", "1.1", "2.0"]


def test_effective_date_resolution() -> None:
    store = PolicyStore.from_builtin()
    mid_v10 = datetime(2024, 3, 1, tzinfo=timezone.utc)
    mid_v11 = datetime(2024, 9, 1, tzinfo=timezone.utc)
    mid_v20 = datetime(2025, 3, 1, tzinfo=timezone.utc)

    p10 = store.resolve_active(Domain.MATERIALS, mid_v10)
    p11 = store.resolve_active(Domain.MATERIALS, mid_v11)
    p20 = store.resolve_active(Domain.MATERIALS, mid_v20)
    assert p10 is not None and p10.version == "1.0"
    assert p11 is not None and p11.version == "1.1"
    assert p20 is not None and p20.version == "2.0"


def test_superseded_and_stale_detection() -> None:
    store = PolicyStore.from_builtin()
    when = datetime(2025, 3, 1, tzinfo=timezone.utc)
    assert store.is_stale("POL-MAT-RECV", "1.0", when)
    assert store.is_stale("POL-MAT-RECV", "1.1", when)
    assert not store.is_stale("POL-MAT-RECV", "2.0", when)


def test_boundary_exactly_at_effective() -> None:
    store = PolicyStore.from_builtin()
    boundary = datetime(2025, 1, 1, tzinfo=timezone.utc)
    active = store.resolve_active(Domain.LABORATORY, boundary)
    assert active is not None
    assert active.version == "2.0"


def test_checksum_present() -> None:
    for policy in build_all_policies():
        assert policy.checksum
        assert len(policy.checksum) == 64


def test_change_logs_nonempty() -> None:
    for policy in build_all_policies():
        if policy.version != "1.0":
            assert policy.change_log
            assert policy.supersedes
