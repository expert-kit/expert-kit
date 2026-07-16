"""Tests for computation-ready weight references and withdrawal."""

from __future__ import annotations

import threading

import pytest

from expertkit_worker.weights import ReadyWeightTable, WeightsNotReady


def test_acquire_many_returns_direct_objects_and_counts_once_per_expert() -> None:
    table: ReadyWeightTable[object] = ReadyWeightTable(2, 4)
    first = object()
    second = object()
    table.publish(1, 0, first)
    table.publish(1, 3, second)

    lease = table.acquire_many(1, (0, 3))

    assert lease.expert_ids == (0, 3)
    assert lease.objects[0] is first
    assert lease.objects[1] is second
    assert table.usage_count(1, 0) == 1
    assert table.usage_count(1, 3) == 1
    lease.close()
    lease.close()
    assert table.usage_count(1, 0) == 0
    assert table.usage_count(1, 3) == 0


def test_missing_acquisition_changes_no_usage_counts() -> None:
    table: ReadyWeightTable[object] = ReadyWeightTable(1, 4)
    table.publish(0, 0, object())
    table.publish(0, 3, object())

    with pytest.raises(WeightsNotReady) as caught:
        table.acquire_many(0, (0, 2, 3))

    assert caught.value.expert_ids == (2,)
    assert table.usage_count(0, 0) == 0
    assert table.usage_count(0, 3) == 0


def test_withdrawal_blocks_new_acquisition_until_active_lease_releases() -> None:
    table: ReadyWeightTable[object] = ReadyWeightTable(1, 2)
    weight = object()
    table.publish(0, 1, weight)
    lease = table.acquire_many(0, (1,))
    table.begin_withdrawal(0, 1)
    removed: list[object] = []
    started = threading.Event()
    finished = threading.Event()

    def withdraw() -> None:
        started.set()
        removed.append(table.finish_withdrawal(0, 1, timeout=2))
        finished.set()

    thread = threading.Thread(target=withdraw)
    thread.start()
    assert started.wait(timeout=2) is True
    with pytest.raises(WeightsNotReady) as caught:
        table.acquire_many(0, (1,))
    assert caught.value.expert_ids == (1,)

    lease.close()
    assert finished.wait(timeout=2) is True
    thread.join(timeout=2)
    assert thread.is_alive() is False
    assert removed == [weight]
    assert table.is_ready(0, 1) is False


def test_publish_cannot_replace_a_ready_or_withdrawing_object() -> None:
    table: ReadyWeightTable[object] = ReadyWeightTable(1, 1)
    table.publish(0, 0, object())

    with pytest.raises(RuntimeError, match="already occupied"):
        table.publish(0, 0, object())
    table.begin_withdrawal(0, 0)
    with pytest.raises(RuntimeError, match="already occupied"):
        table.publish(0, 0, object())


@pytest.mark.parametrize("expert_ids", [(1, 0), (0, 0), (-1,), (4,)])
def test_acquire_many_rejects_invalid_distinct_expert_metadata(
    expert_ids: tuple[int, ...],
) -> None:
    table: ReadyWeightTable[object] = ReadyWeightTable(1, 4)

    with pytest.raises(ValueError):
        table.acquire_many(0, expert_ids)


def test_finish_withdrawal_requires_begin_and_honors_timeout() -> None:
    table: ReadyWeightTable[object] = ReadyWeightTable(1, 1)
    table.publish(0, 0, object())
    with pytest.raises(RuntimeError, match="begin_withdrawal"):
        table.finish_withdrawal(0, 0)

    lease = table.acquire_many(0, (0,))
    table.begin_withdrawal(0, 0)
    with pytest.raises(TimeoutError, match="timed out"):
        table.finish_withdrawal(0, 0, timeout=0)
    lease.close()
    table.finish_withdrawal(0, 0, timeout=0)
