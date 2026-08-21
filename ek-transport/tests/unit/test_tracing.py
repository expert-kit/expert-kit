"""Tests for Frontend tracing exporter lifecycle."""

from expertkit_transport import tracing


class RecordingProvider:
    def __init__(self) -> None:
        self.flushes = 0
        self.shutdowns = 0

    def force_flush(self) -> None:
        self.flushes += 1

    def shutdown(self) -> None:
        self.shutdowns += 1


def test_flush_keeps_provider_available(monkeypatch) -> None:
    provider = RecordingProvider()
    monkeypatch.setattr(tracing, "_TRACER_PROVIDER", provider)

    tracing.flush_tracing()
    tracing.flush_tracing()

    assert provider.flushes == 2
    assert provider.shutdowns == 0
    assert tracing._TRACER_PROVIDER is provider


def test_shutdown_closes_provider_exactly_once(monkeypatch) -> None:
    provider = RecordingProvider()
    monkeypatch.setattr(tracing, "_TRACER_PROVIDER", provider)
    monkeypatch.setattr(tracing, "_TRACER", object())
    monkeypatch.setattr(tracing, "_TRACING_SHUT_DOWN", False)

    tracing.shutdown_tracing()
    tracing.shutdown_tracing()

    assert provider.shutdowns == 1
    assert tracing._TRACER_PROVIDER is None
    assert tracing._TRACER is None
    assert tracing._TRACING_SHUT_DOWN is True
