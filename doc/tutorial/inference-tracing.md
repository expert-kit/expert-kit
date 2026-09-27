# Inspect inference traces in Jaeger

Use an existing Jaeger collector that accepts plaintext OTLP/gRPC on port
`4317` and exposes its UI on port `16686`. The collector address must be
reachable from the attention container and every Worker container. Tracing is
disabled by default.

In the Ascend deployment's `dev/ascend/configs/experiment.yaml`, set:

```yaml
tracing:
  enabled: true
  endpoint: http://<COLLECTOR_HOST>:4317
  sample_ratio: 1.0
```

Use `1.0` for a short diagnostic run; lower it for longer runs because
recording and exporting spans adds overhead. From the repository root,
regenerate the deployment configuration:

```bash
uv run python dev/ascend/cli.py generate
```

Distribute the regenerated `dev/ascend/generated/` directory to every
participating Host, as in the [Ascend deployment workflow](ascend/README.md).
Restart this deployment's attention service and each expert pool with their
usual `dev/ascend/run-compose.sh ... up -d --force-recreate` commands so all
processes read the new configuration. Generated tracing enables vLLM eager
execution automatically.

Send a short inference request. Open `http://<JAEGER_HOST>:16686`, select the
`expertkit-frontend` service, and find a recent `frontend.model_forward` trace.
Open it to inspect its Worker spans under the same trace ID. Allow a short
export delay; normal process shutdown also flushes buffered spans. If the
Frontend span appears without Worker spans, check that every Worker can reach
the collector and that it restarted with the generated YAML.

To turn tracing off, set `tracing.enabled: false`, regenerate, redistribute,
and recreate the same services. For an overhead comparison, also set
`serve.enforce_eager: true` for both the tracing-on and tracing-off runs so
execution mode stays the same.

Deployments that do not use the Ascend generator can still configure the
Frontend with `EK_TRACE_ENDPOINT` and `EK_TRACE_SAMPLE_RATIO` plus vLLM
`--enforce-eager`, and configure each Worker through its
`observability.tracing` YAML settings.
