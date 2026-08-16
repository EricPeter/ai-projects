import os

from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource

_tracer: trace.Tracer | None = None


def setup_telemetry() -> trace.Tracer:
    global _tracer
    if _tracer:
        return _tracer

    resource = Resource.create({"service.name": os.getenv("OTEL_SERVICE_NAME", "ai-proxy")})
    provider = TracerProvider(resource=resource)

    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if endpoint:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))

    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer("ai-proxy")
    return _tracer


def get_tracer() -> trace.Tracer:
    return _tracer or setup_telemetry()
