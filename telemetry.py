import logging
import os
import sys
from opentelemetry import trace, metrics
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.instrumentation.logging import LoggingInstrumentor

_tracer: trace.Tracer | None = None

chat_request_counter: metrics.Counter | None = None
chat_error_counter: metrics.Counter | None = None
tool_called_counter: metrics.Counter | None = None
phase1_latency_histo: metrics.Histogram | None = None
rag_latency_histo: metrics.Histogram | None = None


class _JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        import json
        _SKIP = {
            "name", "msg", "args", "levelname", "levelno", "pathname",
            "filename", "module", "exc_info", "exc_text", "stack_info",
            "lineno", "funcName", "created", "msecs", "relativeCreated",
            "thread", "threadName", "processName", "process", "message",
            "otelTraceID", "otelSpanID", "otelServiceName", "taskName",
        }
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S.%f"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "trace_id": getattr(record, "otelTraceID", None),
            "span_id": getattr(record, "otelSpanID", None),
            "service": "tinywebapp-chatbot",
        }
        for key, val in record.__dict__.items():
            if key not in _SKIP:
                payload[key] = val
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def setup_telemetry() -> None:
    global _tracer
    global chat_request_counter, chat_error_counter, tool_called_counter
    global phase1_latency_histo, rag_latency_histo

    resource = Resource.create({
        "service.name": "tinywebapp-chatbot",
        "service.version": os.getenv("IMAGE_TAG", "unknown"),
        "deployment.environment": os.getenv("ENVIRONMENT", "production"),
    })

    phoenix_api_key = os.getenv("PHOENIX_API_KEY", "")
    auth_headers = {"api_key": phoenix_api_key} if phoenix_api_key else {}

    traces_endpoint = os.getenv(
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        "https://app.phoenix.arize.com/v1/traces",
    )
    metrics_endpoint = os.getenv(
        "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT",
        "https://app.phoenix.arize.com/v1/metrics",
    )

    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=traces_endpoint, headers=auth_headers))
    )
    trace.set_tracer_provider(tracer_provider)

    metric_reader = PeriodicExportingMetricReader(
        OTLPMetricExporter(endpoint=metrics_endpoint, headers=auth_headers),
        export_interval_millis=60_000,
    )
    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)

    LoggingInstrumentor().instrument(set_logging_format=False)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JSONFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)

    _tracer = trace.get_tracer("tinywebapp")
    meter = metrics.get_meter("tinywebapp")

    chat_request_counter = meter.create_counter("chat.requests.total")
    chat_error_counter = meter.create_counter("chat.errors.total")
    tool_called_counter = meter.create_counter("chat.tool_calls.total")
    phase1_latency_histo = meter.create_histogram("bedrock.phase1.duration_ms")
    rag_latency_histo = meter.create_histogram("rag.duration_ms")


def get_tracer() -> trace.Tracer:
    return _tracer or trace.get_tracer("tinywebapp")
