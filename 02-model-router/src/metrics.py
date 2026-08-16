from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

requests_total = Counter(
    "router_requests_total",
    "Total routed requests",
    ["model", "complexity"],
)
cost_total = Counter(
    "router_cost_dollars_total",
    "Cumulative cost in USD",
    ["model"],
)
latency = Histogram(
    "router_latency_seconds",
    "Request latency",
    ["model"],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30],
)


def get_metrics_output() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
