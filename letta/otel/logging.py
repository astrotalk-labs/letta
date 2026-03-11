import logging

from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor

from letta.log import get_logger
from letta.otel.resource import get_resource, is_pytest_environment

logger = get_logger(__name__)
_is_log_exporting_initialized = False


def setup_logging(endpoint: str, service_name: str = "letta-server") -> None:
    if is_pytest_environment():
        return
    assert endpoint

    global _is_log_exporting_initialized

    log_exporter = OTLPLogExporter(endpoint=endpoint)
    logger_provider = LoggerProvider(resource=get_resource(service_name))
    logger_provider.add_log_record_processor(BatchLogRecordProcessor(log_exporter))
    set_logger_provider(logger_provider)

    # Attach OTel handler to root "Letta" logger — all child loggers inherit it
    otel_handler = LoggingHandler(level=logging.DEBUG, logger_provider=logger_provider)
    logging.getLogger("Letta").addHandler(otel_handler)

    _is_log_exporting_initialized = True
    logger.info("OTel log exporting initialised (endpoint=%s)", endpoint)
