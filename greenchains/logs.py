from greenchains.config import LOGGER_NAME, LOG_FORMAT
import logging

logger = logging.getLogger(LOGGER_NAME)
logger.setLevel(logging.INFO)

# Configure handlers
handler = logging.StreamHandler()
log_formatter = logging.Formatter(fmt=LOG_FORMAT)
handler.setFormatter(log_formatter)
logger.addHandler(handler)
