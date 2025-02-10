import logging
import sys

class SkipDebugPrintFilter(logging.Filter):
    """Skips any messages flagged as 'from_debug_print'."""
    def filter(self, record):
        return not getattr(record, "from_debug_print", False)

class ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG:    "\033[35m",	# magenta
        logging.INFO:     "\033[37m",	# white
        logging.WARNING:  "\033[33m",	# yellow
        logging.ERROR:    "\033[44m",	# blue background
        logging.CRITICAL: "\033[41m",	# red background
    }
    RESET = "\033[0m"
    def format(self, record):
        message = super().format(record)
        color = self.COLORS.get(record.levelno, self.RESET)
        return f"{color}{message}{self.RESET}"

logger = logging.getLogger("zyria_server")
logger.setLevel(logging.DEBUG)

# File handler (no color)
file_handler = logging.FileHandler("server.log", mode="w")
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)

# Console handler (color + skip debug_print logs)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(ColorFormatter("%(levelname)s: %(message)s"))
stream_handler.addFilter(SkipDebugPrintFilter())
logger.addHandler(stream_handler)

# =========== Debug Print ===========

COLORS = {
    "none": "\033[0m",
    "red": "\033[91m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "blue": "\033[94m",
    "magenta": "\033[95m",
    "cyan": "\033[96m",
    "white": "\033[97m"
}

def debug_print(message, level="info", color="white"):
    """Manual color print + log to file (with skip-filter on console)."""
    level = level.lower()
    color_code = COLORS.get(color, COLORS["none"])
    print(f"{color_code}{level.upper()}: {message}{COLORS['none']}")

    log_func = getattr(logger, level, logger.debug)
    log_func(message, extra={"from_debug_print": True})
