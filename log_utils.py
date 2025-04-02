import logging
import sys
import inspect
import os

# ====== Color tables ======

COLORS = {
	"none":			"\033[0m",
	"dark_red":		"\033[31m",
	"dark_green":	"\033[32m",
	"dark_yellow":	"\033[33m",
	"dark_blue":	"\033[34m",
	"dark_magenta": "\033[35m",
	"dark_cyan":	"\033[36m",
	"grey":			"\033[37m",
	"red":			"\033[91m",
	"green":		"\033[92m",
	"yellow":		"\033[93m",
	"blue":			"\033[94m",
	"magenta":		"\033[95m",
	"cyan":			"\033[96m",
	"white":		"\033[97m"
}

LEVEL_DEFAULT_COLORS = {
	logging.DEBUG:	  "dark_magenta",
	logging.INFO:	  "grey",
	logging.WARNING:  "dark_yellow",
	logging.ERROR:	  "\033[44m",	# White on blue
	logging.CRITICAL: "\033[41m"	# White on red
}

# ====== Logger Components ======

class SkipDebugPrintFilter(logging.Filter):
	def filter(self, record):
		return not getattr(record, "from_debug_print", False)

class ColorFormatter(logging.Formatter):
	def __init__(self, fmt=None, color_enabled=True):
		super().__init__(fmt)
		self.color_enabled = color_enabled

	def format(self, record):
		if not self.color_enabled:
			return super().format(record)
		color = getattr(record, "color", None)
		if not color:
			color = LEVEL_DEFAULT_COLORS.get(record.levelno, "none")
		ansi = COLORS.get(color, COLORS["none"])
		message = super().format(record)
		return f"{ansi}{message}{COLORS['none']}"

# ====== Setup ======

def _apply_handlers(color_console=True, color_file=True):
	"""(Internal) Applies handlers to the logger."""
	logger.handlers = []

	# File handler
	file_handler = logging.FileHandler("zyria.log", mode="w")
	file_handler.setFormatter(ColorFormatter("%(asctime)s - %(levelname)s - %(message)s", color_enabled=color_file))
	file_handler.setLevel(logging.DEBUG)
	logger.addHandler(file_handler)

	# Console handler
	stream_handler = logging.StreamHandler(sys.stdout)
	stream_handler.setLevel(logging.DEBUG)
	stream_handler.setFormatter(ColorFormatter("%(levelname)s: %(message)s", color_enabled=color_console))
	stream_handler.addFilter(SkipDebugPrintFilter())
	logger.addHandler(stream_handler)

# ======== Initial Logger ========

logger = logging.getLogger("zyria_server")
logger.setLevel(logging.DEBUG)

# Default configuration when imported
_apply_handlers(color_console=True, color_file=True)

# ======== Reconfigure ========

def logger_reconfigure(color_console=True, color_file=True):
	"""Reconfigures logger handlers dynamically."""
	_apply_handlers(color_console=color_console, color_file=color_file)
	debug_print(f"Logger reconfigured: console_color={color_console}, file_color={color_file}", level="info", color="magenta")

# ====== debug_print() stays simple ======

def debug_print(message, level="info", color=None, quiet=False, end="\n"):
	"""Manual color print + logs to file (color can be overridden)."""
	level = level.lower()
	if color is None:
		color = "white"

	# Get caller info
	frame = inspect.currentframe()
	outer_frames = inspect.getouterframes(frame)
	caller_self = outer_frames[1].frame.f_locals.get('self', None)
	if caller_self:
		caller_name = caller_self.__class__.__name__
	else:
		# fallback to module name
		caller_name = os.path.splitext(os.path.basename(outer_frames[1].filename))[0]

	color_code = COLORS.get(color, COLORS["none"])

	if quiet:
		print(f"{color_code}{message}{COLORS['none']}", end=end)
	else:
		print(f"{color_code}[{level.upper()}]{caller_name}: {message}{COLORS['none']}", end=end)

	log_func = getattr(logger, level, logger.debug)
	log_func(f"{caller_name}: {message}", extra={"from_debug_print": True, "color": color})
