from flask import Flask, request, jsonify
import configparser
from context_manager import ContextManager
from llm_manager import LLMManager
from memory_manager import MemoryManager
from prompt_builder import PromptBuilder
from npc_manager import NPCManager
from generator import Generator
from conversation_manager import ConversationManager
from attention_manager import AttentionManager
import traceback
import time
from log_utils import debug_print, logger, logger_reconfigure
import uuid
import sys
import json

config = configparser.ConfigParser()
config.read(['zyria.conf', 'prompts.conf'])
server_port = config.get('Server', 'Port', fallback='5050')
server_timeout = int(config.get('Server', 'ServerTimeout', fallback=90))

color_console = config.getboolean("Logging", "ColorConsole", fallback=True)
color_file = config.getboolean("Logging", "ColorFile", fallback=True)
logger_reconfigure(color_console=color_console, color_file=color_file)

# Initialize Flask app
app = Flask(__name__)

# Initialize all components
npc_manager = NPCManager(config)
context_manager = ContextManager(config)
conversation_manager = ConversationManager(config, context_manager) # Initialize ConversationManager, pass it ContextManager

llm_manager = LLMManager(config,	# Initialize LLMManager, pass it ContextManager
	context_manager=context_manager,
	response_callback=conversation_manager.receive_llm_response
)
llm_manager.update_words()	# Check for english dictionary and download if missing

memory_manager = MemoryManager(config, llm_manager) # Initialize MemoryManager, pass it LLMManager
prompt_builder = PromptBuilder(config, memory_manager, context_manager, npc_manager) # Initialize PromptBuilder
attention_manager = AttentionManager(config)

generator = Generator(config,	# Initialize Generator, pass all to it
	prompt_builder=prompt_builder,
	conversation_manager=conversation_manager,
	memory_manager=memory_manager,
	npc_manager=npc_manager,
	context_manager=context_manager,
	llm_manager=llm_manager,
	attention_manager=attention_manager
)

def generate_request_id(time_received):
	return f"{time_received}-{uuid.uuid4().hex[:8]}"

@app.route("/zyria/v1/generate", methods=["POST"])
def generate_request():
	request_data = request.get_json()

	# Track when the request was received (in milliseconds)
	time_received = int(time.time() * 1000)
	logger.info(f"Received request from cMaNGOS at {time_received}")

	if not request_data:
		return jsonify({"error": "Invalid JSON payload"}), 400

	# ✅ Validate required fields BEFORE processing
	required_fields = ["time_created", "message_type", "llm_channel"]
	missing_fields = [field for field in required_fields if field not in request_data]

	if missing_fields:
		return jsonify({"error": f"Missing required fields: {', '.join(missing_fields)}"}), 400

	# ✅ Ensure time_created is a valid float
	try:
		time_created = float(request_data["time_created"])
	except (ValueError, TypeError):
		return jsonify({"error": "Invalid time_created timestamp"}), 400

	# ✅ Extract message_type and llm_channel
	message_type = request_data["message_type"]
	llm_channel = request_data["llm_channel"]

	# ✅ Default messages
	default_messages = {
		"new": "__initiate_chat__",
		"rpg": "__rpg_chat__"
	}
	message = request_data.get("message", default_messages.get(message_type, "__none__"))
	try:
		unescaped_message = json.loads(f'"{message}"')	# wraps the message in quotes and decodes it
	except json.JSONDecodeError:
		unescaped_message = message	 # fallback if somehow malformed

	request_data["message"] = unescaped_message.strip()

	# ✅ Get RPG sender/speaker if applicable
	if message_type == "rpg":
		speaker_sender_request_data = get_rpg_speaker_sender(request_data)
	else:
		speaker_sender_request_data = {
			"sender_type": request_data.get("sender", {}).get("type", "unknown"),
			"sender_name": request_data.get("sender", {}).get("name", "Unknown"),
			"speaker_name": request_data.get("speaker", {}).get("name", "Unknown"),
			"speaker_afk": request_data.get("speaker", {}).get("afk", False)
		}

	sender_type = speaker_sender_request_data["sender_type"]
	sender_name = speaker_sender_request_data["sender_name"]
	speaker_name = speaker_sender_request_data["speaker_name"]
	speaker_afk = speaker_sender_request_data["speaker_afk"]
	channel_members = request_data.get("channel_members", {})

	if sender_type == "player":
		conversation_manager.prioritize_player_message(llm_channel)
		conversation_manager.unsuspend_queue(llm_channel)
		debug_print(f"Allowed request from player <{sender_name}> and released channel {llm_channel} from suspension", color="cyan")

	else:
		if conversation_manager.is_bot_busy(speaker_name):
			# Reject request if bot is busy (ignoring new messages)
			remaining_busy_time = conversation_manager.get_bot_remaining_busy_time(speaker_name)
			debug_print(f"Rejected request for <{speaker_name}> (entity is busy, time remaining = {remaining_busy_time})", color="red")
			return jsonify({"error": f"Entity <{speaker_name}> is busy"}), 400
		elif speaker_afk:
			debug_print(f"Rejected request for <{speaker_name}> (bot is AFK)", color="red")
			return jsonify({"error": f"Bot <{speaker_name}> is AFK"}), 400
		elif channel_members and speaker_name not in channel_members:
			debug_print(f"Rejected request for <{speaker_name}> (not in channel members)", color="red")
			return jsonify({"error": f"Entity <{speaker_name}> not a channel member"}), 400
		else:
			if conversation_manager.is_suspended(llm_channel):
				if message_type == "new":
					conversation_manager.unsuspend_queue(llm_channel)
					debug_print(f"Released channel {llm_channel} from suspension for new message from <{sender_name}>", color="cyan")
				else:
					debug_print(f"Rejected request for <{speaker_name}> (LLM channel is suspended)", color="red")
					return jsonify({"error": f"Channel {llm_channel}> is suspended"}), 400
			elif conversation_manager.is_channel_overloaded(llm_channel):
				debug_print(f"Rejected request for <{speaker_name}> (Channel {llm_channel} is overloaded)", color="red")
				return jsonify({"error": "LLM channel is overloaded"}), 400

	request_id = generate_request_id(time_received)
	request_data["request_id"] = request_id
	request_data["time_received"] = time_received
	request_data["status"] = "pending"
	request_data["sender_name"] = sender_name
	request_data["speaker_name"] = speaker_name

	debug_print(request_data, color={"new": "green", "reply": "yellow", "rpg": "cyan"}.get(message_type, "white"), quiet=True)

	generator.process_or_start(llm_channel)
	conversation_manager.add_request(llm_channel, request_data)

	# Mark the speaker as busy
	conversation_manager.set_bot_busy(speaker_name)

	try:
		start_time = time.time()

		while time.time() - start_time < server_timeout:
			# Try to fetch the completed request
			completed_request = conversation_manager.fetch_completed_request(llm_channel, request_id)

			if completed_request:
				response_delay = completed_request.get("response_delay", 0)
				if response_delay > 0:
					debug_print(f"Delaying response from <{speaker_name}> for {response_delay:.2f} seconds", color="yellow")
					time.sleep(response_delay)

				response = completed_request.get("mangos_response", {})
				debug_print(f"✅ Successfully processed request {request_id} and returned response to cMaNGOS after {time.time() - start_time:.2f} seconds.")
				return jsonify(response)

			# ✅ Sleep briefly before checking again to avoid busy-waiting
			time.sleep(0.1)

		# 🚨 Timeout: No response received in time
		logger.error(f"❌ Server timeout! No response for request_id {request_id} after {server_timeout} seconds.")

		# 🔥 Release the bot to prevent permanent busy state
		for req in conversation_manager.conversation_queues[llm_channel]:
			if req.get("request_id") == request_id:
				sname = req.get("speaker", {}).get("name", "Unknown")
				debug_print(f"Timeout detected! Releasing stuck bot <{sname}>", color="red")
				conversation_manager.set_bot_busy(sname, delay=0)
				break

		return jsonify({"error": "LLM processing timeout"}), 504

	except Exception as e:
		print("locals() =", locals())

		logger.error(f"❌ Server error: {e}")

		# 🔥 Full traceback, will show exact source file, line, and code line where the exception ACTUALLY comes from
		exc_type, exc_value, exc_traceback = sys.exc_info()
		traceback.print_exception(exc_type, exc_value, exc_traceback)

		return jsonify({"error": "Internal Server Error"}), 500

def get_rpg_speaker_sender(data):
	"""Handles RPG speaker/sender assignment based on turn-based role system."""
	speaker_role = data.get("speaker_role", "unknown")

	if speaker_role == "npc":
		return {
			"sender_type": "bot",
			"sender_name": data.get("bot", {}).get("name", "Unknown"),
			"speaker_type": "npc",
			"speaker_name": data.get("npc", {}).get("name", "Unknown"),
			"speaker_afk": False
		}
	else:
		return {
			"sender_type": "npc",
			"sender_name": data.get("npc", {}).get("name", "Unknown"),
			"speaker_type": "bot",
			"speaker_name": data.get("bot", {}).get("name", "Unknown"),
			"speaker_afk": data.get("bot", {}).get("afk", False)
		}

if __name__ == "__main__":
	# Start Flask app
	app.run(host="0.0.0.0", port=server_port, debug=True, use_reloader=False)
