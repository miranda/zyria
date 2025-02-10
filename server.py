from flask import Flask, request, jsonify
import configparser
from memory_manager import MemoryManager
from llm_manager import LLMManager
from generator import Generator
import traceback, atexit
import threading
import time, random
from queue import Queue, Empty
import json
import logging
from log_utils import debug_print, logger

# Initialize Flask app
app = Flask(__name__)

# Initialize the request queue
request_queue = Queue(maxsize=100)	# Limit the queue size to prevent flooding

# Track pending responses with a dictionary
pending_requests = {}

def receive_llm_response(request_id, response):
	"""Callback function for LLM responses."""
	if request_id in pending_requests:
		pending_requests[request_id].put(response)	# Send to waiting thread

# Now it's safe to initialize LLMManager
llm_manager = LLMManager(receive_llm_response)

# Initialize the Memory Manager
memory_manager = MemoryManager()
# Initialize the Generator
generator = Generator(memory_manager=memory_manager, llm_manager=llm_manager)

@app.route("/zyria/v1/generate", methods=["POST"])
def generate_request():
	"""Handles incoming chat requests from Mangos."""
	data = request.get_json()
	if not data:
		return jsonify({"error": "Invalid JSON payload"}), 400

	request_id = str(time.time())  # Unique ID for tracking
	pending_requests[request_id] = Queue()	# Create queue for waiting response
	thinking_delay = 0

	# Check if request is for standard or dynamic prompt generation
	if "prompt" in data:
		generator.generate(data, request_id)  # Send to generator, do NOT wait
	else:
		message_type = data.get("message_type", {})
		if message_type == "rpg":
			generator.rpg_generate(data, request_id)  # Same here
		else:
			# Add timestamp to track when the request was queued
			data["request_time"] = time.time()

			# Simulate thinking delay
			thinking_delay = random.uniform(4, 10)  # Random delay
			data["delay"] = thinking_delay

			generator.dynamic_generate(data, request_id)  # Same here

	# Wait for response from llm_manager (non-blocking)
	try:
		response = pending_requests[request_id].get(timeout=60)	 # Wait max 60s
		del pending_requests[request_id]  # Clean up tracking

		time.sleep(thinking_delay)
		json_response = json.dumps(response)
		debug_print(f"Server: sending to json to MaNGOS - {json_response}")
		return jsonify(response)
	except Empty:
		return jsonify({"error": "Server: LLM processing timed out"}), 504

if __name__ == "__main__":
	config = configparser.ConfigParser()
	config.read('zyria.conf')
	port = config.get('Server', 'Port', fallback='5050')

	app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)
