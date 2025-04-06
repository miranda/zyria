import configparser
import re
from collections import defaultdict
import json
import time
import threading
from log_utils import debug_print, logger
import string

config = configparser.ConfigParser()
config.read(['zyria.conf', 'prompts.conf'])

ATTENTION_TIMEOUT = 10	# seconds

def load_class_overrides():
	"""Loads class overrides from a JSON file."""
	try:
		with open("class_overrides.json", "r") as f:
			return json.load(f)
	except FileNotFoundError:
		return {}  # Return an empty dictionary if file is missing
	except json.JSONDecodeError as e:
		logger.error(f"Error loading class_overrides.json: {e}")
		return {}  # Return empty if JSON is corrupted

def get_placeholders(template):
	return [field_name for _, field_name, _, _ in string.Formatter().parse(template) if field_name]

def render_template(template, data):
	required = get_placeholders(template)
	used = {k: data.get(k, "") for k in required}
	return template.format_map(SafeDict(used))

class SafeDict(dict):
	def __missing__(self, key):
		return f"{{{key}}}"	 # Leaves placeholder as-is if missing

class Generator:
	def __init__(self, config, *, prompt_builder, conversation_manager, memory_manager, npc_manager, context_manager, llm_manager, attention_manager):
		self.config = config
		self.prompt_builder = prompt_builder
		self.conversation_manager = conversation_manager
		self.memory_manager = memory_manager
		self.npc_manager = npc_manager
		self.context_manager = context_manager
		self.llm_manager = llm_manager
		self.attention_manager = attention_manager

		self.batch_cycle_time = float(config.get('Generator', 'BatchCycleTime', fallback='4.0'))

		# Load class overrides
		self.class_overrides = load_class_overrides()

		# ✅ Regular Channel Processing Threads
		self.channel_threads = {}

		# ✅ Separate RPG Processing Thread
		self.rpg_processing_thread = threading.Thread(target=self.rpg_generator_loop, daemon=True)
		self.rpg_processing_thread.start()

	def process_or_start(self, llm_channel):
		if llm_channel.startswith("RPG"):
			# RPG threads are handled separately, do not spawn normal generator threads
			return

		if llm_channel not in self.channel_threads:
			t = threading.Thread(target=self.generator_loop, args=(llm_channel,), daemon=True)
			self.channel_threads[llm_channel] = t
			t.start()
			debug_print(f"Started generator_loop() thread for channel {llm_channel}", color="green")

	def generator_loop(self, llm_channel):
		while True:
			requests, message_type = self.conversation_manager.fetch_pending_requests(llm_channel)
			if not requests:
				time.sleep(0.5)	 # Short nap to avoid busy loop
				continue

			with self.conversation_manager.queues_lock:
				for request in requests:
					request["status"] = "processing"

			if message_type == "new":
				debug_print(f"Processing 'new' message immediately in channel {llm_channel}", color="blue")
				self.batch_generate(requests)
				continue

			# Only stall this channel while waiting for more
			wait_time = max(self.batch_cycle_time - len(requests), 1.0)
			time.sleep(wait_time)

			more_requests, more_message_type = self.conversation_manager.fetch_pending_requests(
				llm_channel
			)

			if more_requests:
				with self.conversation_manager.queues_lock:
					for request in more_requests:
						request["status"] = "processing"
					requests.extend(more_requests)

			self.batch_generate(requests)
			time.sleep(0.1)

	def rpg_generator_loop(self):
		"""Continuously fetches and processes RPG requests every 1 second."""
		while True:
			for llm_channel in self.conversation_manager.get_active_channels():
				if not llm_channel.startswith("RPG"):
					continue  # Skip non-RPG channels

				requests, message_type = self.conversation_manager.fetch_pending_rpg_requests(llm_channel)
				if requests:
					if len(requests) == 1:
						requests[0]["status"] = "processing"

						debug_print(f"Processing RPG request {requests[0]['request_id']}", color="cyan")
						self.rpg_generate(requests)
					else:
						logger.error("❌ ERROR: Attempted to process more than one RPG request simultaneously.")

			time.sleep(1.5)	 # Poll RPG channels every 1.5 seconds

	def extract_details(self, entity_data, entity_type="player"):
		""" Extracts common entity attributes, adding extra fields for players and NPCs. """

		name = entity_data.get("name", "Unknown")

		# ✅ Apply class overrides if the entity is a player/bot
		if entity_type == "player" and name in self.class_overrides:
			override_class = self.class_overrides[name]
			debug_print(f"Player <{name}> found in class overrides, changing class to {override_class}", color="dark_yellow")
			entity_class = override_class
		else:
			entity_class = entity_data.get("class", "Unknown")

		data = {
			"type": entity_data.get("type", "Unknown"),
			"name": name,
			"gender": entity_data.get("gender", "Unknown"),
			"level": entity_data.get("level", "Unknown"),
			"class": entity_class
		}

		if entity_type == "player":	 # Players/bots
			data.update({
				"race": entity_data.get("race", "Unknown"),
				"spec": entity_data.get("spec", ""),
				"guild": entity_data.get("guild", "No Guild"),
				"zone": entity_data.get("zone", "Unknown"),
				"subzone": entity_data.get("subzone", ""),
				"continent": entity_data.get("continent", "Unknown")
			})

		elif entity_type == "npc":	# NPCs have faction and creature type instead
			faction = entity_data.get("faction", "")
			faction = re.sub(r" generic$", "", faction)	 # Remove "generic"
			faction = "" if re.match(r"^PLAYER, ", faction) else faction	
			data.update({
				"faction": faction,
				"creature_type": entity_data.get("creature_type", "Unknown"),
				"creature_subname": entity_data.get("sub_name", "")
			})

		return data

	def batch_generate(self, requests_list):
		"""Generates reply responses for a batch of bot messages"""
		debug_print(f"batch_generate() called with request IDs {[r.get('request_id') for r in requests_list]}", color="cyan")

		if not requests_list:
			logger.warning("batch_generate received an empty request list.")
			return []

		# ✅ Extract first request details (assuming all requests in batch share context)
		first_request = requests_list[0]
		message_type = first_request.get("message_type", "unknown")
		expansion = first_request.get("expansion", 1)  # Default to TBC
		channel_members = first_request.get("channel_members", {})
		llm_channel = first_request.get("llm_channel", None)
		channel_label = first_request.get("channel_label", None)
		member_names = list(channel_members.keys())

		# Update channel members
		self.context_manager.update_channel_members(llm_channel, member_names)

		# ✅ Initialize response mappings
		request_speaker_map = {}  # {request_id -> speaker_name}
		new_messages = set()	 # ✅ Use a set to track unique (sender, message) pairs
		updated_characters = set()	# ✅ Track characters that have already been updated
		attention_prompt_data = defaultdict(list)  # sender_name -> list of speaker_names

		# ✅ Unique collector helpers
		unique_senders = {}
		unique_speakers = {}

		# ✅ Process all requests
		for request in requests_list:
			request_id = request.get("request_id", None)
			message = request.get("message", None)
			sender = request.get("sender", None)
			speaker = request.get("speaker", {})

			if not request_id or not message or not sender or not speaker:
				logger.warning(f"Skipping malformed request: {request}")
				continue

			# ✅ Extract sender details
			sender_data = self.extract_details(sender)
			sender_type = sender_data.get("type", "Unknown")
			sender_update = sender_data.copy()
			sender_name = sender_update.pop("name")

			# ✅ Update sender if needed
			if sender_name not in updated_characters:
				self.memory_manager.update_character_info(sender_name, **sender_update)
				updated_characters.add(sender_name)

			# ✅ Store sender uniquely
			unique_senders[sender_name] = sender_data

			# ✅ Extract speaker details
			speaker_data = self.extract_details(speaker)
			speaker_update = speaker_data.copy()
			speaker_name = speaker_update.pop("name")

			# ✅ If the speaker already has a request, cancel the previous one
			if message_type == "reply":
				for prev_request_id, prev_speaker_name in list(request_speaker_map.items()):
					if prev_speaker_name == speaker_name:
						del request_speaker_map[prev_request_id]
						debug_print(f"Cancelling previous redundant request {request_id} for <{speaker_name}>", color="dark_cyan")
						self.conversation_manager.cancel_request(prev_request_id, speaker_name, llm_channel)
						break

			ignore_votes = []
			for sender_data_entry in unique_senders.values():
				s_name = sender_data_entry["name"]
				s_type = sender_data_entry.get("type", "Unknown")

				should_ignore, prompt_attention = self.attention_manager.get_message_attention_data(
					s_name, speaker_name, member_names, message, llm_channel
				)

				if s_type == "player" and prompt_attention:
					attention_prompt_data[s_name].append(speaker_name)

				ignore_votes.append(should_ignore)

			if all(ignore_votes):
				debug_print(f"Speaker <{speaker_name}> cancelled due to attention check (ignored by all senders)", color="cyan")
				self.conversation_manager.cancel_request(request_id, speaker_name, llm_channel, release=True)
				continue

			# ✅ Store speaker uniquely
			unique_speakers[speaker_name] = speaker_data

			# ✅ Track request to speaker
			request_speaker_map[request_id] = speaker_name

			if speaker_name not in updated_characters:
				self.memory_manager.update_character_info(speaker_name, **speaker_update)
				updated_characters.add(speaker_name)

			# ✅ Track sender-message pair
			if message_type == "reply":
				targets = tuple(attention_prompt_data.get(sender_name, []))
				new_messages.add((sender_type, sender_name, message, targets))

		# ✅ Convert back to lists:
		senders = list(unique_senders.values())
		sender_names = list(unique_senders.keys())
		speakers = list(unique_speakers.values())
		speaker_names = list(unique_speakers.keys())

		# Player messages must be added to the context here or they are lost forever
		for stype, sname, msg, _ in new_messages:
			if stype == "player":
				debug_print(f"Adding player message \"{msg}\" from <{sname}> to context", color="yellow")
				self.context_manager.add_message(llm_channel, sname, msg, 0)

		# Cancel batch if only a single speaker self-reply remains
		if (
			message_type == "reply"
			and len(request_speaker_map) == 1
			and len(senders) == 1
			and speaker_names[0] == sender_names[0]
		):
			request_id = next(iter(request_speaker_map))
			debug_print(f"Cancelling request {request_id} with same sender/speaker to block self-reply", color="dark_yellow")
			self.conversation_manager.cancel_request(request_id, speaker_names[0], llm_channel, release=True)
			request_speaker_map = {}

		if not request_speaker_map:
			debug_print("Final request speaker map is empty, aborting batch", color="red")
			return	# Nothing left to send to LLM
		else:
			debug_print(f"Final request speaker map for senders {sender_names}:\n{json.dumps(request_speaker_map, indent=4)}", color="dark_cyan")

		# Construct string for removing echoes during LLM output parsing
		new_messages_text = "\n".join([f"{sender}: {msg}" for _, sender, msg, _ in new_messages])

		# Build the prompt with PromptBuilder
		prompt = self.prompt_builder.build_prompt(
			llm_channel=llm_channel,
			message_type=message_type,
			senders=senders,
			speakers=speakers,
			member_names=member_names,
			channel_label=channel_label,
			expansion=expansion,
			new_messages=new_messages
		)

		# Construct the payload
		request_payload = {
			"prompt_data": {
				"prompt": prompt,
				"speaker_names": speaker_names,
				"member_names": member_names,
				"llm_channel": llm_channel,
				"new_messages": new_messages_text
			},
			"priority": 1,
			"request_speaker_map": request_speaker_map
		}
		self.llm_manager.queue_request(request_payload)

	def rpg_generate(self, requests_list):
		"""Generates responses for a batch of bot messages"""
		debug_print(f"rpg_generate() called with request IDs {[r.get('request_id') for r in requests_list]}", color="dark_magenta")

		if not requests_list:
			logger.warning("rpg_generate() received an empty request list.")
			return []

		if len(requests_list) > 1:
			logger.warning("rpg_generate() received multiple requests in an RPG batch, rejecting.")
			return []

		# ✅ Extract first request details (assuming only one)
		data = requests_list[0]

		request_id = data.get("request_id", None)
		message_type = data.get("message_type", "unknown")
		speaker_role = data.get("speaker_role", "unknown")
		bot_details = data.get("bot", {})
		npc_details = data.get("npc", {})
		expansion = data.get("expansion", 1)
		channel_members = data.get("channel_members", {})
		rpg_triggers = data.get("rpg_triggers", {})
		chat_topic = data.get("chat_topic", "general")
		llm_channel = data.get("llm_channel", None)
		member_names = list(channel_members.keys())

		bot_data = self.extract_details(bot_details)
		bot_update = bot_data.copy()
		bot_name = bot_update.pop("name")
		self.memory_manager.update_character_info(bot_name, **bot_update)

		npc_data = self.extract_details(npc_details, entity_type="npc")
		npc_name = npc_data["name"]
		npc_role_info = npc_details.get("npc_options", {})

		sender_name, speaker_name = (npc_name, bot_name) if speaker_role == "bot" else (bot_name, npc_name)
		# Store request_id -> speaker_name mapping
		request_speaker_map = {request_id: speaker_name}

		recent_context, _ = self.context_manager.get_context(llm_channel, lines=2, new_messages=[])
		if not recent_context:
			self.context_manager.add_message(llm_channel, npc_name, f"Greetings, {bot_name}.", 0)
			recent_context, _ = self.context_manager.get_context(llm_channel, lines=2, new_messages=[])

		# Build the prompt with PromptBuilder
		prompt = self.prompt_builder.build_rpg_prompt(
			llm_channel=llm_channel,
			speaker_role=speaker_role,
			bot_data=bot_data,
			npc_data=npc_data,
			npc_role_info=npc_role_info,
			rpg_triggers=rpg_triggers,
			chat_topic=chat_topic,
			expansion=expansion
		)

		# Construct the payload
		request_payload = {
			"prompt_data": {
				"prompt": prompt,
				"speaker_names": [(bot_name if speaker_role == "bot" else npc_name)],
				"member_names": member_names,
				"llm_channel": llm_channel,
				"chat_topic": chat_topic,
				"new_messages": recent_context
			},
			"request_speaker_map": request_speaker_map
		}
		self.llm_manager.queue_rpg_request(request_payload)

	def filter_and_transform(self, text, speaker, other_names):
		# helper functions
		def remove_color(match):	# Delete color codes
			if not isinstance(match, str):
				return ""  # Return an empty string for invalid inputs
			return re.sub(r"\|c[0-9A-Fa-f]{2,8}|\|r", "", match)

		def remove_link(match):
			if not isinstance(match, str):
				return ""  # Return an empty string for invalid inputs
			# Step 1: Remove the hyperlink brackets and content
			item = re.sub(r"\|H.+?\|h\[|\].*?\|h", "", match)

			# Step 2: Reformat the item name if it starts with a prefix like "Pattern:", "Plans:", etc.
			prefix_match = re.match(r"^(Pattern|Plans|Design|Schematic|Recipe): (.+)", item)
			if prefix_match:
				prefix = prefix_match.group(1).lower()	# Get the prefix and convert to lowercase
				name = prefix_match.group(2)  # Get the rest of the item name
				item = f"{name} {prefix}"  # Combine name and prefix in the desired order

			return item

		professions = [
			"Alchemy", "Blacksmithing", "Cooking", "Enchanting", "Engineering",
			"First Aid", "Fishing", "Herbalism", "Jewelcrafting", "Leatherworking",
			"Mining", "Skinning", "Tailoring"
		]
		# Build the profession regex dynamically from the list
		professions_pattern = "|".join(professions)

		# 1️⃣ Match profession training messages
		profession_training_regex = rf"(?:\b\w+:)?\s*(?:{professions_pattern})\d+[csg]"

		# 2️⃣ Match total cost messages
		total_cost_regex = r"(?:\b\w+:)?\s*Total cost:\s*(?:\d+[gsc]\s*)+"

		debug_print(f"filter_and_transform INPUT: {text}", color="none")

		matches = re.findall(r"\|c[0-9A-Fa-f]{2,8}.+?\|r", text)
		for match in matches:
			text = text.replace(match, remove_color(match) or "")
		
		matches = re.findall(r"\|H.+?\|h\[.*?\].*?\|h", text)
		for match in matches:
			text = text.replace(match, remove_link(match) or "")

		pattern = r"Traveling \d+y for grinding money to grind mob"
		text = re.sub(pattern, "I'm going to grind mob", text)
		pattern = r"Continuing \d+y for grinding money to grind mob"
		text = re.sub(pattern, "I'm still grinding mob", text)
		pattern = r"^Looting"
		text = re.sub(pattern, "I just got", text)

		text = re.sub(r"(?:\b\w+:)?No where to travel\. Idling a bit\.\s*", "", text)
		text = re.sub(r"(?:\b\w+:)?--- Can learn from [A-Za-z\s]+ ---\s*", "", text)
		text = re.sub(r"(?:\b\w+:)?[A-Za-z\s]+ ---\s*", "", text)
		text = re.sub(profession_training_regex, "", text)
		text = re.sub(total_cost_regex, "", text)

		debug_print(f"filter_and_transform OUTPUT: {text}", color="green")
		return text

	def process_command(self, bot_name, message):
		"""
		Detects memory commands (remember, forget, list) and delegates processing.
		
		- If it's a valid command, calls `memory_manager.handle_memory_command()`.
		- If not a command, returns "" (normal chat processing).
		"""
		message = message.strip()
		bot_name_lower = bot_name.lower()

		# Regex pattern to detect memory commands
		pattern = rf"^{bot_name_lower}\s+(remember|forget|list):\s+(.+)$"
		match = re.match(pattern, message, re.IGNORECASE)

		if not match:
			return ""  # Not a memory command, pass to LLM normally

		command_type = match.group(1).lower()  # "remember", "forget", or "list"
		command_body = match.group(2).strip()  # Everything after "remember:"

		# Delegate command processing to memory_manager
		return self.memory_manager.handle_memory_command(bot_name, command_type, command_body)

	def get_priority(self, message_type, sender, directed_names, mentioned_names):
		# Define base priorities
		base_priority = {
			"reply": 1,	   # Highest priority (responding to a player)
			"new": 3	   # Lowest priority (bot-initiated chat)
		}

		# Start with a base priority based on message type
		priority = base_priority.get(message_type, 2)  # Default to 2 if unknown type

		# Adjust based on sender & mentions
		priority += 1 if sender == "bot" else 0 # Bots are lower priority
		priority += 1 if not directed_names else 0	# If not directed at bot, lower priority
		priority += 1 if not mentioned_names else 0	 # If no mention, lower priority

		return priority
