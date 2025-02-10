import configparser, re, regex
from collections import defaultdict
import json, time, random
import threading
import logging
from log_utils import debug_print, logger

config = configparser.ConfigParser()
config.read('zyria.conf')

generator_bot_reply_chance			= int(config.get('Generator', 'BotToBotReplyChance', fallback='50'))
generator_expansions				= config.get('Generator', 'Expansions').split(', ')
generator_prompt_reply				= config.get('Generator', 'PromptReply')
generator_prompt_new				= config.get('Generator', 'PromptNew')
generator_prompt_rpg				= config.get('Generator', 'PromptRpg')
generator_prompt_location			= config.get('Generator', 'PromptLocation')
generator_prompt_location_party		= config.get('Generator', 'PromptLocationParty')
generator_prompt_location_nearby	= config.get('Generator', 'PromptLocationNearby')
generator_prompt_location_apart		= config.get('Generator', 'PromptLocationApart')
generator_prompt_channel			= config.get('Generator', 'PromptChannel')
generator_prompt_speaking_to		= config.get('Generator', 'PromptSpeakingTo')
generator_prompt_mentioned			= config.get('Generator', 'PromptMentioned')
generator_prompt_context			= config.get('Generator', 'PromptContext')
generator_max_tokens				= int(config.get('Generator', 'MaxTokens', fallback='50'))
generator_max_group_size			= int(config.get('Generator', 'MaxGroupSize', fallback='5'))
generator_attention_timeout			= int(config.get('Generator', 'AttentionTimeout', fallback='10'))

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

class DefaultDict(dict):
	def __missing__(self, key):
		return f"{{{key}}}"	 # Keeps the placeholder as `{key}`

class Generator:
	def __init__(self, memory_manager, llm_manager):
		self.memory_manager = memory_manager
		self.llm_manager = llm_manager

		# Attention state: bot_name -> {player: str, timestamp: float}
		self.attention_state = defaultdict(dict)
		self.attention_lock = threading.Lock()	# Prevents modification conflicts

		# Load class overrides
		self.class_overrides = load_class_overrides()

	def send_to_llm(self, prompt, speaker, other_names, context, priority, request_id):
		debug_print(f"send_to_llm received other_names: {other_names}", color="yellow")
		request_payload = {
			"prompt_data": {
				"prompt": prompt,
				"speaker": speaker,
				"other_names": other_names,
				"context": context
			},
			"request_id": request_id,
			"priority": priority
		}
		self.llm_manager.queue_request(request_payload)

	def generate(self, data, request_id) -> dict:	# fallback koboldcpp compatible mode
		debug_print("Kobold-compatible generate called", color="red")
		prompt = data.get("prompt", "")
		max_length = data.get("max_length", 100)
		debug_print(f"koboldcpp mode: prompt received - {prompt}", color="cyan")	

		# Fix the function name call
		speaker = self.extract_speaker(prompt)
		assert speaker is not None, "Speaker is unexpectedly None"

		self.send_to_llm(prompt, speaker, [], "", 4, request_id)

	def rpg_generate(self, data, request_id) -> dict:	# bot to NPC interaction
		debug_print("RPG Generate called", color="red")
		sender_type		= data.get("sender", {})
		message_type	= data.get("message_type", {})
		bot_details		= data.get("bot_details", {})
		unit_details	= data.get("unit_details", {})
		expansion		= data.get("expansion", 1)

		message			= data.get("message", "").strip()
		context			= data.get("context", "").strip()
		pre_prompt		= data.get("pre_prompt", "").strip()
		post_prompt		= data.get("post_prompt", "").strip()

		bot_name		= bot_details.get("name", "unknown")
		bot_class		= bot_details.get("class", "unknown")
		bot_guild		= bot_details.get("guild", "unknown")
		bot_subzone		= bot_details.get("subzone", "unknown")
		bot_zone		= bot_details.get("zone", "unknown")
		unit_name		= unit_details.get("name", "unknown")
		unit_subname	= unit_details.get("subname", "")
		unit_type		= unit_details.get("type", "unknown")
		unit_faction	= re.sub(r" generic$", "", unit_details.get("faction", "unknown"))
		
		unit_type = "" if unit_type == "unknown" else unit_type
		unit_faction = "" if re.match(r"^PLAYER, ", unit_faction) else unit_faction

		# Apply override if bot is in the class overrides dictionary
		bot_class = self.class_overrides.get(bot_name, bot_class)

		speaker, other_names = (
			(unit_name, [bot_name]) if sender_type == "npc" else (bot_name, [unit_name])
		)

		# Generate location and guild prompt
		bot_location = self.get_location(bot_subzone, bot_zone)
		bot_guild_prompt = f", a member of the guild {bot_guild}" if bot_guild != "No Guild" and bot_guild != "unknown" else ""
		
		format_data = {
			"bot_name": bot_name,
			"bot_level": bot_details.get("level", "unknown"),
			"bot_gender": bot_details.get("gender", "unknown"),
			"bot_race": bot_details.get("race", "unknown"),
			"bot_class": bot_class,
			"bot_guild": bot_guild_prompt,
			"bot_location": bot_location,
			"unit_name": unit_name,
			"unit_subname": unit_subname,
			"unit_type": unit_type,
			"unit_faction": unit_faction,
			"unit_level": unit_details.get("level", "unknown"),
			"unit_gender": unit_details.get("gender", "unknown"),
			"unit_race": unit_details.get("race", "unknown"),
			"unit_class": unit_details.get("class", "unknown"),
			"expansion": generator_expansions[expansion],
			"max_tokens": generator_max_tokens
		}

		# Get personality if available
		personality = self.memory_manager.get_personality_formatted(bot_name)
		personality_section = f"{bot_name}'s Personality:[{personality}] " if speaker == bot_name and personality else ""

		# Conditionally add sections only if they contain data
		context_section = ""
		if context:
			context = self.filter_and_transform(context, speaker, other_names)
			context = self.remove_duplicate_messages(context)
			context_section = f"Previous conversation:[{context}] "

		prompt = f"Scene:[{generator_prompt_rpg}] {personality_section}Options:[{pre_prompt}] {context_section}-- {post_prompt} {speaker}:"	  
		prompt = prompt.format_map(DefaultDict(format_data))
		prompt = re.sub(r'\s+', ' ', prompt)	# Collapse multiple spaces

		message = self.filter_and_transform(message, speaker, other_names) if message else message

		debug_print(f"generate_rpg sending prompt to LLM - {prompt}", color="cyan")
		self.send_to_llm(prompt, speaker, other_names, context, 3, request_id)

	def dynamic_generate(self, data, request_id) -> dict:
		request_time	= data.get("request_time", time.time())
		thinking_delay	= data.get("delay", 0)

		sender_type		= data.get("sender", {})
		message_type	= data.get("message_type", {})
		bot_details		= data.get("bot_details", {})
		other_details	= data.get("other_details", {})
		expansion		= data.get("expansion", 1)

		channel_name	= data.get("channel", "")
		channel_members = data.get("channel_members", {})
		message			= data.get("message", "").strip()
		context			= data.get("context", "")

		bot_name		= bot_details.get("name", "unknown")
		bot_class		= bot_details.get("class", "unknown")
		bot_guild		= bot_details.get("guild", "unknown")
		bot_subzone		= bot_details.get("subzone", "unknown")
		bot_zone		= bot_details.get("zone", "unknown")
		bot_location	= self.get_location(bot_subzone, bot_zone)

		def get_other_value(key, default="unknown"):
			return None if message_type == "new" else other_details.get(key, default)

		other_name		= get_other_value('name')
		other_class		= get_other_value('class')
		other_guild		= get_other_value('guild')
		other_subzone	= get_other_value('subzone')
		other_zone		= get_other_value('zone')
		other_location	= None if message_type == "new" else self.get_location(other_subzone, other_zone)

		# Apply class overrides from dictionary
		bot_class = self.class_overrides.get(bot_name, bot_class)
		if other_name:	# Ensure other_name is not None before lookup
			other_class = self.class_overrides.get(other_name, other_class)

		in_same_location = bot_location != "unknown" and other_location != "unknown" and bot_location == other_location
		in_guild_chat	= channel_name == "in guild chat"
		in_party_chat	= channel_name == "in party chat"
		debug_print(f"Channel members: {channel_members}", color="yellow")
		other_names		= list(channel_members.keys()) if channel_members else [other_name]
		debug_print(f"Extracted names: {other_names}", color="cyan")

		# Process party members & distances
		is_nearby = False
		if in_party_chat:
			debug_print(f"Party Chat Members for {bot_name}:")
			for member, info in channel_members.items():  # channel_members is always a dict
				distance = info.get("distance", "unknown")
				debug_print(f"- {member}: {distance} units away", color="cyan")
				if member == other_name:
					other_distance = distance
					is_nearby = other_distance < 100  # Assume < 100 means "nearby"

		elif in_guild_chat:
			debug_print(f"Guild Chat Members for {bot_name}:")
			for member, metadata in channel_members.items():
				debug_print(f"- {member}", color="green")	 # No need to access metadata for guild chat

		# Generate guild, location, and speaking prompts
		bot_guild_prompt, other_guild_prompt = self.generate_guild_prompts(bot_guild, other_guild)
		location_prompt	 = self.generate_location_prompt(bot_name, is_nearby or in_same_location, channel_name, message_type)
		channel_prompt	 = self.generate_channel_prompt(bot_name, channel_name, channel_members)
		
		# 🛠 Ensure speaking_prompt is only calculated when needed
		directed_names, mentioned_names = [], []
		speaking_prompt = ""

		process_timestamp = request_time + thinking_delay
		command_response = self.process_command(bot_name, message)	# Process commands
		if not command_response and message_type == "reply":
			directed_names, mentioned_names = self.is_speaking_to_you(
				other_name, bot_name, channel_members, message, process_timestamp
			)
			speaking_prompt = self.generate_speaking_prompt(bot_name, other_name, directed_names, mentioned_names, message_type)

		debug_print(f"Directed names: {directed_names}", color="yellow")
		debug_print(f"Mentioned names: {mentioned_names}", color="yellow")

		# Ensure environment doesn't appear when both parts are empty
		environment = location_prompt + channel_prompt if (location_prompt or channel_prompt) else ""

		# ✅ Build `format_data` BEFORE formatting `environment`
		format_data = {
			"bot_name": bot_name,
			"bot_level": bot_details.get("level", "unknown"),
			"bot_gender": bot_details.get("gender", "unknown"),
			"bot_race": bot_details.get("race", "unknown"),
			"bot_class": bot_class,
			"bot_guild": bot_guild_prompt,
			"bot_location": bot_location,
			"expansion": generator_expansions[expansion],
			"environment": environment,	 # Temporary placeholder (will be updated below)
			"channel_name": channel_name,
			"max_tokens": generator_max_tokens,
		}

		# ✅ Add "other" details if this is a reply
		if message_type == "reply":
			format_data.update({
				"other_name": other_name,
				"other_level": other_details.get("level", "unknown"),
				"other_gender": other_details.get("gender", "unknown"),
				"other_race": other_details.get("race", "unknown"),
				"other_class": other_details.get("class", "unknown"),
				"other_guild": other_guild_prompt,
				"other_location": other_location,
			})

		# ✅ Format `environment` only after `format_data` is ready
		format_data["environment"] = environment.format_map(DefaultDict(format_data))

		# Choose the appropriate base prompt
		prompt = generator_prompt_new if message_type == "new" else generator_prompt_reply
		prompt = prompt.format_map(DefaultDict(format_data))

		# Get personality if available
		personality = self.memory_manager.get_personality_formatted(bot_name)
		personality_section = f"{bot_name}'s Personality:[{personality}] " if personality else ""

		# Conditionally add sections only if they contain data
		context_section = ""
		if context:
			context = self.filter_and_transform(context, bot_name, other_names)
			context = self.remove_duplicate_messages(context)
			context_section = f"Previous conversation:[{context}] "

		filtered_message = self.filter_and_transform(message, bot_name, other_names) if message else message

		# 🛠 Ensure logging happens BEFORE returning
		if message_type == "reply" and self.should_ignore_message(filtered_message, bot_name, directed_names, mentioned_names):
			debug_print(f"Generator: ignored {message_type} message '{message}' from {other_name} to {bot_name}.", color="yellow")
			ignored_payload = {
				"prompt_data": {"ignored": True},	# ✅ Mark request as ignored
				"request_id": request_id,
				"priority": 0,	# Highest priority (remove from queue ASAP)
			}
			self.llm_manager.queue_request(ignored_payload)
			return	# Exit early

		# Build the final prompt dynamically
		final_prompt = f"Instructions:[{prompt}]{personality_section}{context_section}"
		if message_type == "reply":
			final_prompt += f"{speaking_prompt} {other_name}:{filtered_message}"

		final_prompt += f" {bot_name}:"

		priority = self.get_priority(message_type, sender_type, directed_names, mentioned_names)
		debug_print(f"Passing other_names: {other_names} to send_to_llm", color="cyan")
		self.send_to_llm(final_prompt, bot_name, other_names, context, priority, request_id)
	
	# sender_name is other_name. new messages should use other_name not bot_name
	def is_speaking_to_you(self, sender_name, bot_name, channel_members, message, process_timestamp):
		if not channel_members:	 # Open-world scenario
			logger.debug(f"No channel members tracked. Assuming open-world context for sender: {sender_name}")
			return [], []  # Return empty list for assumptive response

		# Convert channel_members keys to a simple lower-cased list
		sender_name_lower = sender_name.lower()	 # Precompute lowercased sender_name
		directed_names = []
		mentioned_names = []

		# Reset attention state and exit if a generic greeting is detected
		if re.search(r"\b(hey|hello|hi|yo)\s+(guys|everyone|all|people|guildies)\b", message, re.IGNORECASE):
			debug_print(f"Generic greeting detected: '{message}'. Clearing attention state and resetting lists.", color="green")

			# Safely clear the attention state for the bot and sender
			if bot_name in self.attention_state:
				if sender_name in self.attention_state[bot_name]:
					# Clear the entire nested attention state for this sender
					self.attention_state[bot_name][sender_name].clear()
					debug_print(f"Cleared attention state for bot: {bot_name}, sender: {sender_name}", color="yellow")

			# Return empty lists for directed and mentioned names
			return [], []

		# Match potential names in the message for direct addressing
		for regex_pattern in [
			(
				r"\b(\w+)(,|\s(what|why|how|where|when|you|you'?re|you'?ll|"
				r"are|do|don'?t|have|haven'?t|go|come|see|run|want|need|think)\b)"
			),
			r"\b(\w+)\b(?:\?|[.?!,]+$)",
			r"\b(hi|hello|hey)\s+(\w+)",  # Match greetings followed by a name
		]:
			match = re.search(regex_pattern, message, re.IGNORECASE)
			if match:
				# Extract the name depending on the regex pattern
				if regex_pattern == r"\b(hi|hello|hey|yo)\s+(\w+)":
					possible_name = match.group(2)	# Name is in the second group for greetings
				else:
					possible_name = match.group(1)	# Name is in the first group for other patterns

				# Resolve the name and add to directed_names
				name = self.full_name(possible_name, channel_members)
				if name and name not in directed_names:
					directed_names.append(name)
					
		# Match names for passive mentions
		for regex_pattern in [
			r"\b(\w+)\b",  # Match any word
		]:
			matches = re.findall(regex_pattern, message, re.IGNORECASE)
			for possible_name in matches:
				name = self.full_name(possible_name, channel_members)
				if name and name not in directed_names and name not in mentioned_names:
					mentioned_names.append(name)

		# Load existing attention state
		old_directed_names = []
		old_mentioned_names = []
		attention_state = self.check_attention(sender_name, bot_name, process_timestamp)
		logger.debug(f"Existing attention for {bot_name}: {attention_state}")

		# Categorize old attention states
		for name, data in attention_state.items():
			attention = data.get("attention")
			logger.debug(f"Name: {name}, Attention Type: {attention}")
			if attention == "directed":
				old_directed_names.append(name)
			elif attention == "mentioned":
				old_mentioned_names.append(name)

		# Update attention states and timestamps
		updated_directed_names = set(directed_names)
		updated_mentioned_names = set(mentioned_names)

		# Refresh timestamps for current attention
		with self.attention_lock:
			for name in updated_mentioned_names:
				self.attention_state[bot_name][sender_name][name] = {
					"attention": "mentioned",
					"timestamp": process_timestamp,
				}
			for name in updated_directed_names:
				logger.debug(f"Updated directed name for {bot_name}: Sender: {sender_name}, Name: {name}, Attention: directed")
				self.attention_state[bot_name][sender_name][name] = {
					"attention": "directed",
					"timestamp": process_timestamp,
				}

		# Merge with old states, avoiding duplicates
		directed_names = list(updated_directed_names | set(old_directed_names))	 # Union of current and old directed
		mentioned_names = list(updated_mentioned_names | set(old_mentioned_names))	# Union of current and old mentioned

		# Remove bot_name from mentioned_names if it exists in both lists
		if bot_name in directed_names and bot_name in mentioned_names:
			debug_print(f"Duplicate name found in both directed and mentioned name lists. Removing from mentioned_names.", color="red")
			mentioned_names.remove(bot_name)

		return directed_names, mentioned_names

	def should_ignore_message(self, message, bot_name, directed_names, mentioned_names):
		debug_print(f"should_ignore_message called", color="white")
		if message.strip() == "":
			debug_print(f"Ignore message: empty string", color="yellow")
			return True

		# Case 1: Directed names exist, bot is not addressed, and mentioned names are empty
		if directed_names and bot_name not in directed_names and not mentioned_names:
			debug_print(f"Ignore message: {bot_name} not addressed ({directed_names}), no mentioned names", color="yellow")
			return True

		# Case 2: Directed names exist, bot is not addressed, and bot is not mentioned	e.g. Bunking hears "Hi Zyria"
		if directed_names and bot_name not in directed_names and mentioned_names and bot_name not in mentioned_names:
			debug_print(f"ignore message: {bot_name} not addressed ({directed_names}) or mentioned ({mentioned_names})", color="yellow")
			return True

		# Case 3: No directed names, but bot is mentioned	e.g. Yvelza hears "Flowerbasket is crazy"
		if not directed_names and mentioned_names and bot_name in mentioned_names:
			debug_print(f"Allow message: no directe names but {bot_name} is mentioned ({mentioned_names})", color="yellow")
			return False

		# Default: Allow the bot to process the message based on reply chance
		chance = random.randint(1, 100) > generator_bot_reply_chance
		if chance:
			debug_print(f"Ignore message: {bot_name} passes check, but fails bot reply chance", color="yellow")
		else:
			debug_print(f"Allow message: directed names = {directed_names} - mentioned names = {mentioned_names}", color="yellow")

		return chance

	def check_attention(self, sender_name, bot_name, process_timestamp):
		with self.attention_lock:
			# Ensure the top-level structure for bot_name exists
			if bot_name not in self.attention_state:
				self.attention_state[bot_name] = {}

			# Ensure the sender_name structure exists under bot_name
			if sender_name not in self.attention_state[bot_name]:
				self.attention_state[bot_name][sender_name] = {}

			# Iterate over the names tracked for this bot and sender
			to_remove = []	# Collect keys to remove if attention is outdated
			for name, attention_data in self.attention_state[bot_name][sender_name].items():
				attention_timestamp = attention_data.get("timestamp", 0)

				debug_print(f"Processing timestamp: {process_timestamp}")
				debug_print(f"Attention timestamp for {name}: {attention_timestamp}")
				debug_print(f"Elapsed time: {process_timestamp - attention_timestamp}")
				debug_print(f"Attention timeout: {generator_attention_timeout}")

				# Clear outdated attention state
				if process_timestamp - attention_timestamp > generator_attention_timeout:
					debug_print(f"Clearing outdated attention state for {name} from {sender_name} to {bot_name}.", color="yellow")
					to_remove.append(name)

			# Remove outdated attention entries
			for name in to_remove:
				del self.attention_state[bot_name][sender_name][name]

			# Return the remaining valid attention state for this bot and sender
			return self.attention_state[bot_name][sender_name]

	def remove_duplicate_messages(self, context):
		"""
		Removes duplicate messages from a long single-line chat log,
		keeping only the last occurrence of each identical message.
		"""
		lines = context.split()	 # Split by whitespace (since there's no newlines)
		seen = set()
		filtered_lines = []

		# Process in reverse to keep only the last occurrence
		for line in reversed(lines):
			if line not in seen:
				seen.add(line)
				filtered_lines.append(line)

		# Reverse back to original order
		return " ".join(reversed(filtered_lines))

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

	def process_context(self, context, speaker, other_names):
		# Abort if context is empty
		if not context or not context.strip():
			debug_print("process_context: context is empty. Skipping.")
			return ""

		context = self.filter_and_transform(context, speaker, other_names)
		return context
	
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

	def split_dialog(self, text):
		pattern = r"(\b[^\s:]+:[^\s:].*?)(?=\s[^\s:]+:|$)"
		return [match.strip() for match in re.findall(pattern, text)]

	def extract_speaker(self, prompt):
		# or rename to extract_speaker_name in your generate() if you prefer
		match = regex.search(r"\s(\w+):$", prompt)
		if match:
			return match.group(1)
		return None

	def full_name(self, shortened, members):
		shortened_lower = shortened.lower()

		# Exclude if the shortened name is less than 2 characters
		if len(shortened_lower) < 2:
			return None

		# Define a set of excluded common words
		excluded_words = {"on", "of", "and", "the", "in", "at", "to", "for", "by", "with"}
		if shortened_lower in excluded_words:
			return None

		# Check if shortened matches the start of any full name
		for full_name in members:
			if full_name.lower().startswith(shortened_lower):
				logger.debug(f"Likely name reference '{shortened_lower}' expanded to '{full_name}'")
				return full_name	# Return the full name if a match is found

		return None

	def get_location(self, subzone, zone):
		if subzone and zone:
			location = f"{subzone}, {zone}"
		elif subzone:
			location = subzone
		elif zone:
			location = zone
		else:
			location = "unknown"

		return location

	def generate_guild_prompts(self, bot_guild, other_guild):
		guild_prompts = {
			"unknown": "",
			"No Guild": " and are not in a guild"
		}

		bot_guild_prompt = guild_prompts.get(bot_guild, f" and are a member of the guild {bot_guild}")

		if other_guild == bot_guild and bot_guild not in {"unknown", "No Guild"}:
			other_guild_prompt = f" and is also a member of {bot_guild}"
		else:
			other_guild_prompt = guild_prompts.get(other_guild, f" and is a member of the guild {other_guild}")

		return bot_guild_prompt, other_guild_prompt

	def generate_location_prompt(self, bot_name, in_same_location, channel_name, message_type):
		if message_type == "new":
			return generator_prompt_location

		location_prompts = {
			(True, "in party chat"): generator_prompt_location_party,
			(True, None): generator_prompt_location_nearby,	 # Any other channel when `in_same_location` is True
			(False, None): generator_prompt_location_apart	# Always used when `in_same_location` is False
		}

		return location_prompts.get((in_same_location, channel_name), generator_prompt_location_apart)

	def generate_channel_prompt(self, bot_name, channel_name, channel_members):
		prompt = ""
		if len(channel_members) > 1:
			channel_members.pop(bot_name, None)	 # Safely remove the bot's name if it exists

			# Insert channel members as english readable list into prompt
			prompt = " " + generator_prompt_channel.format(
				channel_name = channel_name,
				channel_members = self.format_list_to_english(list(channel_members.keys())),
			)

		return prompt

	def generate_speaking_prompt(self, bot_name, other_name, directed_names, mentioned_names, message_type):
		prompt = ""
		if message_type == "new":
			return prompt

		if bot_name in directed_names:
			if len(directed_names) > 1:
				directed_names.remove(bot_name)
				directed_names.append("yourself")
			else:
				directed_names = ["you"]

			# Add speaking-to prompt
			prompt += generator_prompt_speaking_to.format(
				other_name = other_name,
				directed_names = self.format_list_to_english(directed_names),
			)
		elif bot_name in mentioned_names:
			# Add mentioned prompt
			prompt += generator_prompt_mentioned.format(
				other_name = other_name,
				directed_names = self.format_list_to_english(directed_names),
			)
		else:
			# Default to assuming 'you' as the addressed name
			prompt += generator_prompt_speaking_to.format(
				other_name = other_name,
				directed_names = "you",
			)

		return prompt

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

	def convert_placeholders(self, text):
		"""Convert C++ <placeholders> to Python {placeholders}."""
		return re.sub(r"<(.*?)>", lambda m: f"{{{m.group(1).strip().replace(' ', '_')}}}", text)

	def format_list_to_english(self, items):
		"""
		Formats a list of items into a grammatically correct English string.
		
		Args:
			items (list): The list of items to format.

		Returns:
			str: A string with the items in proper English format.
		"""
		if not items:
			return ""  # Return empty string for empty lists
		if len(items) == 1:
			return items[0]	 # Return the only item
		elif len(items) == 2:
			return f"{items[0]} and {items[1]}"	 # Return two items joined with 'and'
		else:
			return f"{', '.join(items[:-1])}, and {items[-1]}"	# Properly join for 3+ items

	def dump_attention_state(self, filepath="attention_state.json"):
		"""
		Saves the current attention state to a JSON file.
		"""
		try:
			with open(filepath, "w") as f:
				json.dump(self.attention_state, f, indent=4)
			logger.debug(f"Attention state saved to {filepath}.")
		except Exception as e:
			logger.error(f"Failed to save attention state: {e}")

