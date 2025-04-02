import os
import json
import re
import tempfile
from log_utils import debug_print, logger
import threading

def atomic_save(memory_file, data):
	# Write data to a temporary file first.
	dir_name = os.path.dirname(memory_file)
	with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False, encoding="utf-8") as tf:
		json.dump(data, tf, indent=4)
		temp_name = tf.name
	# Atomically replace the original file with the temporary file.
	os.replace(temp_name, memory_file)

class MemoryManager:
	def __init__(self, config, llm_manager):
		self.llm_manager = llm_manager

		self.memory_dir = config.get('Memory Manager', 'MemoryDir', fallback='memory')
		os.makedirs(self.memory_dir, exist_ok=True)

		self.memories = {}
		self.lock = threading.RLock()  # 🔒 Lock for thread safety

		# Load the race-to-faction mapping
		with open("faction.json", "r") as file:
			self.race_faction_lookup = json.load(file)

	def load_memory(self, name):
		with self.lock:	 # 🔒 Prevent concurrent read/write
			if name not in self.memories:
				memory_file = os.path.join(self.memory_dir, f"{name}.json")
				try:
					with open(memory_file, "r", encoding="utf-8") as f:
						content = f.read().strip()
						if not content:
							raise json.JSONDecodeError("Empty file", content, 0)
						self.memories[name] = json.loads(content)
				except (FileNotFoundError, json.JSONDecodeError):
					self.memories[name] = {
						"personality": {
							"backstory": [],
							"pets": [],
							"likes": [],
							"dislikes": []
						},
						"events": [],
						"relationships": {}
					}

				if not "character" in self.memories[name]:
					self.memories[name]["character"] = {}
				
				self.save_memory(name)

	def save_memory(self, name):
		"""Save memory to a file."""
		with self.lock:	 # 🔒 Ensure only one save happens at a time
			memory_file = os.path.join(self.memory_dir, f"{name}.json")
			if name in self.memories:
				atomic_save(memory_file, self.memories[name])

	def handle_memory_command(self, name, command_type, command_body):
		"""
		Routes memory commands to the correct function.
		- "remember" -> `remember()`
		- "forget" -> `forget()`
		- "list" -> `list_memories()`
		"""

		if command_type == "remember":
			return self.process_remember_command(name, command_body)
		elif command_type == "forget":
			return self.process_forget_command(name, command_body)
		elif command_type == "list":
			return self.list_memories(name, command_body)

		return "I don't understand that memory command."  # Should never reach this

	def process_remember_command(self, name, command_body):
		"""
		Extracts the correct category and adds a new memory.
		"""
		command_patterns = {
			r"^you like (.+)$": ("likes", "Got it! I now remember that I like {}."),
			r"^you dislike (.+)$": ("dislikes", "Understood! I now dislike {}."),
			r"^add to backstory (.+)$": ("backstory", "Backstory updated! {}"),
			r"^your friend is (.+)$": ("relationships", "I’ll remember that {} is my friend.", "friend"),
			r"^your rival is (.+)$": ("relationships", "I now know that {} is my rival.", "rival"),
			r"^you have pet (.+)$": ("pets", "A pet? Nice! I now have {}.")
		}

		for pattern, details in command_patterns.items():
			match = re.match(pattern, command_body, re.IGNORECASE)
			if match:
				content = match.group(1).strip().capitalize()
				category = details[0]
				response_template = details[1]
				relation = details[2] if len(details) > 2 else None

				self.remember(name, category, content, relation)
				return response_template.format(content)

		return "I don't understand that memory command."

	def process_forget_command(self, name, command_body):
		"""
		Parses the forget command and removes the memory.
		Example: "forget: pet luna" will remove "Luna" from pets.
		"""
		command_patterns = {
			r"^like (.+)$": "likes",
			r"^dislike (.+)$": "dislikes",
			r"^backstory (.+)$": "backstory",
			r"^friend (.+)$": "friend",
			r"^rival (.+)$": "rival",
			r"^pet (.+)$": "pets"
		}

		for pattern, category in command_patterns.items():
			match = re.match(pattern, command_body, re.IGNORECASE)
			if match:
				content = match.group(1).strip().capitalize()

				if self.forget(name, category, content):
					return f"Memory removed: {content} from {category}."
				else:
					return f"I couldn't find {content} in {category}."

		return "I don't understand that forget command."

	def list_memories(self, name, command_body):
		"""
		Lists memory in a given category.
		Example: "list: pets" returns all pets.
		"""
		valid_categories = {
			"likes": "Things I like",
			"dislikes": "Things I dislike",
			"backstory": "My backstory",
			"pets": "My pets",
			"friend": "My friends",
			"rival": "My rivals"
		}

		category = command_body.lower()
		if category not in valid_categories:
			return "I don't know that memory category."

		# Retrieve memory
		if category in ["friend", "rival"]:
			data = self.memories.get(name, {}).get("relationships", {}).get(category, [])
		else:
			data = self.memories.get(name, {}).get("personality", {}).get(category, [])

		if not data:
			return f"I don't have any {category} saved."

		return f"{valid_categories[category]}: {', '.join(data)}."

	def get_character_info(self, name):
		"""Retrieve character info."""
		with self.lock:
			self.load_memory(name)
			return self.memories[name]["character"]

	def get_player_status(self, name, member_names):
		"""Determine if a player is present in the chat."""
		return "currently in the chat" if name in member_names else "not present or offline"

	def replace_placeholders(self, text, member_names):
		"""
		Replace {player_info} and {player_status} placeholders in memory text.
		"""
		# Match placeholders in the text
		matches = re.findall(r"(\w+) (\{player_info\}|\{player_status\})", text)

		for char_name, placeholder in matches:
			status = self.get_player_status(char_name, member_names)
			if placeholder == "{player_info}":
				char_info = self.get_character_info(char_name)	# Returns a dict

				# 🔹 Ensure all required fields exist before formatting
				if all(k in char_info for k in ["gender", "race", "spec", "class"]):
					char_info_str = (
						f"{char_name} ({char_info['gender']} {char_info['race']} {char_info['spec']} {char_info['class']} - {status})"
					)
				else:
					char_info_str = char_name  # ✅ Keep the name but remove `{player_info}`

				text = text.replace(f"{char_name} {placeholder}", char_info_str)

			elif placeholder == "{player_status}":
				text = text.replace(f"{char_name} {placeholder}", f"{char_name} ({status})")
				text = re.sub(r'\s+', ' ', text).strip() 
		return text

	def get_processed_memory(self, name, member_names):
		"""Retrieve memory with placeholders replaced."""
		with self.lock:
			self.load_memory(name)
			memory = self.memories[name]

			# Process placeholders in all string fields
			def process_data(data):
				if isinstance(data, str):
					return self.replace_placeholders(data, member_names)
				elif isinstance(data, list):
					return [process_data(item) for item in data]
				elif isinstance(data, dict):
					return {key: process_data(value) for key, value in data.items()}
				return data

			return process_data(memory)

	def update_character_info(self, name, **kwargs):
		"""Completely replace the 'character' dictionary, ensuring it exists and removing missing keys."""
		with self.lock:
			self.load_memory(name)

			# Replace 'character' with a new dictionary containing only kwargs
			self.memories[name]["character"] = {key: value for key, value in kwargs.items() if key != "name"}

			debug_print(f"Replacing {name}'s character data with {self.memories[name]['character']}", color="blue")

			self.save_memory(name)

	def get_formatted_character_info(self, name, member_names, basic=False, location=False, show_personality=False):
		with self.lock:
			self.load_memory(name)
			character = self.memories[name]["character"]
			text = name

			if "level" in character:
				if basic:
					text += (
						f" - {character.get('gender', '')} "
						f"{character.get('race', '')} "
						f"{character.get('class', '')}"
					)
				else:
					faction = self.race_faction_lookup.get(character.get("race", "Unknown"), "Unknown")
					text += (
						f" - A level {character.get('level', '??')}"
						f" {character.get('gender', '')}"
						f" {character.get('race', '')}"
						f" {character.get('spec', '')}"
						f" {character.get('class', 'Unknown')}\n"
						f" * Faction: {faction}\n"
						f" * Guild: {character.get('guild', 'No guild')}\n"
					)

			if show_personality and (personality := self.get_personality_formatted(name, member_names)):
				text += personality

			if location:
				location_text = self.format_location(character.get('zone', ''), character.get('subzone', ''), character.get('continent', ''))
				text += "** {name}'s current location is: {location}.\n" if location_text != "Unknown" else ""

			return text

	def update_speakers_character_info(self, speakers):
		"""Update memory for all speakers with their character details."""
		for speaker in speakers:
			name = speaker.get("name")
			if not name or name == "Unknown":
				continue  # Skip invalid names
			
			# ✅ Loop through all keys in speaker and update memory
			for key, value in speaker.items():
				if key == "name":  # Skip updating the name itself
					continue
				debug_print(f"Updating {key} for {name}")
				self.update_character_info(name, **{key: value})

	def format_location(self, zone, subzone, continent):
		parts = []

		if subzone and zone and zone != "Unknown":
			parts.append(f"{subzone}, {zone}")
		elif zone and zone != "Unknown":
			parts.append(zone)

		if continent and continent != "Unknown":
			if parts:
				parts[-1] += f", in {continent}"
			else:
				parts.append(continent)

		return parts[0] if parts else ""

	def get_personality(self, name, member_names):
		"""Retrieve personality traits with placeholders processed."""
		with self.lock:
			processed_memory = self.get_processed_memory(name, member_names)
			return processed_memory["personality"]

	def get_relationships(self, name, member_names):
		"""Retrieve personality traits with placeholders processed."""
		with self.lock:
			processed_memory = self.get_processed_memory(name, member_names)
			return processed_memory["relationships"]

	def get_personality_formatted(self, name, member_names):
		personality = self.get_personality(name, member_names)
		relationships = self.get_relationships(name, member_names)

		if self.is_effectively_empty(personality):
			return ""

		"""Format the personality data into a readable string."""
		formatted = ""

		if "backstory" in personality and personality["backstory"]:
			formatted += " * Backstory: "
			formatted += ". ".join(personality["backstory"]) + ".\n"

		if "pets" in personality and personality["pets"]:
			formatted += " * Pets: "
			formatted += ", ".join(personality["pets"]) + ".\n"

		if "likes" in personality and personality["likes"]:
			formatted += " * Likes: "
			formatted += ", ".join(personality["likes"]) + ".\n"

		if "dislikes" in personality and personality["dislikes"]:
			formatted += " * Dislikes: "
			formatted += ", ".join(personality["dislikes"]) + ".\n"

		# 🆕 Handle relationships dynamically
		if relationships:
			for relation_type, people in relationships.items():
				if people:
					formatted += f" * {relation_type.capitalize()}: " + ", ".join(people) + "\n"

#		formatted = re.sub(r'[ \t]+', ' ', formatted).strip()
		return formatted

	def update_personality(self, name, key, value):
		"""Update personality traits."""
		with self.lock:
			self.load_memory(name)
			if key in self.memories[name]["personality"]:
				if isinstance(self.memories[name]["personality"][key], list):
					self.memories[name]["personality"][key].append(value)
				else:
					self.memories[name]["personality"][key] = value
			self.save_memory(name)

	def is_effectively_empty(self, memory):
		if not memory:	# If the entire object is empty
			return True
		for key, value in memory.items():
			if isinstance(value, dict):
				if not self.is_effectively_empty(value):	 # Recursively check sub-dictionaries
					return False
			elif isinstance(value, list) or isinstance(value, str):
				if value:  # Check if list or string is non-empty
					return False
		return True
