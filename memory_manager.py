import configparser
import os, json, logging
import tempfile
import logging
from log_utils import debug_print, logger

config = configparser.ConfigParser()
config.read('zyria.conf')

memory_dir = config.get('Memory Manager', 'MemoryDir', fallback='memory')

def atomic_save(memory_file, data):
	# Write data to a temporary file first.
	dir_name = os.path.dirname(memory_file)
	with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False, encoding="utf-8") as tf:
		json.dump(data, tf, indent=4)
		temp_name = tf.name
	# Atomically replace the original file with the temporary file.
	os.replace(temp_name, memory_file)

class MemoryManager:
	def __init__(self):
		self.memory_dir = memory_dir
		os.makedirs(self.memory_dir, exist_ok=True)
		self.memories = {}

	def load_memory(self, bot_name):
		if bot_name not in self.memories:
			memory_file = os.path.join(self.memory_dir, f"{bot_name}.json")
			try:
				with open(memory_file, "r", encoding="utf-8") as f:
					content = f.read().strip()
					if not content:
						raise json.JSONDecodeError("Empty file", content, 0)
					self.memories[bot_name] = json.loads(content)
			except (FileNotFoundError, json.JSONDecodeError):
				self.memories[bot_name] = {
					"personality": {
						"backstory": [],
						"pets": [],
						"likes": [],
						"dislikes": []
					},
					"events": [],
					"relationships": {}
				}
			self.save_memory(bot_name)

	def save_memory(self, bot_name):
		"""Save memory to a file."""
		memory_file = os.path.join(self.memory_dir, f"{bot_name}.json")
		if bot_name in self.memories:
			atomic_save(memory_file, self.memories[bot_name])

	def handle_memory_command(self, bot_name, command_type, command_body):
		"""
		Routes memory commands to the correct function.
		- "remember" -> `remember()`
		- "forget" -> `forget()`
		- "list" -> `list_memories()`
		"""

		if command_type == "remember":
			return self.process_remember_command(bot_name, command_body)
		elif command_type == "forget":
			return self.process_forget_command(bot_name, command_body)
		elif command_type == "list":
			return self.list_memories(bot_name, command_body)

		return "I don't understand that memory command."  # Should never reach this

	def process_remember_command(self, bot_name, command_body):
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

				self.remember(bot_name, category, content, relation)
				return response_template.format(content)

		return "I don't understand that memory command."

	def process_forget_command(self, bot_name, command_body):
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

				if self.forget(bot_name, category, content):
					return f"Memory removed: {content} from {category}."
				else:
					return f"I couldn't find {content} in {category}."

		return "I don't understand that forget command."

	def list_memories(self, bot_name, command_body):
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
			data = self.memories.get(bot_name, {}).get("relationships", {}).get(category, [])
		else:
			data = self.memories.get(bot_name, {}).get("personality", {}).get(category, [])

		if not data:
			return f"I don't have any {category} saved."

		return f"{valid_categories[category]}: {', '.join(data)}."

	def update_personality(self, bot_name, key, value):
		"""Update personality traits."""
		self.load_memory(bot_name)
		if key in self.memories[bot_name]["personality"]:
			if isinstance(self.memories[bot_name]["personality"][key], list):
				self.memories[bot_name]["personality"][key].append(value)
			else:
				self.memories[bot_name]["personality"][key] = value
		self.save_memory(bot_name)

	def get_personality(self, bot_name):
		"""Retrieve personality traits."""
		self.load_memory(bot_name)
		return self.memories[bot_name]["personality"]

	def get_personality_formatted(self, bot_name):
		personality = self.get_personality(bot_name)
		if self.is_effectively_empty(personality):
			return ""

		"""Format the personality data into a readable string."""
		formatted = ""

		if "backstory" in personality and personality["backstory"]:
			formatted += "Backstory: "
			formatted += ", ".join(personality["backstory"]) + " - "

		if "pets" in personality and personality["pets"]:
			formatted += "Pets: "
			formatted += ", ".join(personality["pets"]) + " - "

		if "likes" in personality and personality["likes"]:
			formatted += "Likes: "
			formatted += ", ".join(personality["likes"]) + " - "

		if "dislikes" in personality and personality["dislikes"]:
			formatted += "Dislikes: "
			formatted += ", ".join(personality["dislikes"]) + " - "

		# 🆕 Handle relationships dynamically
		if "relationships" in personality and personality["relationships"]:
			for relation_type, people in personality["relationships"].items():
				if people:	# Only include if the list is not empty
					formatted += f"{relation_type.capitalize()}: " + ", ".join(people) + " - "

		formatted = formatted.removesuffix(f" - ")
		return formatted.strip()

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
