import time
import re
import threading
from sortedcontainers import SortedDict
from log_utils import debug_print, logger
import atexit
import signal
import json
import os, sys
import random

SAVE_FILE = "context_restore.json"

class ContextManager:
	def __init__(self, config):
		self.config = config

		self.world_expiration_time		= int(config.get('Context Manager', 'WorldExpirationTime', fallback='600'))
		self.party_expiration_time		= int(config.get('Context Manager', 'PartyExpirationTime', fallback='900'))
		self.raid_expiration_time		= int(config.get('Context Manager', 'RaidExpirationTime', fallback='1800'))
		self.guild_expiration_time		= int(config.get('Context Manager', 'GuildExpirationTime', fallback='1800'))
		self.max_messages_per_channel	= int(config.get('Context Manager', 'MaxMessagesPerChannel', fallback='50'))
		self.prune_interval				= int(config.get('Context Manager', 'PruneInterval', fallback='300'))

		self.typing_min_speed				= int(config.get('Context Manager', 'TypingMinSpeed', fallback='50'))
		self.typing_max_speed				= int(config.get('Context Manager', 'TypingMaxSpeed', fallback='150'))
		self.typing_hesitation_chance		= float(config.get('Context Manager', 'TypingHesitationSpeed', fallback='0.05'))
		self.typing_hesitation_multiplier	= float(config.get('Context Manager', 'TypingHesitationMultiplier', fallback='2.0'))
		self.typing_space_multiplier		= float(config.get('Context Manager', 'TypingSpaceMultiplier', fallback='0.5'))
		self.thinking_min_delay				= float(config.get('Context Manager', 'ThinkingMinDelay', fallback='2.0'))
		self.thinking_max_delay				= float(config.get('Context Manager', 'ThinkingMaxDelay', fallback='3.0'))

		self.contexts = {}			# Maps llm_channel -> SortedDict(timestamp -> message)
		self.channel_members = {}	# Maps llm_channel -> set(names)
		self.context_lock = threading.Lock()
		self.restore_context()		# Restore context from last session
		self.prune_context()		# Clear out stale context from last session

		# Start cleanup thread
		self.prune_thread = threading.Thread(target=self._prune_loop, daemon=True)
		self.prune_thread.start()

		# Ensure context is saved on exit
		atexit.register(self.save_context)	# Runs on normal exit

		# Handle termination signals (Ctrl+C, kill command)
		signal.signal(signal.SIGTERM, lambda sig, frame: self.exit_gracefully())
		signal.signal(signal.SIGINT, lambda sig, frame: self.exit_gracefully())

	def _prune_loop(self):
		"""Continuously prunes old messages at intervals."""
		while True:
			time.sleep(self.prune_interval)	 # Sleep before next cleanup
			self.prune_context()

	def get_expiration_time(self, llm_channel):
		expiration_map = {
			"Guild": self.guild_expiration_time,
			"Raid": self.raid_expiration_time,
			"Party": self.party_expiration_time,
		}
		return next((v for k, v in expiration_map.items() if llm_channel.startswith(k)), self.world_expiration_time)

	def add_message(self, llm_channel, speaker_name, message, response_delay):
		"""Adds a message to the conversation history of the LLM channel, avoiding near-duplicate entries."""
		now = time.time()

		if llm_channel not in self.contexts:
			self.contexts[llm_channel] = SortedDict()

		this_delay = 0
		lines = message.split("|")

		with self.context_lock:
			for line in lines:
				next_delay_match = re.search(r"\[DELAY:(\d+)\]", line)
				next_delay = int(next_delay_match.group(1)) / 1000 if next_delay_match else 0

				line = re.sub(r"\[DELAY:\d+\]", "", line).strip()
				timestamp = now + response_delay + this_delay

				# 🔎 Check for near-duplicates within ±1 second
				nearby_start = timestamp - 10
				nearby_end = timestamp + 3

				is_duplicate = any(
					nearby_start <= ts <= nearby_end and entry["name"] == speaker_name and entry["text"] == line
					for ts, entry in self.contexts[llm_channel].items()
				)

				if not is_duplicate:
					self.contexts[llm_channel][timestamp] = {
						"name": speaker_name,
						"text": line
					}
				else:
					debug_print(f"Skipped duplicate message from <{speaker_name}>: '{line}'", color="yellow")

				this_delay += next_delay

	def get_context(self, llm_channel, lines=5, new_messages=[]):
		"""Retrieve the last N messages from the context, ignoring future timestamps and filtering duplicates."""
		if llm_channel not in self.contexts:
			return "", []  # No context available

		now = time.time()

		# Convert new_messages into a set for fast lookups
		new_messages_set = {(sender, msg) for _, sender, msg, _ in new_messages}

		with self.context_lock:
			# Filter out future messages and extract stored name-text pairs
			past_messages = [
				(msg["name"], msg["text"], ts)	# Extract speaker_name, message text, and timestamp
				for ts, msg in self.contexts[llm_channel].items()
				if ts <= now
			]

			# Get the last N messages while filtering out duplicates
			recent_messages = []
			participants = []
			for name, text, timestamp in past_messages[-lines:]:
				if (name, text) in new_messages_set:
					continue  # Skip duplicates

				# Format SYSTEM messages differently
				if name == "SYSTEM":
					time_elapsed = int(now - timestamp)
					if time_elapsed < 60:
						time_ago = "just now"
					elif time_elapsed < 3600:
						time_ago = f"{time_elapsed // 60} mins ago"
					else:
						time_ago = f"{time_elapsed // 3600} hrs ago"

					recent_messages.append(f"[ {text} ({time_ago}) ]")
				else:
					recent_messages.append(f"{name}: {text}")
					if name not in participants:
						participants.append(name)

		return "\n".join(recent_messages), participants	 # Return as a single text block

	def prune_context(self):
		"""Removes old messages from non-permanent channels and trims excess messages from permanent channels."""
		now = time.time()
		with self.context_lock:
			for llm_channel, messages in list(self.contexts.items()):
				# Remove stale messages
				expiration_time = self.get_expiration_time(llm_channel)
				keys_to_remove = [ts for ts in messages if ts < now - expiration_time]
				for key in keys_to_remove:
					del messages[key]
				if keys_to_remove:
					debug_print(f"Removed {len(keys_to_remove)} stale messages from channel {llm_channel}", color="red")

				# Trim old messages if too many exist
				if len(messages) > self.max_messages_per_channel:
					keys_to_remove = list(messages.keys())[:-self.max_messages_per_channel]
					for key in keys_to_remove:
						del messages[key]
					debug_print(f"Pruned context for channel {llm_channel} to {self.max_messages_per_channel} messages", color="blue")

	def update_channel_members(self, llm_channel, new_members):
		"""Updates channel members and logs system messages when players join/leave."""
		if not (llm_channel.startswith("Guild") or llm_channel.startswith("Party") or llm_channel.startswith("Raid")):
			return	# Only applies to Guild, Party, and Raid channels

		now = time.time()
		new_members = set(new_members)	# Convert to a set for easy comparison

		with self.context_lock:
			if llm_channel not in self.contexts:
				self.contexts[llm_channel] = SortedDict()

			# Get old members, default to an empty set if the channel is new
			old_members = self.channel_members.get(llm_channel, set())

			# Determine who joined and who left
			joined = new_members - old_members
			left = old_members - new_members

			# Update stored members
			self.channel_members[llm_channel] = new_members

			# Insert system messages
			collision_offset = 0.0
			for name in joined:
				self.contexts[llm_channel][now + collision_offset] = {
					"name": "SYSTEM",
					"text": f"{name} has entered the chat"
				}
				collision_offset += 0.000001 
				debug_print(f"SYSTEM: {name} joined {llm_channel}", color="yellow")

			for name in left:
				group_type = "the chat"
				if llm_channel.startswith("Party"):
					group_type = "the party"
				elif llm_channel.startswith("Raid"):
					group_type = "the raid group"

				self.contexts[llm_channel][now + collision_offset] = {
					"name": "SYSTEM",
					"text": f"{name} has left {group_type}"
				}
				collision_offset += 0.000001 
				debug_print(f"SYSTEM: {name} left {llm_channel}", color="yellow")

	def save_context(self):
		"""Saves only 'Guild' channel contexts to a JSON file."""
		try:
			with self.context_lock:
				save_data = {
					"saved_at": time.time(),
					"contexts": {channel: list(messages.items()) for channel, messages in self.contexts.items() if channel.startswith("Guild")},
					"channel_members": {channel: list(members) for channel, members in self.channel_members.items() if channel.startswith("Guild")}
				}

			with open(SAVE_FILE, "w") as f:
				json.dump(save_data, f)

			debug_print("Saved Guild channel context to disk.", color="green")

		except Exception as e:
			debug_print(f"ERROR: Failed to save context: {e}", color="red")

	def restore_context(self):
		"""Loads Guild channel contexts and channel members from a JSON file if it exists."""
		if not os.path.exists(SAVE_FILE):
			return

		with open(SAVE_FILE, "r") as f:
			save_data = json.load(f)

		saved_at = save_data.get("saved_at", 0)
		if time.time() - saved_at > self.guild_expiration_time:
			debug_print(f"Saved channel member state expired (>{self.guild_expiration_time / 60} min). Not restoring members.", color="yellow")
			restore_members = False
		else:
			restore_members = True

		with self.context_lock:
			restored_contexts = save_data.get("contexts", {})
			for channel, messages in restored_contexts.items():
				self.contexts[channel] = SortedDict({float(ts): msg for ts, msg in messages})

			if restore_members:
				restored_members = save_data.get("channel_members", {})
				for channel, members in restored_members.items():
					self.channel_members[channel] = set(members)
				debug_print("Restored channel members.", color="green")

	def calculate_typing_delay(self, text, thinking=False):
		"""
		Calculate a humanized typing delay based on text length, random typing speed, and character-level variations.
		"""
		total_delay = random.uniform(self.thinking_min_delay, self.thinking_max_delay) if thinking else 0.0

		for char in text:
			base_delay = random.uniform(self.typing_min_speed, self.typing_max_speed) / 1000.0
			if char == ' ':
				base_delay *= self.typing_space_multiplier
			elif random.random() < self.typing_hesitation_chance:
				base_delay *= self.typing_hesitation_multiplier
			total_delay += base_delay

		return total_delay 

	def exit_gracefully(self):
		"""Handles program exit by saving Guild channel context."""
		debug_print("Shutting down, saving persistent context...", color="red")
		self.prune_context()
		sys.exit(0)	 # More graceful than exit(0)
