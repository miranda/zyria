import threading
import time
import random
import re
from log_utils import debug_print, logger
import json

class AttentionManager:
	def __init__(self, config):
		self.attention_state = {}
		self.attention_lock = threading.Lock()

		self.directed_attention_duration	= int(config.get('Attention Manager', 'DirectedAttentionDuration', fallback='15'))
		self.mentioned_attention_duration	= int(config.get('Attention Manager', 'MentionedAttentionDuration', fallback='10'))
		self.bot_reply_chance				= int(config.get('Attention Manager', 'BotToBotReplyChance', fallback='100'))

	def set_attention(self, sender_name, target_bot_name, llm_channel, attention_type, duration):
		with self.attention_lock:
			sender_attention = self.attention_state.setdefault(sender_name, {})
			bot_attention = sender_attention.setdefault(target_bot_name, {"llm_channel": llm_channel})

			if attention_type == "direct":
				bot_attention["direct_expires"] = time.time() + duration
			elif attention_type == "mention":
				bot_attention["mention_expires"] = time.time() + duration

	def check_attention(self, sender_name, llm_channel):
		direct = []
		mentions = []

		with self.attention_lock:
			if sender_name not in self.attention_state:
				return direct, mentions

			expired_bots = []

			for bot_name, data in self.attention_state[sender_name].items():
				if data.get("llm_channel") != llm_channel:
					continue

				now = time.time()
				direct_active = now < data.get("direct_expires", 0)
				mention_active = now < data.get("mention_expires", 0)

				if not direct_active and not mention_active:
					expired_bots.append(bot_name)
				elif direct_active:
					direct.append(bot_name)
				elif mention_active:
					mentions.append(bot_name)

			for bot_name in expired_bots:
				del self.attention_state[sender_name][bot_name]

			if not self.attention_state[sender_name]:
				del self.attention_state[sender_name]

		return direct, mentions

	def get_message_attention_data(self, sender_name, speaker_name, member_names, message, llm_channel):
		""" Fully modular attention check """

		directed_names, mentioned_names = self.is_speaking_to_you(speaker_name, member_names, message)
		prompt_attention = False

		if speaker_name in directed_names:
			self.set_attention(sender_name, speaker_name, llm_channel, "direct", self.directed_attention_duration)
			prompt_attention = True
		elif speaker_name in mentioned_names:
			self.set_attention(sender_name, speaker_name, llm_channel, "mention", self.mentioned_attention_duration)

		direct, mentions = self.check_attention(sender_name, llm_channel)

		should_ignore = random.randint(1, 100) > self.bot_reply_chance
		if direct:
			should_ignore = speaker_name not in direct	# only direct bots allowed
		if mentions:
			should_ignore = speaker_name not in mentions and random.randint(1, 100) > self.bot_reply_chance

		return should_ignore, prompt_attention

	def is_speaking_to_you(self, speaker_name, member_names, message):
		""" Try to figure out if a speaker is being addressed or mentioned """
		if not member_names: # No way to track, don't filter anyone
			debug_print("No member names, no way to track.", color="cyan")
			return [], []

		# Check for generic greeting, assume true
		if re.search(r"\b(hey|hello|hi|yo)\s+(guys|everyone|all|people|guildies)\b", message, re.IGNORECASE):
			debug_print(f"Generic greeting detected: '{message}'.", color="cyan")
			return [], []  # Return empty directed/mentioned lists meaning don't filter anyone

		directed_names = []
		mentioned_names = []

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
				# Detect if it's a greeting pattern
				if regex_pattern.startswith(r"\b(hi|hello|hey)"):
					possible_name = match.group(2)
				else:
					possible_name = match.group(1)

				# Resolve the name and add to directed_names
				matched_names = self.full_name(possible_name, member_names)
				for name in matched_names:
					if name not in directed_names:
						directed_names.append(name)

		# Match names for passive mentions
		for regex_pattern in [
			r"\b(\w+)\b",  # Match any word
		]:
			matches = re.findall(regex_pattern, message, re.IGNORECASE)
			for possible_name in matches:
				matched_names = self.full_name(possible_name, member_names)
				for name in matched_names:
					if name and name not in directed_names and name not in mentioned_names:
						mentioned_names.append(name)

		debug_print(f"Detected directed names {directed_names} and mentioned names {mentioned_names}", color="yellow")
		return directed_names, mentioned_names

	def full_name(self, shortened, members):
		shortened_lower = shortened.lower()

		# Exclude if the shortened name is less than 2 characters
		if len(shortened_lower) < 2:
			return []

		# Define a set of excluded common words
		excluded_words = {"me", "my", "you", "on", "of", "off", "and", "the", "in", "inn", "at", "to", "for", "by", "buy", "with"}
		if shortened_lower in excluded_words:
			return []

		# Exclude if <=3 characters and ends with a vowel (anti-sloppy match)
		if len(shortened_lower) <= 3 and shortened_lower[-1] in "aeiou":
			return []

		# Match: prioritize exact, then sort partials by length descending
		exact_matches = [full_name for full_name in members if full_name.lower() == shortened_lower]
		partial_matches = [full_name for full_name in members if full_name.lower().startswith(shortened_lower) and full_name.lower() != shortened_lower]

		partial_matches.sort(key=len, reverse=True)	 # Prefer longer matches first

		matches = exact_matches + partial_matches

		for match in matches:
			if match.lower() != shortened_lower:
				debug_print(f"Likely name reference \"{shortened}\" expanded to \"{match}\"", color="magenta")

		return matches	# Always returns a list, possibly empty

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
