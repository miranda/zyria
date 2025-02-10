from llama_cpp import Llama
import configparser, re, regex
from collections import defaultdict
import json, time, random
import threading
import queue, heapq
import traceback
import logging
from log_utils import debug_print, logger
from rapidfuzz import process, fuzz
from nltk.corpus import words

# Load English words for dictionary check
english_words = set(words.words())

config = configparser.ConfigParser()
config.read('zyria.conf')

llm_model_path		= config.get('LLM Manager', 'ModelPath')
llm_gpu_layers		= int(config.get('LLM Manager', 'GPULayers', fallback='0'))
llm_threads			= int(config.get('LLM Manager', 'Threads', fallback='1'))
llm_max_tokens		= int(config.get('LLM Manager', 'MaxTokens', fallback='60'))
llm_context_tokens	= int(config.get('LLM Manager', 'ContextTokens', fallback='2048'))

class LLMManager:
	def __init__(self, response_callback):
		self.model = Llama(
			model_path=llm_model_path,
			n_gpu_layers=llm_gpu_layers,
			n_threads=llm_threads,
			n_ctx=llm_context_tokens
		)
		self.response_callback = response_callback
		self.queue = queue.PriorityQueue(maxsize=100)
		self.lock = threading.Lock()
		self.bot_last_speak_time = {}  # Tracks last message time for each bot
		self.worker_thread = threading.Thread(target=self.process_queue, daemon=True)
		self.worker_thread.start()

	def queue_request(self, request_payload):
		"""Queues a request and schedules its processing to prevent bots from speaking at the same time."""
		priority = request_payload["priority"]
		request_id = request_payload["request_id"]
		prompt_data = request_payload["prompt_data"]

		# ✅ Handle ignored requests immediately and return
		if prompt_data.get("ignored", False):
			logger.debug(f"LLMManager: Request {request_id} ignored. Returning empty response immediately.")
			ignored_response = {
				"text": "",
				"finish_reason": "stop",
				"prompt_tokens": 0,
				"completion_tokens": 0,
			}
			self.response_callback(request_id, ignored_response)
			return	# ✅ Do not add ignored requests to the queue

		bot_name = prompt_data["speaker"]  # Get the bot name

		with self.lock:
			now = time.time()
			last_speak_time = self.bot_last_speak_time.get(bot_name, now)

			# Calculate delay (minimum of 3 sec, +2 sec for each stacked request)
			base_delay = 3
			additional_delay = 2 * sum(1 for _, _, _, p in self.queue.queue if p["speaker"] == bot_name)
			next_speak_time = max(now, last_speak_time) + base_delay + additional_delay

			self.bot_last_speak_time[bot_name] = next_speak_time  # Update last speak time

			debug_print(f"Scheduling {bot_name} to speak in {next_speak_time - now:.2f}s", color="cyan")

			# Schedule the message at the correct time
			heapq.heappush(self.queue.queue, (next_speak_time, priority, request_id, prompt_data))

	def process_queue(self):
		"""Processes the queue, sending requests to the LLM when their scheduled time arrives."""
		while True:
			with self.lock:
				if not self.queue.queue:
					time.sleep(0.1)
					continue

				now = time.time()
				next_speak_time, priority, request_id, prompt_data = self.queue.queue[0]

				if now >= next_speak_time:
					heapq.heappop(self.queue.queue)	 # Remove from queue
					
					debug_print(f"{prompt_data['speaker']} speaking now: {prompt_data['prompt']}", color="green")

					# Call the LLM and return the result
					result = self.call_llm(prompt_data)
					self.response_callback(request_id, result)

			time.sleep(0.1)	 # Prevent CPU overload

	def call_llm(self, prompt_data):
		"""Generates a response from the LLM"""
		try:
			result = self.model(prompt=prompt_data["prompt"], max_tokens=llm_max_tokens)

			# Extract response text
			result_text = result['choices'][0]['text'] if 'choices' in result and len(result['choices']) > 0 else ""
			debug_print(f"LLMManager: Generated text - {result_text}", color="none")

			return {
				"text": result_text,
				"finish_reason": result["choices"][0].get("finish_reason", "unknown"),
				"prompt_tokens": result.get("usage", {}).get("prompt_tokens", 0),
				"completion_tokens": result.get("usage", {}).get("completion_tokens", 0),
			}

		except Exception as e:
			logger.error(f"LLMManager: Error calling LLM - {e}")
			return {
				"text": "",
				"finish_reason": "error",
				"prompt_tokens": 0,
				"completion_tokens": 0,
			}

	def get_scheduling_offset(self, bot_name):
		"""Returns the total scheduling delay for a bot."""
		return sum(2 for item in self.queue.queue if isinstance(item, tuple) and len(item) == 3 and isinstance(item[2], dict) and item[2].get("speaker") == bot_name)

	def call_llm(self, prompt_data):
		"""Generates a response from the LLM"""
		prompt = prompt_data["prompt"]
		speaker = prompt_data["speaker"]
		other_names = prompt_data.get("other_names", [])
		context = prompt_data.get("context", "")
		debug_print(f"call_llm received other_names: {prompt_data.get('other_names', [])}", color="green")

		# ✅ Dynamically format the context section
		context_section = f"Previous conversation:[{context}] " if context else ""
		formatted_prompt = prompt.format(context_section=context_section)

		debug_print(f"LLMManager: Sending prompt to LLM - {formatted_prompt}", color="blue")

		# Debugging: Check if anything is None
		if prompt is None:
			logger.error("LLMManager: ERROR - prompt is None!")
		if speaker is None:
			logger.error("LLMManager: ERROR - speaker is None!")
		if other_names is None:
			logger.error("LLMManager: ERROR - other_names is None!")
		if context is None:
			logger.error("LLMManager: ERROR - context is None!")

		try:
			result = self.model(prompt=formatted_prompt, max_tokens=llm_max_tokens)

			# Extract response text
			result_text = result['choices'][0]['text'] if 'choices' in result and len(result['choices']) > 0 else ""
			debug_print(f"LLMManager: generated text - {result_text}", color="none")

			# Filter names and dialog 
			result_text = self.filter_text(result_text, speaker, other_names, context)
			result_text = self.trim_to_last_punctuation(result_text)	# Drop incomplete sentences

			if result_text:
				result_text = self.correct_misspelled_names(result_text, other_names)
				result_text = self.insert_split_markers(result_text, random.uniform(25, 50))

			debug_print(f"LLMManager: Final response from {speaker}: '{result_text}'")

			return {
				"text": result_text,
				"finish_reason": result["choices"][0].get("finish_reason", "unknown"),
				"prompt_tokens": result.get("usage", {}).get("prompt_tokens", 0),
				"completion_tokens": result.get("usage", {}).get("completion_tokens", 0),
			}

		except Exception as e:
			logger.error(f"LLMManager: Error calling LLM - {e}")
			return {
				"text": "",
				"finish_reason": "error",
				"prompt_tokens": 0,
				"completion_tokens": 0,
			}

	def filter_text(self, text, speaker, other_names, context=""):
		"""
		Filters out improperly formatted name markers while keeping the intended dialogue.
		Removes repeated name markers and stops processing when an unrelated name appears.
		"""
		if not text or re.fullmatch(r"[^a-zA-Z]+", text):
			return ""

		# === Step 1: Initial Cleanup ===
		text = text.strip()
		text = re.sub(r"</?[^>]+>", "", text)	# Delete hallucination HTML tags
		text = re.sub(r"^[^\w\*]+", "", text)	# Delete leading non-alphanumerics
		text = re.sub(r'\s+', ' ', text)		# Collapse multiple spaces
		text = re.sub(r'\[|\]', "", text)		# Strip goofy brackets
		text = re.sub(r"\*\*", "", text)		# Strip goofy double asterisks
		text = self.clean_single_quotes(text)	# Strip stupid single quotes

		if not text:
			return ""

		# === Step 2: Create regex for name markers ===
		valid_names = [speaker] + other_names
		valid_names_sorted = sorted(valid_names, key=len, reverse=True)
		valid_pattern = r"(?:{})".format("|".join(map(re.escape, valid_names_sorted)))

		# Match valid name markers and guessed names (single words before `:`)
		name_marker_regex = re.compile(
			rf"(?:\b(?P<valid>{valid_pattern}):\s*)"  # Valid name markers (speaker or known names)
			r"|(?:\b(?P<guess>\w+):\s*)"			  # Guessed (single-word) name markers
		)

		# === Step 3: Process Text to Remove Name Markers ===
		filtered_text = []
		last_end = 0
		first_non_speaker_found = False	 # Track when we find a name that isn't the speaker

		for match in name_marker_regex.finditer(text):
			# Collect text between markers
			segment = text[last_end:match.start()].strip()
			if segment:
				filtered_text.append(segment)

			# Get detected name
			marker_name = match.group("valid") if match.group("valid") else match.group("guess")

			# **STOP** processing if the name is NOT the speaker
			if marker_name.lower() != speaker.lower():
				first_non_speaker_found = True
				break  # Discard this marker and everything after

			# Move past the marker
			last_end = match.end()

		# After loop, add any trailing dialogue if we never stopped early
		if not first_non_speaker_found and last_end < len(text):
			filtered_text.append(text[last_end:].strip())

		# === Step 4: Final Cleanup ===
		result = " ".join(filtered_text).strip()
		speaker_lower = speaker.lower()

		# **Extra Fix: Remove trailing name artifacts like `Zyria:Zyria:Zyria:Zyria:`
		result = re.sub(rf"\b{valid_pattern}:\s*", "", result)

		# === Step 5: Special Cases ===

		# Abort if LLM is being weird with things like: [\quote]
		if re.search(r"\[\\.*\]", result):
			logger.debug(f"filter_text: Found malformed content in '{result}'. Discarding.")
			return ""

		# Abort if talking about self in 3rd person
		pattern = rf"^[\W\s]*{re.escape(speaker_lower)}[\W\s]+.*"
		if re.match(pattern, result.lower()):
			logger.debug(f"filter_text: '{speaker} is talking about self in 3rd person. Discarding.")
			return ""

		# Abort if repeating context
		if f"{speaker}:{result}" in context:
			logger.debug(f"filter_text: '{speaker}:{result}' already exists in context. Discarding.")
			return ""
		
		if not self.is_valid_parentheses(result):	# Block corrupted output
			return ""

		result = re.sub(r'\s+', ' ', result)  # Collapse spaces again
		return result.strip()

	def is_valid_parentheses(self, text):
		""" Returns False if the parentheses count is invalid. """
		open_count = text.count("(")
		close_count = text.count(")")
		
		# Must be a matching pair and at most 2 total
		return open_count == close_count and open_count <= 1  

	def trim_to_last_punctuation(self, text):
		"""
		Trims text backwards to the last punctuation (. ! ?) if extra words follow it.
		If no punctuation exists, return an empty string.
		"""
		if not text.strip():  # Handle empty or whitespace-only input
			return ""

		# Find the last occurrence of a sentence-ending punctuation
		match = re.search(r'[.!?](?!.*[.!?])', text)  # Find the LAST terminal punctuation
		if match:
			return text[:match.end()].strip()  # Trim everything after it

		return ""  # No punctuation found, return empty string

	def insert_split_markers(self, text, min_length=100):
		"""
		Iteratively insert a '|' after punctuation ('.', '!', '?') followed by a space,
		but only if at least min_length characters have passed since the last insertion.
		Also, ignore periods that are preceded by another period.
		"""
		i = 0
		char_count = 0	# characters since the last insertion

		while i < len(text) - 1:  # stop at the second-to-last character
			char_count += 1

			# Check for punctuation and a following space
			if char_count >= min_length and text[i] in ".!?" and text[i+1] == " ":
				# If it's a period, ignore it if it's preceded by another period.
				if text[i] == '.' and i > 0 and text[i-1] == '.':
					i += 1
					continue  # skip this punctuation
				
				# Insert the marker in place of the space
				text = text[:i+1] + "|" + text[i+2:]
				char_count = 0	# reset count
				i += 1	# advance to avoid re-checking the same spot
			else:
				i += 1

		return text

	def clean_single_quotes(self, text):
		debug_print(f"clean_single_quotes - Before: {text}", color="red")

		# Normalize curly apostrophes (’ -> ')
		text = text.replace("’", "'")

		# 1) Preserve single quotes **inside words** (e.g., "don't" stays intact)
		preserved = re.sub(r"(\b\w+)'(\w+\b)", r"\1###QUOTE###\2", text)

		# 2) Strip out all remaining single quotes
		no_extras = re.sub(r"'", "", preserved)

		# 3) Restore **preserved** quotes back into words
		result = no_extras.replace("###QUOTE###", "'")

		debug_print(f"clean_single_quotes - After: {result}", color="red")
		return result

	def correct_misspelled_names(self, text, known_names, threshold=80, min_length=0.8):
		"""
		Detects and corrects misspelled names in a given text.
		
		- text: Input string to check.
		- known_names: List of valid character names.
		- threshold: Minimum similarity score (0-100) for a match.
		
		Returns: Corrected text with misspelled names fixed while preserving punctuation.
		"""
		debug_print(f"Looking for misspelled names in \'{text}\' out of {known_names}", color="magenta") 
		words_in_text = re.findall(r"\b\w+(?:'s)?\b|\W+", text)	 # Capture words & keep punctuation separate
		corrected_words = []

		for word in words_in_text:
			# Skip if it's just punctuation
			if re.match(r"^\W+$", word):
				corrected_words.append(word)
				continue

			# Handle possessive names (e.g., "Bunkins's")
			is_possessive = word.endswith("'s")
			base_word = word[:-2] if is_possessive else word  # Remove 's for checking

			# Skip if the base word is a valid dictionary word
			if base_word.lower() in english_words:
				corrected_words.append(word)
				continue

			# Try to find the closest matching name
			match = process.extractOne(base_word, known_names, scorer=fuzz.ratio)

			# If a match is found and is above the threshold, replace it
			if match and match[1] >= threshold and len(base_word) / len(match[0]) >= min_length:
				corrected_name = match[0]  # Use the corrected name
				if is_possessive:
					corrected_name += "'s"	# Restore possessive form
				corrected_words.append(corrected_name)
			else:
				corrected_words.append(word)  # Keep original if no match found

		result =  "".join(corrected_words)	 # Join without adding extra spaces
		debug_print(f"Corrected names result - {result}", color="magenta")

		return result
