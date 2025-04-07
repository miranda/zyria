from llama_cpp import Llama
import re
from collections import OrderedDict
import random
import threading
import queue
import heapq
import traceback
from log_utils import debug_print, logger
from rapidfuzz import process, fuzz
import nltk
from nltk.corpus import words
import unicodedata
import itertools
import time

class LLMManager:
	def __init__(self, config, *, context_manager, response_callback):
		self.config = config
		self.context_manager = context_manager
		self.response_callback = response_callback	# Store callback function

		self.blocked_tokens				= config.get('LLM Manager', 'BlockedTokens', fallback='')
		self.base_tokens_per_speaker	= int(config.get('LLM Manager', 'BaseTokensPerSpeaker', fallback='30'))
		self.tokens_dialog_factor		= float(config.get('LLM Manager', 'TokensDialogFactor', fallback='1.5'))
		self.min_tokens					= int(config.get('LLM Manager', 'MinTokens', fallback='50'))
		self.max_tokens					= int(config.get('LLM Manager', 'MaxTokens', fallback='150'))
		self.rpg_max_tokens				= int(config.get('LLM Manager', 'RpgMaxTokens', fallback='100'))
		self.max_rpg_request_age		= float(config.get('LLM Manager', 'MaxRpgRequetAge', fallback='15.0'))
		self.main_queue_size			= int(config.get('LLM Manager', 'MainQueueSize', fallback='100'))
		self.rpg_queue_size				= int(config.get('LLM Manager', 'RpgQueueSize', fallback='20'))
		
		self.rpg_text_cache = {}	# Text cache for RPG conversations to reduce LLM calls

		self.model = Llama(
			model_path=config.get('LLM Manager', 'ModelPath'),
			n_gpu_layers=int(config.get('LLM Manager', 'GPULayers', fallback='0')),
			n_threads=int(config.get('LLM Manager', 'Threads', fallback='1')),
			n_ctx=int(config.get('LLM Manager', 'ContextTokens', fallback='4096'))
		)

		# Predefined blocked words (these never change)
		self.logit_bias = self.tokenize_blocked_chars(self.blocked_tokens)
			
		# Load English words for dictionary check
		self.english_words = set(words.words())

		# Set a maxsize of 100 for our priority queue.
		self.queue = queue.PriorityQueue(maxsize=self.main_queue_size)	# Main queue
		self.rpg_queue = queue.Queue(maxsize=self.rpg_queue_size)		# RPG queue (FIFO)
		self.queue_lock = threading.Lock()				# for queue management (purge, put)

		self.request_counter = itertools.count()
		self.llm_lock = threading.Lock()	 # for LLM exclusive access

		self.worker_thread = threading.Thread(target=self.process_queue, daemon=True)
		self.worker_thread.start()

		self.rpg_worker_thread = threading.Thread(target=self.process_rpg_queue, daemon=True)
		self.rpg_worker_thread.start()
		self.last_main_process_time = time.time()

	def update_words(self):
		try:
			nltk.data.find('corpora/words')
		except LookupError:
			nltk.download('words')

	def tokenize(self, word):
		if not isinstance(word, bytes):	 # Ensure conversion only if necessary
			word = word.encode("utf-8")
		
		result = self.model.tokenize(word, add_bos=False)
		return result

	def queue_request(self, request_payload):
		"""
		Queues a request for the LLM.
		
		If the queue is full (100 items), it removes all requests with the worst (i.e. highest
		numerical) priority from the queue (calling the callback for each with a 'purged' result).
		If the new request itself is not better than what is already in the queue, it is dropped.
		"""

		priority = request_payload["priority"]
		prompt_data = request_payload["prompt_data"]
		request_speaker_map = request_payload["request_speaker_map"]

		logger.info(f"Queueing request batch {request_speaker_map} - Priority {priority}")

		with self.queue_lock:
			# If the queue is full, purge lowest priority requests.
			if self.queue.qsize() >= self.queue.maxsize:
				# Get current items (thread-safe under lock)
				current_items = list(self.queue.queue)

				# Determine the worst priority in the queue
				worst_priority = max(item[0] for item in current_items)

				if priority < worst_priority:
					# New request is of higher importance → Purge lowest-priority items
					purge_candidates = [item for item in current_items if item[0] == worst_priority]

					for item in purge_candidates:
						try:
							self.queue.queue.remove(item)
							self.queue.task_done()	# ✅ Mark as processed

							_, purged_prompt_data, purged_request_map = item  # ✅ Extract purged data
							logger.debug(f"Purging request (priority {item[0]})")

							# ✅ Call callback with 'purged' result
							self.cancel_response(purged_request_map, finish_reason="purged")

						except ValueError:
							pass  # Item was already removed

					# Re-heapify after manual removals
					heapq.heapify(self.queue.queue)

				else:
					# New request is lower priority → Drop it
					logger.debug(f"Dropping request (priority {priority}) because queue is full")
					self.cancel_response(request_speaker_map, finish_reason="dropped")
					return	# ✅ Do not add this request

			# ✅ Add new request to queue
			self.queue.put((priority, next(self.request_counter), prompt_data, request_speaker_map))

	def queue_rpg_request(self, request_payload):
		"""Queues an RPG request, handles full queue with proper cancellation."""
		prompt_data = request_payload["prompt_data"]
		request_speaker_map = request_payload["request_speaker_map"]

		if self.rpg_queue.full():
			logger.debug("RPG queue full, cancelling request cleanly.")
			self.cancel_response(request_speaker_map, finish_reason="rpg_queue_full")
			return

		logger.info(f"Queueing RPG request batch {request_speaker_map}")
		self.rpg_queue.put(request_payload)

	def process_queue(self):
		"""Processes the queue, sending requests to the LLM"""
		while True:
			time.sleep(0.1)			
			try:
				queue_size = self.queue.qsize()
				if queue_size > 0:
					debug_print(f"Queue length: {queue_size} pending requests.", color="yellow")

				priority, batch_id, prompt_data, request_speaker_map = self.queue.get()
				debug_print(f"Processing request batch {request_speaker_map} - Priority {priority}")

				start_time = time.time()
				self.last_main_process_time = start_time
				with self.llm_lock:
					response_dict = self.call_llm(prompt_data, request_speaker_map)
				elapsed = time.time() - start_time

				debug_print(f"LLM inference completed in {elapsed:.2f} seconds.", color="dark_green")

				# Send back responses to server via callback
				self.response_callback(response_dict)

				# Mark task as completed
				self.queue.task_done()

			except Exception as e:
				logger.error(f"Error processing queue - {e}")
				logger.error(traceback.format_exc())  # ✅ Print full stack trace

	def process_rpg_queue(self):
		"""Processes the RPG queue in FIFO order, skips old requests based on time_received."""
		while True:
			queue_size_check = self.queue.qsize()
			time.sleep(5.0)
			time_since_main_process = time.time() - self.last_main_process_time
			try:
				# Skip RPG processing if the main queue is still busy
				if (queue_size_check > 0 or self.queue.qsize() > 0 or time_since_main_process < 5.0):
					continue  # Main queue takes priority

				if not self.rpg_queue.empty():
					queue_size = self.rpg_queue.qsize()
					debug_print(f"RPG queue length: {queue_size} pending requests.", color="yellow")

				request_payload = self.rpg_queue.get()
				request_speaker_map = request_payload["request_speaker_map"]
				time_received = request_payload.get("time_received", time.time())

				if time.time() - time_received > self.max_rpg_request_age:
					logger.debug("RPG request too old, cancelling cleanly.")
					self.cancel_response(request_speaker_map, finish_reason="rpg_request_stale")
					self.rpg_queue.task_done()
					continue

				prompt_data = request_payload["prompt_data"]

				debug_print(f"Processing RPG request batch {request_speaker_map}")

				start_time = time.time()
				with self.llm_lock:
					response_dict = self.call_llm(prompt_data, request_speaker_map)
				elapsed = time.time() - start_time

				debug_print(f"RPG LLM inference completed in {elapsed:.2f} seconds.", color="dark_green")

				self.response_callback(response_dict)
				self.rpg_queue.task_done()

			except Exception as e:
				logger.error(f"Error processing RPG queue - {e}")
				logger.error(traceback.format_exc())

	def cancel_response(self, request_speaker_map, finish_reason="stop"):
		"""
		Cancels a response and returns an empty response dictionary.
		This is used when a request is dropped or purged.
		"""

		response_dict = {}

		for request_id in request_speaker_map:
			response_dict[request_id] = {
				"text": "",
				"finish_reason": finish_reason,
				"prompt_tokens": 0,
				"completion_tokens": 0,
			}

		return response_dict  # Return correctly structured response dictionary

	def estimate_tokens(self, num_speakers): 
		"""
		Estimate token budget based on number of speakers and expected dialog complexity.
		- base_per_speaker = base tokens assuming each speaker says something.
		- dialog_factor = average number of utterances per speaker (1.0 = 1 line each, 1.5 = about half speak twice).
		"""
		estimate = int(num_speakers * self.base_tokens_per_speaker * self.tokens_dialog_factor)
		return max(self.min_tokens, min(estimate, self.max_tokens))

	def is_rpg_cache_waiting(self, llm_channel):
		"""Check if cached RPG text is waiting to be used."""
		if llm_channel.startswith("RPG") and llm_channel in self.rpg_text_cache:
			cached_text = self.rpg_text_cache.get(llm_channel, "")
			if cached_text:
				return True

		return False

	def call_llm(self, prompt_data, request_speaker_map):
		"""Generates responses from the LLM for multiple speakers in a batch."""

		prompt = prompt_data["prompt"]
		message_type = prompt_data["message_type"]
		llm_channel = prompt_data["llm_channel"]
		member_names = prompt_data["member_names"]
		num_speakers = len(request_speaker_map)
		if llm_channel.startswith("RPG"):
			max_tokens = self.min_tokens if prompt_data.get("chat_topic", "") == "goodbye" else self.rpg_max_tokens
		else:
			max_tokens = self.estimate_tokens(num_speakers)

		try:
			# Call LLM with the formatted prompt
			result = self.model(
				prompt=prompt,
				max_tokens=max_tokens,
				logit_bias=self.logit_bias,
#					temperature=0.6,		 # Lower randomness
#					top_k=40,				 # Limits word choices
#					top_p=0.8,				 # Probability mass filtering
#					repeat_penalty=1.2,
			)

			choice = result['choices'][0] if 'choices' in result and result['choices'] else {}
			finish_reason = choice.get('finish_reason', '')
			completion_tokens = result.get('usage', {}).get('completion_tokens', 0)

			# ✅ Extract response text
			raw_output = choice.get('text', "")
			logger.info(f"Generated response for {num_speakers} speakers, Max tokens {max_tokens}")
			debug_print(f"Raw LLM generated output:\n{raw_output}", color="blue")

			# Process batch response for multiple speakers
			dialogues, speaker_order = self.parse_batch_text(raw_output, prompt_data, request_speaker_map)
			final_responses = self.apply_delays_to_dialogues(dialogues, speaker_order, request_speaker_map, message_type)
			
			response_dict = {}

			for request_id, response_data in final_responses.items():  # Unpack request_id and data
				response_text = response_data.get("speaker_response", "")
				response_delay = response_data.get("response_delay", 0)
				speaker_name = response_data.get("speaker_name", "Unknown")	 # Now correctly extracted
				if response_text:
					debug_print(f"Parsed valid response for <{speaker_name}>: ", end="")
				else:
					debug_print(f"Returned empty response for <{speaker_name}>: ", end="")
				debug_print(f"\"{response_text}\"", color="cyan", quiet=True)

				response_dict[request_id] = {
					"mangos_response": {
						"text": response_text,
						"finish_reason": result["choices"][0].get("finish_reason", "unknown"),
						"prompt_tokens": result.get("usage", {}).get("prompt_tokens", 0),
						"completion_tokens": result.get("usage", {}).get("completion_tokens", 0),
					},
					"speaker_name": speaker_name,
					"llm_channel": llm_channel,
					"response_delay": response_delay,
					"conversation_members": list(request_speaker_map.values())	# all speakers involved in the LLM response
				}

			return response_dict  # Now returns {request_id: {mangos_response + delay}}

		except Exception as e:
			logger.error(f"Error calling LLM - {e}")
			return {
				request_id: {
					"mangos_response": {
						"text": "",
						"finish_reason": "error",
						"prompt_tokens": 0,
						"completion_tokens": 0,
					},
					"speaker_name": request_speaker_map.get(request_id, "Unknown"),
					"llm_channel": llm_channel,
					"response_delay": 0	# Default delays to 0 in case of an error
				} for request_id in request_speaker_map.keys()
			}  # ✅ Return structured empty responses for all

	def clean_raw_output(self, raw_output):
		# Remove any hidden characters
		cleaned_output = re.sub(r'[^\x20-\x7E\n]+', '', raw_output)

		# Remove LLM formatting lines starting with ### or ##
		cleaned_output = re.sub(r'^\s*#+\s*\w.*$', '', cleaned_output, flags=re.MULTILINE)

		# Remove lines like "Reply 1:" or "Bradpittlord:" with no content
		cleaned_output = re.sub(r'^\s*[\w ]+:\s*$', '', cleaned_output, flags=re.MULTILINE)

		# Remove extra blank lines (e.g., from stripped speaker-only lines)
		cleaned_output = re.sub(r'\n{2,}', '\n', cleaned_output)

		# Remove lines with only non-alphanumeric characters
		cleaned_output = re.sub(r'^\s*[^\w\n]+\s*$\n?', '', cleaned_output, flags=re.MULTILINE)

		# Normalize unicode (like smart quotes, em dashes, etc.)
		cleaned_output = unicodedata.normalize("NFKC", cleaned_output).strip()

		# Collapse multiple newlines into one
		cleaned_output = re.sub(r'\n+', '\n', cleaned_output).strip()

		return cleaned_output

	def strip_outer_quotes(self, text):
		if not text:
			return text

		# Remove exactly one leading quote if present
		if text.startswith('"'):
			text = text[1:]

		# Remove exactly one trailing quote if present
		if text.endswith('"'):
			text = text[:-1]

		return text

	def remove_echoes(self, raw_output, context_lines):
		"""
		Removes echoed prefixes from raw_output lines if they start with a dialog from context.
		"""
		# Automatically split context_lines if given as a string
		if isinstance(context_lines, str):
			context_lines = context_lines.strip().split("\n")

		def extract_dialog(line):
			parts = line.split(":", 1)
			return parts[1].strip() if len(parts) == 2 else ""

		context_dialogs = set(extract_dialog(line) for line in context_lines if ":" in line)

		output_lines = raw_output.strip().split('\n')
		filtered_output = []
		removed_echoes = []

		for line in output_lines:
			if ":" not in line:
				filtered_output.append(line)
				continue

			speaker, dialog = line.split(":", 1)
			dialog = dialog.strip()

			# Remove prefix if matching any context line
			removed = False
			for ctx_dialog in context_dialogs:
				if dialog.startswith(ctx_dialog):
					removed_echoes.append(line)
					dialog = dialog[len(ctx_dialog):].lstrip()
					removed = True
					break

			# Only keep non-empty dialog
			if dialog:
				filtered_output.append(f"{speaker}: {dialog}")

		if removed_echoes:
			debug_print("Removed echoes:", color="red")
			for echo in removed_echoes:
				debug_print(f" - {echo}", color="red")

		return '\n'.join(filtered_output)

	def parse_batch_text(self, raw_output, prompt_data, request_speaker_map):
		"""Extracts and filters bot responses, ensuring structured and valid output."""

		def append_segment(dialogues, speaker, segment):
			"""Append text to a speaker's dialogue, ensuring no duplicate segments."""

			segment = self.strip_outer_quotes(segment)
			
			if speaker not in dialogues:
				dialogues[speaker] = []	 # Store as a list, NOT a string

			# Prevent duplicate entries
			if not dialogues[speaker] or dialogues[speaker][-1] != segment:
				dialogues[speaker].append(segment)

		raw_output = self.clean_raw_output(raw_output)

		if not raw_output:
			debug_print("LLM Mansger: No raw_output, nothing to parse. Aborting.", color="red")
			return [], []	# Return empty responses if no text

		# Extract required data
		message_type = prompt_data.get("message_type", "Unknown")
		speaker_names = prompt_data.get("speaker_names", [])
		member_names = prompt_data.get("member_names", [])
		llm_channel = prompt_data.get("llm_channel", [])
		new_messages_text = prompt_data.get("new_messages", "")

		# Create regex for name markers
		valid_speaker_pattern = "|".join(map(re.escape, sorted(set(speaker_names), key=len, reverse=True)))
		valid_member_pattern  = "|".join(map(re.escape, sorted(set(member_names), key=len, reverse=True)))

		# Match name markers properly:
		name_marker_regex = re.compile(
			rf"(?:\b(?P<valid>{valid_speaker_pattern}):\s*)"	# valid speakers anywhere
			rf"|(?:\b(?P<guess>{valid_member_pattern}):\s*)"	# valid member names anywhere
			rf"|(?:^\s*(?P<multi>\w+\s+\w+):\s*)"				# unknown multi-word markers ONLY if exactly 2 words at line start
			rf"|(?:^\s*(?P<single>\w+):\s*)",					# unknown single-word markers only at line start
			re.MULTILINE
		)

		first_line = raw_output.strip().split("\n")[0]
		failed_start = False
		if not name_marker_regex.match(first_line):
			if len(speaker_names) == 1 or (len(speaker_names) == 2 and message_type == "rpg"):
				# Safe case: only one speaker, or two for RPG prompt
				raw_output = f"{speaker_names[0]}: {raw_output.strip()}"
			else:
				# Ambiguous case: multiple speakers but no valid marker, store result for checking after removing echoes
				failed_start = True

		raw_output = self.remove_echoes(raw_output, new_messages_text)

		if failed_start:
			first_line = raw_output.strip().split("\n")[0]
			if not name_marker_regex.match(first_line):
				debug_print("Output rejected - Missing speaker marker in multi-speaker context.", color="red")
				return [], []  # Invalid output

		# Prune invalid output from the bottom up
		lines = raw_output.strip().split("\n")
		last_valid_idx = None

		for i in reversed(range(len(lines))):
			if name_marker_regex.match(lines[i]):
				last_valid_idx = i
				break

		if last_valid_idx is not None:
			lines = lines[:last_valid_idx + 1]
			raw_output = "\n".join(lines)

		raw_output = self.trim_to_last_punctuation(raw_output).strip()
		raw_output = self.correct_misspelled_names(raw_output, member_names)
		debug_print("Corrected raw output for parsing:")
		debug_print(raw_output, color="dark_cyan", quiet=True)

		# Process Text & Extract Dialogues ===
		dialogues = OrderedDict()
		speaker_order = []
		current_speaker = None
		last_end = 0

		for match in name_marker_regex.finditer(raw_output):
			debug_print(f"Matched name marker [{match.group(0)}] at position {match.start()} - {match.end()}", color="yellow")
			segment = raw_output[last_end:match.start()].strip()

			if segment and current_speaker:
				append_segment(dialogues, current_speaker, segment)

			marker_name = (
				match.group("valid")
				or match.group("guess")
				or match.group("multi")
				or match.group("single")
			)

			if marker_name not in speaker_names:
				debug_print(f"Stopping processing - LLM attempted to speak as <{marker_name}>", color="red")

				last_end = len(raw_output)	# Prevent the trailing capture from reprocessing text.
				break  # Stop parsing invalid names

			speaker_order.append(marker_name)

			current_speaker = marker_name
			last_end = match.end()

		if current_speaker and last_end < len(raw_output):
			append_segment(dialogues, current_speaker, raw_output[last_end:].strip())

		return dialogues, speaker_order

	def apply_delays_to_dialogues(self, dialogues, speaker_order, request_speaker_map, message_type):
		"""Applies response delays and embedded delay tags to dialogues."""
		speaker_names = list(request_speaker_map.values())
		response_delays = {}

		# 1. Build a global schedule: each entry represents one dialogue segment.
		# We'll replace each dialogue line with its split segments.
		global_schedule = []  # Each entry: {"speaker": speaker, "line": segment}
		speaker_counter = {speaker: 0 for speaker in speaker_names}

		for speaker in speaker_order:
			# Only process if this speaker has an available dialogue line.
			if speaker in dialogues and speaker_counter[speaker] < len(dialogues[speaker]):
				original_line = dialogues[speaker][speaker_counter[speaker]]
				if original_line.strip():
					# Use a random min_length parameter as before.
					min_length = random.uniform(25, 50)
					processed_line = self.insert_split_markers(original_line, min_length)
					# Split the processed line on the marker. We expect markers to be '|'
					segments = processed_line.split("|")
					# Add each non-empty segment as its own entry.
					for segment in segments:
						seg = segment.strip()
						if seg:
							global_schedule.append({
								"speaker": speaker,
								"line": seg
							})
				speaker_counter[speaker] += 1

		# 2. Compute a global timestamp for each dialogue segment.
		# Compute first line typing delay
		first_line_typing_delay = (
			self.context_manager.calculate_typing_delay(global_schedule[0]["line"], thinking=True)
			if (global_schedule and message_type != "rpg")
			else 0.0
		)
		global_time = first_line_typing_delay  # Shift everything forward by first line's delay

		for i, entry in enumerate(global_schedule):
			if i == 0:
				entry["global_time"] = first_line_typing_delay
			else:
				gap = self.context_manager.calculate_typing_delay(entry["line"], thinking=True)
				entry["global_time"] = global_schedule[i - 1]["global_time"] + gap

		# 3. For each speaker, collect the global times for their segments.
		speaker_times = {speaker: [] for speaker in speaker_names}
		for entry in global_schedule:
			speaker_times[entry["speaker"]].append(entry["global_time"])

		# 4. Now determine, per speaker, the delay tag for each segment.
		# The rule: For a given segment, if the speaker appears again later, its delay tag is the difference
		# between the next segment's global time and this segment's global time. Otherwise, delay is 0.
		modified_dialogues = {speaker: [] for speaker in speaker_names}
		speaker_response_delay = {}

		# Build a per-speaker list of entries (from the global schedule) to compute delays.
		per_speaker_entries = {speaker: [] for speaker in speaker_names}
		for entry in global_schedule:
			sp = entry["speaker"]
			per_speaker_entries[sp].append({
				"line": entry["line"],
				"global_time": entry["global_time"]
			})

		for speaker, entries in per_speaker_entries.items():
			for i, data in enumerate(entries):
				if i < len(entries) - 1:
					delay_tag = entries[i + 1]["global_time"] - data["global_time"]
				else:
					delay_tag = 0.0
				# Append the delay tag (in ms) to the segment.
				modified_line = data["line"] + f"[DELAY:{int(delay_tag * 1000)}]"
				modified_dialogues[speaker].append(modified_line)
			# The initial delay for the speaker is the global time of their first segment.
			if entries:
				speaker_response_delay[speaker] = entries[0]["global_time"]
			else:
				speaker_response_delay[speaker] = 0.0

		final_responses = {
			request_id: {
				"speaker_name": request_speaker_map[request_id],
				"speaker_response": "|".join(modified_dialogues.get(request_speaker_map[request_id], [])),
				"response_delay": speaker_response_delay.get(request_speaker_map[request_id], 0.0)
			}
			for request_id in request_speaker_map.keys()
		}

		return final_responses

	def trim_to_last_punctuation(self, text):
		"""
		Trims text backwards from the end to the first encountered terminating punctuation 
		(".", "!", "?") or newline, whichever comes first.
		
		Returns the substring up to and including that punctuation/newline.
		If no such character is found, returns an empty string.
		"""
		text = text.rstrip()  # Remove trailing whitespace but not internal newlines
		if not text:
			return ""
		
		# Iterate backwards through the text
		for i in range(len(text) - 1, -1, -1):
			if text[i] in ".!?\n":
				# Return text up to and including this character
				return text[:i + 1]
		
		# No terminal punctuation or newline found
		return ""

	def insert_soft_split(self, text, max_length=200):
		if len(text) <= max_length:
			return text

		# Find the last comma before the limit
		split_index = text.rfind(',', 0, max_length)

		if split_index == -1:
			# No soft split found; return original unmodified
			return text

		# Construct the new string with a visual split marker
		return f"{text[:split_index].strip()}|...{text[split_index + 1:].strip()}"

	def insert_split_markers(self, text, min_length=100):
		"""
		Iteratively insert a '|' after punctuation ('.', '!', '?') followed by a space,
		but only if at least min_length characters have passed since the last insertion.
		Also, ignore periods that are preceded by another period.
		"""

		abbreviations = {"vs.", "Mr.", "Ms.", "Mrs.", "Dr.", "Prof.", "Sr.", "Jr.", "e.g.", "i.e.", "etc."}

		ideal_min_length = min_length
		total_length = len(text)

		if total_length > 200:
			if not re.findall(r'[.!?] ', text):
				text = self.insert_soft_split(text)
			else:
				min_length = 20	 # Less restrictive if length exceeds WoW client limitation

		i = 0
		char_count = 0	# characters since the last insertion

		while i < len(text) - 1:  # stop at the second-to-last character
			char_count += 1

			# Check for punctuation and a following space
			if char_count >= min_length and text[i] in ".!?" and text[i+1] == " ":

				# Check for matching abbreviation
				for abbr in abbreviations:
					abbr_len = len(abbr)
					if i + 1 >= abbr_len and text[i - abbr_len + 1:i + 1].lower() == abbr.lower():
						break  # skip, it's an abbreviation

				else:
					# If it's a period, ignore it if it's preceded by another period,
					# unless the total lengthl exceeds WoW client limit of 200 characters.
					if text[i] == '.' and i > 0 and text[i-1] == '.' and len(text) < 200:
						i += 1
						continue  # skip this punctuation
					
					# Insert the marker in place of the space
					text = text[:i+1] + "|" + text[i+2:]

					if (total_length - char_count) < 200:
						min_length = ideal_min_length

					char_count = 0	# reset count

			i += 1	# advance to avoid re-checking the same spot

		return text

	def correct_misspelled_names(self, text, known_names, threshold=80, min_length=0.8):
		"""
		Detects and corrects misspelled names in a given text.
		
		- text: Input string to check.
		- known_names: List of valid character names.
		- threshold: Minimum similarity score (0-100) for a match.
		
		Returns: Corrected text with misspelled names fixed while preserving punctuation.
		"""
		debug_print(f"Looking for misspelled names out of {known_names}", color="magenta") 
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
			if base_word.lower() in self.english_words:
				corrected_words.append(word)
				continue

			# Try to find the closest matching name
			match = process.extractOne(base_word, known_names, scorer=fuzz.ratio)

			# If a match is found and is above the threshold, replace it
			if match and match[1] >= threshold and len(base_word) / len(match[0]) >= min_length:
				corrected_name = match[0]  # Use the corrected name
				if corrected_name != base_word:
					debug_print(f"Corrected misspelled name \"{base_word}\" to <{corrected_name}>", color="magenta")
				if is_possessive:
					corrected_name += "'s"	# Restore possessive form
				corrected_words.append(corrected_name)
			else:
				corrected_words.append(word)  # Keep original if no match found

		result =  "".join(corrected_words)	 # Join without adding extra spaces

		return result

	def tokenize_blocked_chars(self, blocked_chars):
		logit_bias = {}
		
		# Known problem sequences to check
		additional_strings = [
			"\\'", "\\\\'",	"\\\\\\'",		# Variations of single quote escape
			"\\\"", "\\\\\"", "\\\\\\\"",	# Variations of double quote escape
			"\\", "\\\\", "\\\\\\\\",		# Variations of backslash escape
			"\\n", "\\\n", "\\\\n",			# Variations of escaped newlines
			"\'\'", "\'\'\'", "\'\'\'\'",
			" :", " : ", "```", "````", "`````",			# You just never know
			"///", "////", "/////", "//////", "///////",	# what these psychos
			"***", "****", "*****", "******", "*******",	# will try next.
			"###", "####", "#####", "######", "#######"
		]
		
		# Combine blocked characters and additional problem strings
		expanded_chars = set(blocked_chars).union(additional_strings)

		for char in expanded_chars:
			variations = [
				char,			# Single occurrence
				char * 2,		# Double occurrence
				f" {char}",		# Preceded by space
				f"{char} ",		# Followed by space
				f" {char} ",	# Wrapped by spaces
				f" {char * 2}",	# Double occurance preceded by space
				f"{char * 2} ",	# Double occurance Followed by space
				f" {char * 2} ",# Double occurance wrapped by spaces
				f"\\{char}"		# Prefixed with backslash
			]

			for variation in variations:
				token_ids = self.tokenize(variation)

				if isinstance(token_ids, int):
					token_ids = [token_ids]	 # Convert single int to a list

				if len(token_ids) == 1:	 # Only block if it's a single token
					token_id = token_ids[0]
					if token_id not in logit_bias:	# Prevent duplicates
						logit_bias[token_id] = -100	 # Apply negative bias
						# print(f"Blocking token {token_id} for '{variation}'")

		return logit_bias
