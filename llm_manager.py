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
import unicodedata
import itertools
import time

class LLMManager:
	def __init__(self, config, conversation_manager):
		self.config = config
		self.conversation_manager = conversation_manager

		self.blocked_tokens				= config.get('LLM Manager', 'BlockedTokens', fallback='')
		self.base_tokens_per_speaker	= int(config.get('LLM Manager', 'BaseTokensPerSpeaker', fallback='30'))
		self.tokens_dialog_factor		= float(config.get('LLM Manager', 'TokensDialogFactor', fallback='1.5'))
		self.min_tokens					= int(config.get('LLM Manager', 'MinTokens', fallback='50'))
		self.max_tokens					= int(config.get('LLM Manager', 'MaxTokens', fallback='150'))
		self.rpg_max_tokens				= int(config.get('LLM Manager', 'RpgMaxTokens', fallback='100'))
		self.max_rpg_request_age		= float(config.get('LLM Manager', 'MaxRpgRequetAge', fallback='15.0'))
		self.main_queue_size			= int(config.get('LLM Manager', 'MainQueueSize', fallback='100'))
		self.rpg_queue_size				= int(config.get('LLM Manager', 'RpgQueueSize', fallback='20'))
		self.rpg_message_delay			= 3.5

		self.model = Llama(
			model_path=config.get('LLM Manager', 'ModelPath'),
			n_gpu_layers=int(config.get('LLM Manager', 'GPULayers', fallback='0')),
			n_threads=int(config.get('LLM Manager', 'Threads', fallback='1')),
			n_ctx=int(config.get('LLM Manager', 'ContextTokens', fallback='4096'))
		)

		# Predefined blocked words (these never change)
		self.logit_bias = self.tokenize_blocked_chars(self.blocked_tokens)
			
		# Load English words for dictionary check
		self.update_words()

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
		import nltk
		try:
			nltk.data.find('corpora/words')
		except LookupError:
			nltk.download('words')

		from nltk.corpus import words
		self.english_words = set(words.words())

	def tokenize(self, word):
		if not isinstance(word, bytes):	 # Ensure conversion only if necessary
			word = word.encode("utf-8")
		
		result = self.model.tokenize(word, add_bos=False)
		return result

	def queue_request(self, request_payload):
		"""
		Queues a request for the LLM.

		If the queue is full, purges lowest-priority items to make room.
		Will drop the incoming request if it has worse or equal priority and no room.
		"""
		priority = request_payload["priority"]
		prompt_data = request_payload["prompt_data"]
		request_speaker_map = request_payload["request_speaker_map"]
		llm_channel = prompt_data.get("llm_channel", None)

		logger.info(f"Queueing request batch {request_speaker_map} - Priority {priority}")

		with self.queue_lock:
			if self.queue.full():
				# Safely extract all items (Python's queue.PriorityQueue is a wrapper around heapq)
				all_items = []
				while not self.queue.empty():
					all_items.append(self.queue.get_nowait())

				# Sort by priority (lowest number = higher priority)
				all_items.append((priority, next(self.request_counter), prompt_data, request_speaker_map))
				all_items.sort(key=lambda x: x[0])	# Smallest priority = most important

				# Keep only the best N
				kept = all_items[:self.main_queue_size]
				dropped = all_items[self.main_queue_size:]

				# Cancel all dropped
				for dropped_item in dropped:
					_, _, dropped_prompt_data, dropped_request_map = dropped_item
					logger.debug(f"Purging request (priority {dropped_item[0]}) due to full queue")
					self.conversation_manager.receive_llm_responses(
						self.cancel_responses(dropped_request_map, dropped_prompt_data.get("llm_channel"), finish_reason="purged")
					)

				# Re-add all kept items
				for item in kept:
					self.queue.put_nowait(item)
				
				debug_print(f"Queue now contains {self.queue.qsize()} items after purge", color="dark_cyan")

				# If our request wasn't in the final list, it was dropped
				if (priority, prompt_data, request_speaker_map) not in kept:
					logger.debug(f"Request batch dropped due to low priority {priority}")
					self.conversation_manager.receive_llm_responses(
						self.cancel_responses(request_speaker_map, llm_channel, finish_reason="dropped")
					)
					return

			else:
				# Queue has room — add normally
				self.queue.put((priority, next(self.request_counter), prompt_data, request_speaker_map))

	def queue_rpg_request(self, request_payload):
		"""Queues an RPG request, evicting the oldest if full."""
		prompt_data = request_payload["prompt_data"]
		request_speaker_map = request_payload["request_speaker_map"]
		llm_channel = prompt_data.get("llm_channel", None)

		if self.rpg_queue.full():
			try:
				# Drop oldest (FIFO) and cancel it
				oldest = self.rpg_queue.get_nowait()
				old_llm_channel = oldest.get("llm_channel", None)
				old_request_speaker_map = oldest.get("request_speaker_map", {})

				self.conversation_manager.receive_llm_responses(
					self.cancel_responses(old_request_speaker_map, old_llm_channel, finish_reason="rpg_purged")
				)

				debug_print("RPG queue full — dropped oldest request to make room.", color="cyan")

			except queue.Empty:
				logger.warning("Attempted to purge RPG queue, but it was empty")

		# Enqueue the new request
		logger.info(f"Queueing RPG request batch {request_speaker_map}")
		self.rpg_queue.put(request_payload)

	def process_queue(self):
		"""Processes the main LLM queue using priority order."""
		while True:
			time.sleep(0.1)	 # Slight delay to avoid CPU hammering
			try:
				queue_size = self.queue.qsize()
				if queue_size > 0:
					debug_print(f"Queue length: {queue_size} pending requests.", color="yellow")

				# Safely attempt to get an item with timeout
				try:
					priority, batch_id, prompt_data, request_speaker_map = self.queue.get(timeout=1.0)
				except queue.Empty:
					continue  # No request in queue, loop again

				debug_print(f"Processing request batch {request_speaker_map} - Priority {priority}")

				start_time = time.time()
				self.last_main_process_time = start_time

				with self.llm_lock:
					response_dict = self.call_llm(prompt_data, request_speaker_map)

				elapsed = time.time() - start_time
				debug_print(f"LLM inference completed in {elapsed:.2f} seconds.", color="dark_green")

				# Send back responses to server via conversation manager
				self.conversation_manager.receive_llm_response(response_dict)

				# Mark task as completed
				self.queue.task_done()

			except Exception as e:
				logger.error(f"Error processing queue - {e}")
				logger.error(traceback.format_exc())

	def process_rpg_queue(self):
		"""Processes the RPG queue in FIFO order, skips old requests based on time_received."""
		while True:
			time.sleep(5.0)
			time_since_main_process = time.time() - self.last_main_process_time
			try:
				# Skip RPG processing if the main queue is still busy
				if self.queue.qsize() > 0 or time_since_main_process < 5.0:
					continue  # Main queue takes priority

				request_payload = self.rpg_queue.get(timeout=1.0)

				request_speaker_map = request_payload["request_speaker_map"]
				time_received = request_payload.get("time_received", time.time())
				llm_channel = request_payload.get("llm_channel", None)

				if time.time() - time_received > self.max_rpg_request_age:
					logger.debug("RPG request too old, cancelling cleanly.")
					self.conversation_manager.receive_llm_response(
						self.cancel_responses(request_speaker_map, llm_channel, finish_reason="rpg_request_stale")
					)
					self.rpg_queue.task_done()
					continue

				prompt_data = request_payload["prompt_data"]
				debug_print(f"Processing RPG request batch {request_speaker_map}")

				start_time = time.time()
				with self.llm_lock:
					response_dict = self.call_llm(prompt_data, request_speaker_map)
				elapsed = time.time() - start_time

				debug_print(f"RPG LLM inference completed in {elapsed:.2f} seconds.", color="dark_green")

				self.conversation_manager.receive_llm_response(response_dict)
				self.rpg_queue.task_done()

			except queue.Empty:
				# This happens if `get(timeout=1.0)` times out – just loop again
				continue

			except Exception as e:
				logger.error(f"Error processing RPG queue - {e}")
				logger.error(traceback.format_exc())

	def cancel_responses(self, request_speaker_map, llm_channel, finish_reason="stop"):
		response_dict = {
			request_id: {
				"mangos_response": {
					"text": "",
					"finish_reason": finish_reason,
					"prompt_tokens": 0,
					"completion_tokens": 0,
				},
				"speaker_name": speaker_name,
				"llm_channel": llm_channel,
				"response_delay": 0
			} for request_id, speaker_name in request_speaker_map.items()
		}

		return response_dict

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
			if not dialogues or not speaker_order:
				return self.cancel_responses(request_speaker_map, llm_channel, finish_reason="empty_response")

			final_responses = self.apply_delays_to_dialogues(dialogues, speaker_order, prompt_data, request_speaker_map)
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
			return self.cancel_responses(request_speaker_map, llm_channel, finish_reason="error") 

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

		# Strip matching outer single quotes and return
		if text.startswith("'") and text.endswith("'"):
			text = text[1:-1]
			return text

		# Strip nonsense stray leading single quote and return
		if text.startswith("'"):
			text = text[1:]
			return text

		quote_count = text.count('"')

		# Strip matching outer quotes and return
		if quote_count == 2 and text.startswith('"') and text.endswith('"'):
			text = text[1:-1]
			return text

		# Remove stray leading quote if present
		if text.startswith('"') and quote_count == 1:
			text = text[1:]
		# Remove stray trailing quote if present
		elif text.endswith('"') and quote_count == 1:
			text = text[:-1]

		return text

	def dialogue_sanity_check(self, text, speaker_name):
		# Assume speaker is talking about self in 3rd person
		pattern = rf"^[\W\s]*{re.escape(speaker_name)}[\W\s]+.*"
		if re.match(pattern, text):
			debug_print(f"<{speaker_name}> is talking about self in 3rd person", color="dark_yellow")
			return False

		quote_count = text.count('"')

		# Detect LLM trying to do goofy narration-style output
		if text.startswith('"'):
			if text.endswith('"') and quote_count > 2:
				debug_print(f"Detected 3rd person narration-style output for <{speaker_name}>", color="dark_yellow")
				return False
		
		if "(" in text or ")" in text:
			return False

		return True

	def remove_echoes(self, raw_output, context_lines):
		"""Removes echoed dialog from raw_output lines if they match or start with dialog from context."""
		# Automatically split context_lines if given as a string
		if isinstance(context_lines, str):
			context_lines = context_lines.strip().split("\n")
		
		# Parse context into a mapping of speakers to their dialogs
		context_dialogs = {}
		context_dialog_set = set()	# Set of all dialogs regardless of speaker
		
		for line in context_lines:
			if ":" in line:
				parts = line.split(":", 1)
				speaker = parts[0].strip()
				dialog = parts[1].strip()
				
				if speaker not in context_dialogs:
					context_dialogs[speaker] = []
				context_dialogs[speaker].append(dialog)
				context_dialog_set.add(dialog)
		
		# Process the output lines
		output_lines = raw_output.strip().split('\n')
		filtered_output = []
		removed_echoes = []
		
		for line in output_lines:
			if ":" not in line:
				filtered_output.append(line)
				continue
			
			parts = line.split(":", 1)
			speaker = parts[0].strip()
			dialog = parts[1].strip()
			
			should_keep = True
			
			# Check if this is a full dialog line in context
			for ctx_dialog in context_dialog_set:
				if dialog == ctx_dialog or (ctx_dialog.startswith(dialog) and len(dialog) > 5):
					# Either exact match or dialog is a prefix of a context dialog (and not too short)
					removed_echoes.append(line)
					should_keep = False
					break
			
			if not should_keep:
				continue
			
			# Check speaker-specific dialog prefixes
			if speaker in context_dialogs:
				speaker_dialogs = context_dialogs[speaker]
				
				# Sort dialogs by length (descending) to match longest prefix first
				speaker_dialogs.sort(key=len, reverse=True)
				
				for ctx_dialog in speaker_dialogs:
					if dialog.startswith(ctx_dialog):
						removed_echoes.append(line)
						# Remove the prefix and any leading punctuation or whitespace
						remainder = dialog[len(ctx_dialog):].lstrip()
						remainder = remainder.lstrip('.,;:!?"\t ')
						
						if remainder:
							filtered_output.append(f"{speaker}: {remainder}")
						
						should_keep = False
						break
			
			if should_keep:
				filtered_output.append(line)
		
		# Debug output
		if removed_echoes:
			print("Removed echoes from dialogues:")
			for echo in removed_echoes:
				print(f" - {echo}")
		
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
		prompt_context = prompt_data.get("context", "")

		# Create regex for name markers
		valid_speaker_pattern = "|".join(map(re.escape, sorted(set(speaker_names), key=len, reverse=True)))
		valid_member_pattern  = "|".join(map(re.escape, sorted(set(member_names), key=len, reverse=True)))

		# Match name markers properly:
		name_marker_regex = re.compile(
			rf"(?:^\s*(?P<valid>{valid_speaker_pattern}):\s*)"	# valid speakers anywhere
			rf"|(?:^\s*(?P<guess>{valid_member_pattern}):\s*)"	# valid member names anywhere
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

		raw_output = self.remove_echoes(raw_output, prompt_context).strip()
		if not raw_output:
			debug_print("LLM Mansger: Nothing valid left in raw output to parse. Aborting.", color="red")
			return [], []	# Return empty responses if no text

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
				if not self.dialogue_sanity_check(segment, current_speaker):
					debug_print(f"Stopping processing - LLM output for <{current_speaker}> failed sanity check", color="red")
					last_end = len(raw_output)	# Prevent the trailing capture from reprocessing text.
					break
				segment = self.strip_outer_quotes(segment)
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

	def apply_delays_to_dialogues(self, dialogues, speaker_order, prompt_data, request_speaker_map):
		"""Applies response delays and embedded delay tags to dialogues, supporting RPG and non-RPG types."""
		speaker_names = list(request_speaker_map.values())
		message_type = prompt_data.get("message_type", "Unknown")
		llm_channel = prompt_data.get("llm_channel", [])

		if expedited := prompt_data.get("expedited", False):
			debug_print("Bypassing initial thinking delay to expedite responses in batch", color="cyan")
		response_delays = {}

		# Get the time received of the request for the first speaker's dialogue
		first_speaker_time_received = self.conversation_manager.get_first_speaker_time_received(speaker_order[0], request_speaker_map, llm_channel)

		# 1. Build a global schedule: each entry represents one dialogue segment.
		global_schedule = []  # Each entry: {"speaker": speaker, "line": segment}
		speaker_counter = {speaker: 0 for speaker in speaker_names}

		for speaker in speaker_order:
			if speaker in dialogues and speaker_counter[speaker] < len(dialogues[speaker]):
				original_line = dialogues[speaker][speaker_counter[speaker]]
				if original_line.strip():
					min_length = random.uniform(25, 50)
					processed_line = self.insert_split_markers(original_line, min_length)
					segments = processed_line.split("|")
					for segment in segments:
						seg = segment.strip()
						if seg:
							global_schedule.append({
								"speaker": speaker,
								"line": seg
							})
				speaker_counter[speaker] += 1

		# 2. Compute global timestamp for each segment
		global_time = 0.0

		if message_type == "rpg":
			# RPG style: delay is based on the segment's own reading time
			for i, entry in enumerate(global_schedule):
				if i == 0:
					entry["global_time"] = 0.0
				else:
					gap = self.calculate_reading_delay(entry["line"])
					entry["global_time"] = global_schedule[i - 1]["global_time"] + gap
		else:
			# Non-RPG style: first line gets full typing delay; gaps are based on typing delay or fixed rpg delay
			first_line_typing_delay = (
				self.conversation_manager.calculate_typing_delay(global_schedule[0]["line"], thinking=(not expedited))
				if global_schedule else 0.0
			)
			debug_print(f"Calculated {first_line_typing_delay:.2f} seconds typing delay for first speaker <{speaker_order[0]}>", color="grey")
			lag = time.time() - (first_speaker_time_received / 1000.0)
			debug_print(f"Lag = {lag}", color="grey")
			first_line_typing_delay = max(first_line_typing_delay - lag, 0)
			debug_print(f"Adjusting delay of first line to {first_line_typing_delay:.2f} seconds", color="dark_yellow")

			global_time = first_line_typing_delay
			for i, entry in enumerate(global_schedule):
				if i == 0:
					entry["global_time"] = first_line_typing_delay
				else:
					gap = self.conversation_manager.calculate_typing_delay(entry["line"], thinking=True)
					entry["global_time"] = global_schedule[i - 1]["global_time"] + gap

		# 3. Collect segment times per speaker
		speaker_times = {speaker: [] for speaker in speaker_names}
		for entry in global_schedule:
			speaker_times[entry["speaker"]].append(entry["global_time"])

		# 4. Compute [DELAY] tags and first response delay per speaker
		modified_dialogues = {speaker: [] for speaker in speaker_names}
		speaker_response_delay = {}
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
				modified_line = data["line"] + f"[DELAY:{int(delay_tag * 1000)}]"
				modified_dialogues[speaker].append(modified_line)
			speaker_response_delay[speaker] = entries[0]["global_time"] if entries else 0.0

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

	def calculate_reading_delay(self, text):
		"""
		Calculate a reading delay based on number of words with a min and max for RPG messages.
		"""
		words = text.split()
		word_delay = len(words) * 0.5  # 0.3 seconds per word

		return (min(max(2.5, word_delay), 5.0))

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
