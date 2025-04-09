import threading
import time
from log_utils import debug_print, logger
from collections import defaultdict, deque
import math
import re
import random

class ConversationManager:
	def __init__(self, config, context_manager):
		self.context_manager = context_manager

		self.queue_upper_limit			= int(config.get('Conversation Manager', 'QueueUpperLimit', fallback='8'))
		self.queue_lower_limit			= int(config.get('Conversation Manager', 'QueueLowerLimit', fallback='3'))
		self.queue_sleep_random_high	= int(config.get('Conversation Manager', 'QueueSleepRandomHigh', fallback='20'))
		self.queue_sleep_random_low		= int(config.get('Conversation Manager', 'QueueSleepRandomLow', fallback='10'))
		self.queue_sleep_wake_time		= int(config.get('Conversation Manager', 'QueueSleepWakeTime', fallback='45'))
		self.output_max_pause_time		= int(config.get('Conversation Manager', 'OutputMaxPauseTime', fallback='5'))
		self.fatigue_multiplier			= float(config.get('Conversation Manager', 'FatigueMultiplier', fallback='2.0'))
		self.fatigue_reset_time			= int(config.get('Conversation Manager', 'FatigueResetTime', fallback='30'))
		self.max_batch_size				= int(config.get('Conversation Manager', 'MaxBatchSize', fallback='4'))
		self.rpg_pairing_timeout		= float(config.get('Conversation Manager', 'RpgPairingTimeout', fallback='2.0'))

		self.conversation_queues = defaultdict(deque)			# Incoming requests per llm_channel
		self.conversation_locks = defaultdict(threading.Lock)	# Per-channel locks
		self.queues_lock = threading.RLock()					# Lock for modifying global queues
		self.last_speaker_data = defaultdict(dict)				# Track last bot's speaking per channel
		self.fatigue_counter = defaultdict(dict)				# Tracks talking fatigue

		# Lag compensation for player messages per channel
		self.last_adjusted_player_message = defaultdict(lambda: (None, 0.0))  # key → (last_key, lag_value)

		self.busy_bots = {}				# Tracks the scheduled bot busy state release times
		self.paused_queues = {}			# Tracks paused queues with timestamps
		self.suspended_queues = {}		# Tracks suspended queues with timestamps
		self.last_paused_data = {}		# Stores last time a queue was paused
		self.last_paused_times = {}		# Stores last time a queue was paused
		self.sleeping_queues = {}		# Tracks queues in sleep mode 
		self.sleep_thresholds = {}		# Stores random sleep message thresholds
		self.last_request_time = {}		# Stores the last request timestamp per channel
		self.last_speak_time = {}		# Tracks last speaking time for each bot
		self.last_batch_added = {}		# Tracks the sender and message of the last request added to queue
		self.completed_request_count = {}

		self.channel_thresholds = {
			"RPG":				int(config.get("Conversation Manager", "RpgRequestThreshold", fallback=4)),
			"World":			int(config.get("Conversation Manager", "WorldRequestThreshold", fallback=6)),
			"Trade":			int(config.get("Conversation Manager", "DefaultRequestThreshold", fallback=5)),
			"LocalDefense":		int(config.get("Conversation Manager", "DefaultRequestThreshold", fallback=5)),
			"General":			int(config.get("Conversation Manager", "DefaultRequestThreshold", fallback=5)),
			"LookingForGroup":	int(config.get("Conversation Manager", "DefaultRequestThreshold", fallback=5))
		}

		self.cancel_request_data = {
			"mangos_response": {
				"text": "",
				"finish_reason": "stop",
				"prompt_tokens": 0,
				"completion_tokens": 0
			},
			"response_delay": 0
		}

		# Start a background thread for automatic unpausing and unsuspending
		threading.Thread(target=self.unpause_loop, daemon=True).start()

	def unpause_loop(self):
		"""Background thread that periodically checks for expired pauses and suspended queues."""

		while True:
			with self.queues_lock:
				current_time = time.time()

				# ✅ Find empty queues
				empty_queues = [ch for ch, queue in self.conversation_queues.items() if not queue]

				# ✅ Unpause channels that reached their time limit
				expired_pauses = [ch for ch, unpause_time in self.paused_queues.items() if current_time >= unpause_time]
				for ch in expired_pauses:
					del self.paused_queues[ch]
					debug_print(f"Resumed processing of channel {ch} after time-based pause", color="green")

				# ✅ Unsuspend sleeping queues after being idle for long enough
				expired_sleeps = [
					ch for ch in self.sleeping_queues
					if ch in empty_queues and current_time >= self.sleeping_queues[ch] + self.queue_sleep_wake_time
				]

				for ch in expired_sleeps:
					del self.sleeping_queues[ch]  # Remove from sleep state
					self.unsuspend_queue(ch)  # Resume normal operation
					debug_print(f"Resumed queue {ch} after sleeping for {self.queue_sleep_wake_time} seconds", color="cyan")

			time.sleep(0.5)	 # Check every 500ms

	def get_all_channels(self):
		"""Returns all known channels (even if empty)"""
		with self.queues_lock:
			return list(self.conversation_queues.keys())

	def get_active_channels(self):
		"""Returns a list of channels that have pending requests."""
		with self.queues_lock:
			return [channel for channel, queue in self.conversation_queues.items() if queue]

	def is_channel_overloaded(self, llm_channel):
		"""Checks if the given channel is overloaded based on configured thresholds."""
		# Detect channel type by prefix
		channel_type = None 

		for prefix in self.channel_thresholds.keys():
			if llm_channel.startswith(prefix):
				channel_type = prefix
				break

		if not channel_type:
			return False

		threshold = self.channel_thresholds.get(channel_type, None)

		# Count requests for that specific channel
		with self.queues_lock:
			queue = self.conversation_queues.get(llm_channel, [])
			request_count = sum(1 for request in queue if request["status"] in ("pending", "processing"))

		return request_count >= threshold

	def is_suspended(self, llm_channel):
		"""Returns True if the conversation queue for the given channel's input is suspended."""
		with self.queues_lock:
			return llm_channel in self.suspended_queues

	def suspend_queue(self, llm_channel):
		"""Stops a conversation queue from accepting new pending requests."""
		with self.queues_lock:
			self.suspended_queues[llm_channel] = time.time()
			debug_print(f"Suspending channel {llm_channel}", color="red")

	def unsuspend_queue(self, llm_channel):
		"""Resumes a suspended queue by removing it from the suspended list."""
		with self.queues_lock:
			if llm_channel in self.suspended_queues:
				del self.suspended_queues[llm_channel]
				debug_print(f"Released channel {llm_channel} from suspension", color="green")

	def add_request(self, llm_channel, request_data):
		"""Adds a new request to the conversation queue."""
		with self.queues_lock:
			# ✅ Ensure channel queue exists
			self.conversation_queues[llm_channel].append(request_data)
		
			sender_name = request_data["sender_name"]
			message = request_data["message"]
			debug_print(f"Added request {request_data['request_id']} sent by <{sender_name}>")
			if self.last_batch_added.get(llm_channel, "") != (sender_name, message):
				self.manage_queue_suspension(llm_channel)

			self.last_batch_added[llm_channel] = (sender_name, message)

	def fetch_pending_requests(self, llm_channel, current_size=0):
		"""Fetches pending "new" and "reply" requests, ensuring they are processed one at a time."""
		with self.queues_lock:
			queue = self.conversation_queues[llm_channel]
			if not queue:
				return [], None

			pending_requests = []
			batch_sender_type = None
			batch_message_type = None
			batch_size = 0

			for request in queue:
				if request.get("status") != "pending":
					continue

				# Critical error check: message_type missing
				if "message_type" not in request:
					logger.error(f"❌ ERROR: No message_type in request: {request}")

				current_message_type = request["message_type"]
				current_sender_type = request["sender"]["type"]

				# Always process "new" messages in their own batch
				if current_message_type == "new":
					if not pending_requests:
						pending_requests.append(request)
					break

				# If no batch established yet, start a batch
				if not pending_requests:
					pending_requests.append(request)
					batch_sender_type = current_sender_type
					batch_message_type = current_message_type
					batch_size = 1
					continue

				# Check if request can be added to current batch
				if (current_sender_type == batch_sender_type and 
					current_message_type == batch_message_type):
					pending_requests.append(request)
					batch_size += 1
					if batch_size + current_size >= self.max_batch_size:
						break
				else:
					# Different sender or message type, stop collecting
					break

		return pending_requests, batch_message_type or (pending_requests[0]["message_type"] if pending_requests else None)

	def fetch_pending_rpg_requests(self, llm_channel):
		"""Fetches pending RPG request pairs. Cancels unpaired ones after 1 second."""
		with self.queues_lock:
			queue = self.conversation_queues[llm_channel]
			if not queue:
				return [], None

			pending_requests = []
			batch_timestamp = None

			now = int(time.time() * 1000)  # Current time in ms

			for request in queue:
				if request.get("status") != "pending" or request.get("message_type") != "rpg":
					continue

				if "time_created" not in request:
					logger.error(f"❌ ERROR: Missing 'time_created' in RPG request: {request}")
					continue

				current_timestamp = request["time_created"]

				if not pending_requests:
					pending_requests.append(request)
					batch_timestamp = current_timestamp
					continue

				# Only match requests with the *exact* same timestamp
				if current_timestamp == batch_timestamp:
					pending_requests.append(request)
					if len(pending_requests) == 2:
						break
				else:
					break  # Different timestamp means not part of the same pair

			if len(pending_requests) == 2:
				debug_print(f"✅ Found RPG request pair: {[r.get('request_id') for r in pending_requests]}")

				return pending_requests, "rpg"

			# If we only found one, check if it's too old
			if len(pending_requests) == 1:
				age = now - batch_timestamp
				if age < self.rpg_pairing_timeout * 1000:
					return [], None	 # Still within grace period

				# Timeout reached — cancel it
				orphan = pending_requests[0]
				self.cancel_request(orphan)
				debug_print(f"Cancelling orphaned request after RPG pairing timeout: {orphan['request_id']}", color="dark_yellow")

			return [], None

	def prioritize_player_message(self, llm_channel, target_speakers=None):
		"""Prioritize player messages by clearing related bot responses, or all if target_speakers is None."""
		with self.queues_lock:
			queue = self.conversation_queues[llm_channel]

			if not any(request.get("sender", {}).get("type") == "player" for request in queue):
				return

			debug_print(f"🎯 Player message detected in {llm_channel}! Prioritizing.", color="yellow")

			for request in queue:
				if request.get("sender", {}).get("type") == "player":
					continue

				if request["status"] == "pending":
					# Match directly by the pending request's speaker
					if target_speakers is None or request["speaker"]["name"] in target_speakers:
						self.cancel_request(request)

				elif request["status"] == "completed":
					# Completed responses may have been a batch, check `conversation_members`
					speakers_involved = request.get("conversation_members", [])

					if target_speakers is None or any(speaker in target_speakers for speaker in speakers_involved):
						self.cancel_request(request)

			debug_print(f"Cleared relevant bot requests in {llm_channel} to prioritize player message.", color="yellow")

	def adjust_response_delay_for_lag(self, key, llm_channel, response_delay, now, time_created):
		"""removes lag from response delay for all related responses without harming response timings."""
		last_key, last_lag = self.last_adjusted_player_message[llm_channel]

		if last_key != key:
			lag = max(now - time_created, 0)
			self.last_adjusted_player_message[llm_channel] = (key, lag)
		else:
			lag = last_lag

		return max(response_delay - lag, 0)

	def fetch_completed_request(self, llm_channel, request_id):
		"""Fetches a completed request from the queue and removes it, enforcing pacing and updating fatigue."""

		if self.is_paused(llm_channel):
			return None	 # Skip if queue is paused

		with self.queues_lock:
			queue = self.conversation_queues[llm_channel]
			fatigue_counter = self.fatigue_counter[llm_channel]

			for request in queue:
				if request["request_id"] != request_id or request["status"] in ("pending", "processing"):
					continue

				time_created = request.get("time_created", 0)
				message_type = request.get("message_type", "unknown")
				sender_type = request.get("sender", {}).get("type", "Unknown")
				sender_name = request.get("sender_name", "Unknown")
				speaker_name = request.get("speaker_name", "Unknown")
				response_text = (request.get("mangos_response") or {}).get("text", "")
				response_delay = request.get("response_delay", 0.0)

				now = time.time()

				if request["status"] == "completed":
					# Remove processing lag from response delay for player messages
					if sender_type == "player" and response_text:
						key = (sender_name, request.get("message", ""), tuple(request.get("conversation_members", [])))
						response_delay = self.adjust_response_delay_for_lag(key, llm_channel, response_delay, now, time_created)

					# Compute full display duration
					last_word = response_delay + self.extract_message_delays(response_text)
					fatigue_counter.setdefault(speaker_name, 0)

					speaker_fatigue = fatigue_counter[speaker_name]
					effective_fatigue = speaker_fatigue * self.fatigue_multiplier

					# Busy state correction
					if self.is_bot_busy(speaker_name) and self.get_bot_remaining_busy_time(speaker_name) == float('inf'):
						expire_time = last_word + (effective_fatigue if (response_text and sender_type != "player") else 0)
						debug_print(f"Updating busy expire time for <{speaker_name}> (fatigue: {effective_fatigue:.2f} seconds)", color="dark_magenta")
						self.set_bot_busy(speaker_name, delay=expire_time)

					# Insert pause for typing if a speaker's message overlaps too soon with their last message
					if response_delay == 0 and not request.get("pause_triggered"):
						speaker_finish_time = self.last_speaker_data[llm_channel].get(speaker_name, 0)
						remaining = max(speaker_finish_time - now, 0)

						if remaining > 0 and not self.is_paused(llm_channel):
							if self.pause_for_typing(request, llm_channel, remaining):
								request["pause_triggered"] = True
								return None

					last_spoke = self.last_speaker_data[llm_channel].get(speaker_name, now)
					time_since_last_spoke = now - last_spoke

					# Dispatch
					logger.info(f"✅ Found completed request {request_id} in {llm_channel} for <{speaker_name}>")

					if response_text:
						self.last_speaker_data[llm_channel][speaker_name] = now + last_word
						self.completed_request_count[llm_channel] = self.completed_request_count.get(llm_channel, 0) + 1
						self.check_queue_sleep(llm_channel)
						self.context_manager.add_message(llm_channel, speaker_name, response_text, response_delay)

						if time_since_last_spoke >= self.fatigue_reset_time:
							fatigue_counter[speaker_name] = 0
							debug_print(f"Reset fatigue counter for <{speaker_name}> in channel {llm_channel}", color="grey")
						elif sender_type != "player":
							fatigue_counter[speaker_name] += 1
							speaker_fatigue = fatigue_counter[speaker_name]
							effective_fatigue = speaker_fatigue * self.fatigue_multiplier
							debug_print(f"Increased fatigue counter to {speaker_fatigue} for <{speaker_name}> in channel {llm_channel} (fatigue: {effective_fatigue:.2f})", color="dark_magenta")

				else:
					# Clean up canceled request
					logger.info(f"✅ Removed cancelled request {request_id} for <{speaker_name}> from channel {llm_channel}")

				queue.remove(request)
				self.manage_queue_suspension(llm_channel)
				return request

		return None

	def extract_message_delays(self, message):
		"""Returns total combined delay amount from extracted delay tags."""
		# Find all delay values (list of strings)
		delay_matches = re.findall(r"\[DELAY:(\d+)\]", message)

		# Convert to integers and sum (convert milliseconds to seconds)
		return sum(int(d) for d in delay_matches) / 1000

	def is_paused(self, llm_channel):
		"""Returns True if the conversation queue for the given channel's output is suspended."""
		with self.queues_lock:
			unpause_time = self.paused_queues.get(llm_channel)
			if unpause_time is None:
				return False
			if time.time() >= unpause_time:
				# ✅ Pause expired, remove it
				del self.paused_queues[llm_channel]
				return False
			return True

	def pause_for_typing(self, request, llm_channel, remaining_pause):
		with self.queues_lock:
			speaker_name = request["speaker_name"]
			sender_message = request["message"]
			response_text = request["mangos_response"]["text"]
			sender_type = request.get("sender", {}).get("type")
			sender_name = request.get("sender", {}).get("name")
			time_created = request["time_created"]

			# Calculate pause time
			segments = response_text.split("|")
			line = re.sub(r"\[DELAY:\d+\]", "", segments[0]).strip()
			pause_time = remaining_pause + self.context_manager.calculate_typing_delay(line)
			pause_time = min(pause_time, self.output_max_pause_time)

			# Generate a unique ID for this pause trigger
			pause_id = (speaker_name, sender_name, sender_message, response_text)

			# Prevent re-pausing for the same output
			if self.last_paused_data.get(llm_channel) == pause_id:
				return False  # Already paused for this exact content

			if pause_time <= 0.01:	# Skip zero-length pauses
				return False

			# Actually pause
			if self.pause_queue_output(llm_channel, pause_time):
				self.last_paused_data[llm_channel] = pause_id
				debug_print(f"⏳ Pausing channel {llm_channel} for {pause_time:.2f} seconds to simulate <{speaker_name}> typing", color="yellow")
				return True	 # Pause succeeded

			return False  # No pause was made

	def pause_queue_output(self, llm_channel, duration):
		"""Pauses queue output for a channel for a given duration, unless it was paused within the last 10 seconds."""
		if duration <= 0.01:  # ⛔ Skip zero-length pauses
			return False

		if self.is_paused(llm_channel):
			return False

		with self.queues_lock:
			now = time.time()
			duration = min(duration, self.output_max_pause_time)
			unpause_time = now + duration
			self.paused_queues[llm_channel] = unpause_time
			self.last_paused_times[llm_channel] = now  # Track last pause time

			debug_print(f"⏸️ Paused channel {llm_channel} for {duration:.2f} seconds", color="red")
			return True

	def check_queue_sleep(self, llm_channel):
		"""Checks if a queue has reached its sleep threshold and puts it to sleep if needed."""
		with self.queues_lock:
			completed_count = self.completed_request_count.get(llm_channel, 0)
			sleep_threshold = self.sleep_thresholds.get(llm_channel, random.randint(self.queue_sleep_random_low, self.queue_sleep_random_high))	 # Set if missing

			if completed_count >= sleep_threshold and llm_channel not in self.sleeping_queues:
				self.sleeping_queues[llm_channel] = time.time()	 # Mark sleep start time
				self.sleep_thresholds[llm_channel] = random.randint(self.queue_sleep_random_low, self.queue_sleep_random_high)	 # Set new random sleep threshold
				self.completed_request_count[llm_channel] = 0  # Reset count after sleeping
				debug_print(f"Putting queue {llm_channel} to sleep (completed {completed_count} requests, threshold was {sleep_threshold})", color="blue")
				self.suspend_queue(llm_channel)

	def manage_queue_suspension(self, llm_channel, upper_limit=None, lower_limit=None):
		"""Suspend queue if too many pending/processing requests, unsuspend when below threshold."""
		if upper_limit is None:
			upper_limit = self.queue_upper_limit
		if lower_limit is None:
			lower_limit = self.queue_lower_limit

		with self.queues_lock:
			non_completed_count = sum(
				1 for req in self.conversation_queues[llm_channel]
				if req["status"] in {"pending", "processing"}
			)

			# ✅ If queue is over the upper limit, suspend it
			if non_completed_count > upper_limit and not self.is_suspended(llm_channel):
				debug_print(f"🔴 Suspending queue {llm_channel} due to high load ({non_completed_count} active requests)", color="red")
				self.suspend_queue(llm_channel)

			# ✅ If queue is below the lower limit, attempt to unsuspend (but not if sleeping)
			elif non_completed_count <= lower_limit and self.is_suspended(llm_channel) and llm_channel not in self.sleeping_queues:
				debug_print(f"🟢 Unsuspending queue {llm_channel} - Load dropped to {non_completed_count}", color="green")
				self.unsuspend_queue(llm_channel)

	def receive_llm_response(self, response_dict):
		"""Handles responses from LLM and updates the corresponding requests in the queue."""
		with self.queues_lock:
			for request_id, response_data in response_dict.items():
				llm_channel = response_data.get("llm_channel")

				queue = self.conversation_queues[llm_channel]
				if not llm_channel or not any(req.get("request_id") == request_id for req in queue):
					logger.error(f"❌ Dropped response for unknown request {request_id}")
					continue

				for request in queue:
					if request["request_id"] == request_id:
						speaker_name = response_data.get("speaker_name", "Unknown")
						# Update it in place
						request.update({
							"status": "completed",
							"mangos_response": response_data.get("mangos_response", {}),
							"response_delay": response_data.get("response_delay", 0)
						})

						debug_print(f"Updated request {request_id} to 'completed' for <{speaker_name}>", color="green")
						break  # Stop once updated

	def cancel_request(self, request, release=False):
		"""Handles responses from LLM and updates the corresponding requests in the queue."""
		with self.queues_lock:	# 🔒 Ensure thread safety
			# Update it in place
			request.update({
				"status": "cancelled",
				**self.cancel_request_data	# Merge common structure
			})
			speaker_name = request.get("speaker_name", "Unknown")
			debug_print(f"Cancelled request {request['request_id']} for <{speaker_name}>", color="red")
			if self.is_bot_busy(speaker_name):
				if release or (self.get_bot_remaining_busy_time(speaker_name) == float('inf')):
					self.set_bot_busy(speaker_name, delay=0)	# Release bot immediately

	def cancel_by_id_channel(self, request_id, llm_channel, release=False):
		"""Handles responses from LLM and updates the corresponding requests in the queue."""
		with self.queues_lock:
			queue = self.conversation_queues[llm_channel]
			if not llm_channel or not any(req.get("request_id") == request_id for req in queue):
				logger.error(f"❌ Failed to cancel unknown request {request_id}")

			for request in queue:
				if request["request_id"] == request_id and request["status"] != "completed":
					self.cancel_request(request, release=release)
					break  # Stop once updated

	def peek_next_request(self, llm_channel):
		"""Returns the next request without removing it from the queue."""
		if self.conversation_queues[llm_channel]:
			return self.conversation_queues[llm_channel][0]	 # Peek at first item
		return None

	def is_bot_busy(self, bot_name):
		"""Returns True if a bot is still busy, False otherwise. Cleans up expired bots."""
		with self.queues_lock:
			release_time = self.busy_bots.get(bot_name)

			if release_time is None:
				return False  # Bot is not busy

			if release_time == float('inf'):
				return True	 # Permanently busy

			current_time = time.time()
			if release_time - current_time <= 1.0:	# Treat <1 second as expired
				del self.busy_bots[bot_name]  # Expired, remove from dict
				return False

			return True	 # Still busy

	def get_bot_remaining_busy_time(self, bot_name):
		"""Returns the remaining busy time for a bot, 0 if expired, or None if not set."""
		with self.queues_lock:
			release_time = self.busy_bots.get(bot_name)

			if release_time is None:
				return None	 # No expiration set

			if release_time == float('inf'):
				return float('inf')	 # Permanently busy

			remaining_time = max(0.0, release_time - time.time())

			return remaining_time

	def set_bot_busy(self, bot_name, delay=None):
		"""Sets a bot as busy for a given delay. If delay is 0, it expires immediately.
		If delay is None, the bot is set to indefinitely busy."""
		with self.queues_lock:
			if delay is None:
				self.busy_bots[bot_name] = float('inf')	 # Set indefinitely busy
				debug_print(f"Marked <{bot_name}> as indefinitely busy", color="magenta")
			elif delay > 0:
				expiration_time = math.ceil(time.time() + delay)  # Round up to next second
				self.busy_bots[bot_name] = expiration_time
				debug_print(f"Marked <{bot_name}> as busy for {delay:.2f} seconds (expires at {expiration_time})", color="magenta")
			else:
				self.busy_bots.pop(bot_name, None)	# Remove busy state
				debug_print(f"Released <{bot_name}> from busy state", color="green")
