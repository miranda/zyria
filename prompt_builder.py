import string
from collections import defaultdict
from log_utils import debug_print, logger

class SafeDict(dict):
	def __missing__(self, key):
		return f"{{{key}}}"

def get_placeholders(template):
	return [field_name for _, field_name, _, _ in string.Formatter().parse(template) if field_name]

def dedup_ordered(*lists):
	seen = set()
	result = []
	for lst in lists:
		for item in lst:
			if item not in seen:
				seen.add(item)
				result.append(item)
	return result

class PromptBuilder:
	def __init__(self, config, memory_manager, context_manager, npc_manager):
		self.config = config
		self.memory_manager = memory_manager
		self.context_manager = context_manager
		self.npc_manager = npc_manager
		self.prompts = config["Prompts"]

		self.expansions = config.get('Prompt Builder', 'Expansions').split(',')
		self.guild_prompt_context_lines =	int(config.get('Prompt Builder', 'GuildPromptContextLines', fallback='20'))
		self.raid_prompt_context_lines =	int(config.get('Prompt Builder', 'RaidPromptContextLines', fallback='25'))
		self.party_prompt_context_lines =	int(config.get('Prompt Builder', 'PartyPromptContextLines', fallback='20'))
		self.world_prompt_context_lines =	int(config.get('Prompt Builder', 'WorldPromptContextLines', fallback='15'))

	def get_prompt(self, key):
		raw = self.prompts.get(key, "")
		raw = raw.replace("\\'", "\'")
		raw = raw.replace("\\#", "#")
		return raw.replace("\\n", "\n")

	def format_template(self, template, full_kwargs):
		required_fields = get_placeholders(template)
		used_kwargs = {k: full_kwargs.get(k, "") for k in required_fields}
		return template.format_map(SafeDict(used_kwargs))

	def build_prompt(self, *,
		llm_channel,
		message_type,
		senders,
		speakers,
		member_names,
		channel_label, 
		expansion,
		new_messages,
	):
		context, participants = self.context_manager.get_context(
			llm_channel, lines=self.get_max_context_lines(llm_channel), new_messages=new_messages
		)
		
		for entry in list(new_messages):  # iterate over a copy to allow mutation
			sender_type, sender_name, message, _ = entry
			if sender_type != "player":
				# Move non-player messages directly into context
				context += f"\n{sender_name}: {message}"
				new_messages.remove(entry)	# remove from new_messages

		participants = sorted(participants)
		senders = sorted(senders, key=lambda s: s["name"].lower())
		speakers = sorted(speakers, key=lambda s: s["name"].lower())
		sender_names = [s["name"] for s in senders]
		speaker_names = [s["name"] for s in speakers]
		all_names = dedup_ordered(participants, sender_names, speaker_names)

		sender_names_text = self.format_list_to_english(sender_names)
		speaker_names_text = self.format_list_to_english(speaker_names)

		player_messages_text, player_messages_prompt = self.get_player_messages_info(new_messages, speaker_names)
		context += player_messages_text

		single_sender = len(senders) == 1
		single_speaker = len(speakers) == 1
		single_message = len(new_messages) == 1

		location = self.memory_manager.format_location(
			speakers[0].get("zone", ""),
			speakers[0].get("subzone", ""),
			speakers[0].get("continent", "")
		)

		participants_info = [self.memory_manager.get_formatted_character_info(name, member_names, basic=True) for name in all_names]
		participants_info_text = '\n'.join(participants_info)
		
		sender_info = [
			self.memory_manager.get_formatted_character_info(name, member_names, show_personality=True)
			for name in sender_names if name not in speaker_names
		]
		sender_info_text = '\n\n'.join(sender_info)

		speaker_info = [self.memory_manager.get_formatted_character_info(name, member_names, show_personality=True) for name in speaker_names]
		speaker_info_text = '\n\n'.join(speaker_info)

		# Format kwargs for all templates
		format_kwargs = {
			"expansion": self.expansions[expansion],
			"channel_label": channel_label,
			"location": location,
			"guild_name": speakers[0]["guild"],
			"member_count": len(member_names),
			"participants": participants_info_text,
			"sender_names": sender_names_text,
			"speaker_names": speaker_names_text,
			"sender_info": sender_info_text,
			"speaker_info": speaker_info_text,
			"context": context
		}

		# Get pre-prompt template
		pre_prompt_key_map = {
			"in guild chat":				"PrePromptGuild",
			"in party chat":				"PrePromptParty",
			"in raid chat":					"PrePromptRaid",
			"in private message":			"PrePromptWhisper",
			"in world chat":				"PrePromptChannel",
			"in the general channel":		"PrePromptChannel",
			"in the trade channel":			"PrePromptChannel",
			"in looking for group":			"PrePromptChannel",
			"in the local defence channel": "PrePromptChannel",
			"in the world defence channel": "PrePromptChannel",
			"in guild recruitement":		"PrePromptChannel"
		}
		pre_prompt_key = pre_prompt_key_map.get(channel_label, None)
		if pre_prompt_key is None:
			pre_prompt_key = "PrePromptDirectSingle" if single_speaker else "PrePromptDirectMulti"

		pre_prompt_template = self.get_prompt(pre_prompt_key)
		pre_prompt = self.format_template(pre_prompt_template, format_kwargs)

		# Core speaker/sender prompt, start with participants prompt if there is existing conversation
		core_prompt = "\n\n"
		if context or new_messages:
			core_prompt += self.format_template(self.get_prompt("PromptParticipants"), format_kwargs)
		
		# Determine which speaker prompt to use
		if message_type == "new":
			prompt_key = "PromptContinueChat" if context else "PromptInitiateChat"
		else:
			speaker_type = "Single" if single_speaker else "Multi"
			prompt_key = f"Prompt{speaker_type}Speaker"

		speaker_prompt = "\n\n" + self.format_template(self.get_prompt(prompt_key), format_kwargs)

		# Determine if a sender prompt should be used (not used in "new" messages)
		if message_type != "new":
			if single_sender:
				message_part = "SingleMessage" if single_message else "MultiMessage"
				sender_prompt_key = f"PromptSingleSender{message_part}"
			else:
				sender_prompt_key = "PromptMultiSenderMultiMessage"

			if not llm_channel.startswith(("World", "Trade", "LocalDefense", "General", "LookingForGroup")):
				sender_prompt = "\n\n" + self.format_template(self.get_prompt(sender_prompt_key), format_kwargs)
				if sender_info_text:
					sender_prompt += "\n\n" + self.format_template(self.get_prompt("PromptSenderDetails"), format_kwargs)
			else:
				sender_prompt = ""

			core_prompt += sender_prompt + speaker_prompt
		else:
			core_prompt += speaker_prompt

		restrictions_prompt = "\n\n" + self.get_prompt("PromptRestrictions")
		context_prompt = "\n\n" + self.format_template(self.get_prompt("PromptContext"), format_kwargs) if context else ""

		# Post-prompt
		speaker_prompt = "\n\n" + (self.get_prompt("PostPromptConversationContinues") if context else self.get_prompt("PostPromptConversationBegins")) + "\n\n"
		if single_speaker:
			post_prompt_template = self.get_prompt("PostPromptSingleResponse")
			speaker_prompt += f"{speaker_names[0]}:"
		else:
			post_prompt_template = self.get_prompt("PostPromptMultiResponse")

		post_prompt = "\n\n" + self.format_template(post_prompt_template, format_kwargs)

		# Final assembly
		prompt = pre_prompt + core_prompt + restrictions_prompt + context_prompt + post_prompt + player_messages_prompt + speaker_prompt

		debug_print("Assembled prompt preview:")
		for segment, color in [
			(pre_prompt, "cyan"),
			(core_prompt, "cyan"),
			(restrictions_prompt, "dark_cyan"),
			(context_prompt, "grey"),
			(post_prompt, "cyan"),
			(player_messages_prompt, "cyan"),
			(speaker_prompt, "cyan")
		]:
			debug_print(segment, color=color, quiet=True, end="")
		if single_speaker:
			debug_print("\n", quiet=True)

		return prompt

	def build_rpg_prompt(self, *,
		llm_channel,
		speaker_role,
		bot_data,
		npc_data,
		npc_role_info,
		rpg_triggers,
		chat_topic,
		expansion
	):
		# --- Names ---
		bot_name = bot_data.get("name", "Unknown")
		npc_name = npc_data.get("name", "Unknown")

		# --- Who is speaking? Who is listening? ---
		speaker_name, sender_name = (bot_name, npc_name) if speaker_role == "bot" else (npc_name, bot_name)

		# --- Gather dynamic data ---
		location = self.memory_manager.format_location(
			bot_data.get("zone", ""),
			bot_data.get("subzone", ""),
			bot_data.get("continent", "")
		)

		context, _ = self.context_manager.get_context(llm_channel, lines=self.get_max_context_lines(llm_channel))

		# --- NPC Data formatting ---
		formatted_bot_data = {f"bot_{key}": value for key, value in bot_data.items()}
		formatted_npc_data = {f"npc_{key}": value for key, value in npc_data.items()}

		# --- Info blocks ---
		bot_info_text = self.memory_manager.get_formatted_character_info(bot_name, [], show_personality=True)
		npc_roles_text = self.npc_manager.get_roles_formatted(npc_name, npc_role_info)
		rpg_trigger_text = self.npc_manager.get_rpg_triggers_formatted(bot_name, npc_name, rpg_triggers)
		chat_topic_text = self.npc_manager.get_chat_topic_formatted(bot_name, npc_name, chat_topic)

		# --- NPC Info block with safe join ---
		npc_info_text = "\n\n".join(filter(None, [npc_roles_text, rpg_trigger_text]))

		# --- Template kwargs ---
		format_kwargs = {
			**formatted_bot_data,
			**formatted_npc_data,
			"npc_full_name": self.get_npc_full_name(npc_data),
			"expansion": self.expansions[expansion],
			"location": location,
			"sender_name": sender_name,
			"speaker_name": speaker_name,
			"bot_info": bot_info_text,
			"npc_info": npc_info_text,
			"context": context,
			"chat_topic": chat_topic_text
		}

		pre_prompt = self.format_template(self.get_prompt("PrePromptRpg"), format_kwargs)

		core_prompt = (
			"\n\n" + self.format_template(self.get_prompt("PromptRpgDetails"), format_kwargs) +
			"\n\n" + self.format_template(self.get_prompt("PromptRpgInstructions"), format_kwargs)
		)

		restrictions_prompt = "\n\n" + self.get_prompt("PromptRestrictions")
		context_prompt = "\n\n" + self.format_template(self.get_prompt("PromptContext"), format_kwargs)
		chat_topic_prompt = "\n\n" + self.format_template(self.get_prompt("PromptRpgChatTopic"), format_kwargs)

		# Post-prompt
		if chat_topic == "goodbye":
			post_prompt_template = self.get_prompt("PostPromptRpgGoodbye")
		else:
			post_prompt_template = self.get_prompt("PostPromptRpg")

		post_prompt = "\n\n" + self.format_template(post_prompt_template, format_kwargs)

		# Final assembly
		prompt = pre_prompt + core_prompt + restrictions_prompt + context_prompt + chat_topic_prompt + post_prompt
		
		debug_print("Assembled RPG prompt preview:")
		for segment, color in [
			(pre_prompt, "magenta"),
			(core_prompt, "magenta"),
			(restrictions_prompt, "dark_magenta"),
			(context_prompt, "grey"),
			(chat_topic_prompt, "magenta"),
			(post_prompt, "magenta"),
		]:
			debug_print(segment, color=color, quiet=True, end="")
		debug_print("", quiet=True)

		return prompt

	def get_player_messages_info(self, new_messages, speaker_names):
		player_sender_names = set()
		speaker_targets = defaultdict(set)
		player_messages_text = ""

		for _, sender_name, message, targets in new_messages:
			player_sender_names.add(sender_name)
			player_messages_text += f"\n{sender_name}: {message}"	
			if targets:
				speaker_targets[sender_name].update(targets)

		prompt = ""

		for sender in sorted(player_sender_names):
			targets = sorted(speaker_targets[sender])  # optional: sorted for consistent output

			formatted_senders = self.format_list_to_english([sender])
			formatted_targets = self.format_list_to_english(targets)
			formatted_senders_negative = self.format_list_to_english([sender], negative=True)

			prompt_template = self.get_prompt("PostPromptPlayerDirect") if targets else self.get_prompt("PostPromptPlayer") 
	
			format_kwargs = {
				"sender_names": formatted_senders,
				"speaker_names": formatted_targets if targets else self.format_list_to_english(speaker_names),
				"sender_names_negative": formatted_senders_negative
			}
			
			prompt += "\n" + self.format_template(prompt_template, format_kwargs)

		prompt = "\n" + prompt if prompt else prompt

		return player_messages_text, prompt

	def get_npc_full_name(self, npc_data):
		name = npc_data.get("name", "Unknown")
		subname = npc_data.get("creature_subname", "")
		creature_type = npc_data.get("creature_type", "")
		
		return f"{creature_type} {name} {subname}".strip()

	def get_max_context_lines(self, llm_channel):
		channel_map = {
			"Guild": self.guild_prompt_context_lines,
			"Raid": self.raid_prompt_context_lines,
			"Party": self.party_prompt_context_lines,
		}
		return next((v for k, v in channel_map.items() if llm_channel.startswith(k)), self.world_prompt_context_lines)

	def get_guild_status_prompts(self, senders, speakers):
		single_sender = len(senders) == 1
		single_speaker = len(speakers) == 1

		speaker_guild = speakers[0].get("guild", "No guild") if speakers else "No guild"
		sender_guild = senders[0].get("guild", "No guild") if senders else "No guild"

		# Default values
		def get_status(is_also=False, match=False, guild="No guild"):
			if guild == "No guild":
				return self.prompts.get("PromptGuildStatus", "and {is_also_not} a member of {guild_name}").format(
					is_also_not="is not",
					guild_name="any guild"
				)
			return self.prompts.get("PromptGuildStatus", "and {is_also_not} a member of {guild_name}").format(
				is_also_not="is also" if is_also else ("is" if match else "is not"),
				guild_name=guild
			)

		# Logic for each separately
		sender_status = get_status(
			is_also=(single_sender and single_speaker and sender_guild == speaker_guild),
			match=(sender_guild == speaker_guild),
			guild=speaker_guild
		)

		speaker_status = get_status(
			is_also=False,
			match=(speaker_guild == sender_guild),
			guild=sender_guild
		)

		return sender_status, speaker_status

	def format_list_to_english(self, items, negative=False):
		if not items:
			return ""
		elif len(items) == 1:
			return items[0]
		elif len(items) == 2:
			conj = "or" if negative else "and"
			return f"{items[0]} {conj} {items[1]}"
		else:
			conj = "or" if negative else "and"
			return ", ".join(items[:-1]) + f", {conj} {items[-1]}"
