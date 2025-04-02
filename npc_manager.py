import json
from log_utils import debug_print, logger

class NPCManager:
	def __init__(self, config):
		self.config = config
		self.flags_data = {}
		self._load_flags()

	def _load_flags(self):
		try:
			with open("npc_flags.json", "r") as f:
				self.flags_data = json.load(f)
		except Exception as e:
			logger.error(f"Failed to load npc_flags.json: {e}")
			self.flags_data = {}

	def _iter_flag_entries(self):
		"""Yield (flag_value, entry_dict) for each valid flag."""
		for hex_flag, entry in self.flags_data.items():
			if not hex_flag.startswith("0x"):
				continue
			try:
				flag_value = int(hex_flag, 16)
				yield flag_value, entry
			except ValueError:
				logger.warning(f"Invalid flag format: {hex_flag}")

	def get_roles(self, npc_flags_int):
		roles = []
		for flag_value, entry in self._iter_flag_entries():
			if npc_flags_int & flag_value:
				role = entry.get("role")
				if role:
					roles.append(role)
		return roles

	def get_roles_formatted(self, npc_name, npc_role_info):
		npc_flags_int = npc_role_info.get("npc_flags", 0)
		talents = npc_role_info.get("talents", False)
		pet_skills = npc_role_info.get("pet_skills", False)
		
		prompts = []
		for flag_value, entry in self._iter_flag_entries():
			if npc_flags_int & flag_value:
				prompt = entry.get("prompt", "")
				if prompt:
					prompts.append(prompt.format(name=npc_name))

		bonus_options = self.flags_data.get("bonus_options", {})
		if talents:
			prompts.append(bonus_options.get("talents", "").format(name=npc_name))
		if pet_skills:
			prompts.append(bonus_options.get("pet_skills", "").format(name=npc_name))

		return " ".join(prompts)

	def get_chat_topic_formatted(self, bot_name, npc_name, chat_topic):
		topic_map = {
			"zone": f"{bot_name} wants to ask {npc_name} about the zone.",
			"lore": f"{bot_name} wants to ask {npc_name} about lore they know about.",
			"events": f"{bot_name} wants to ask {npc_name} about recent events.",
			"npcs": f"{bot_name} wants to ask {npc_name} about nearby NPCs.",
			"general": f"{bot_name} wants to have a chat with {npc_name}."
		}

		return topic_map.get(chat_topic, topic_map["general"])
		
	def get_rpg_triggers_formatted(self, bot_name, npc_name, triggers):
		trigger_map = {
			"start_quest": f"{npc_name} can offer {bot_name} a new quest.",
			"end_quest": f"{bot_name} has just finished a quest for {npc_name}.",
			"repair": f"{npc_name} can repair {bot_name}'s broken equipment.",
			"sell": f"{bot_name} has items {npc_name} will buy.",
			"buy": f"{npc_name} has items {bot_name} wants to buy.",
			"goodbye": f"{bot_name} has to go. Say goodbye."
		}

		lines = [
			trigger_map[key]
			for key, active in triggers.items()
			if active and key in trigger_map
		]
		return " ".join(lines)
