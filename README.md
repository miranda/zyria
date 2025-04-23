# Zyria LLM Server

Zyria is a Python-based local LLM server designed specifically for use with the [playerbots-zyria](https://github.com/miranda/playerbots-zyria) fork of the cMaNGOS Playerbots module.

The goal of Zyria is to make playerbots as immersive and believable as possible through LLM-driven chat and personality-based behavior.

---

## Key Features

- **Native integration with Playerbots** – Zyria is purpose-built to communicate with playerbots in real-time, leveraging in-game context that generic LLM APIs cannot.
- **Lightweight Python server** – Easy to configure and extend, with no bloated dependencies.
- **Supports `llama-cpp-python`** – Run `gguf` models locally with support for GPU acceleration on:
  - NVIDIA (CUDA)
  - AMD (ROCm)
  - Apple (Metal)
- **LLM generation batching** – Combines multiple LLM requests into single requests for multi-character discourse within single calls to the LLM. 
- **Queue management** – Handles requests smartly, avoiding redundant or spammy replies and prioritizing important conversations (e.g. party/guild over world chat).
- **Attention tracking** – Bots remember who they’re talking to and avoid confusion from overlapping conversations.
- **Dynamic prompt generation** – Prompts adjust automatically based on location, nearby players, context, and chat channel.
- **Persistent memory** – Bots can have persistent personalities and memories, including relationships with other bots or players.
- **Hot-reloadable config** – Since the majority of configuration is done in Zyria itself, there is no need to restart the cMaNGOS server after making adjustments.
- **Time-based context management** - The context manager flushes old context over time, and saves guild chat context to disk upon exit so conversations can start right back up after restarting Zyria.
---

## Prerequisites

- Python 3.10+  
- A supported `gguf` model  
- Compatible system with optional GPU support (NVIDIA, AMD, or Apple M-series)

---

## 1. Clone the Zyria Server

Clone this repository wherever you'd like to run it:

```
git clone https://github.com/miranda/zyria
cd zyria
```

## 2. Set Up a Python Virtual Environment

Inside the Zyria directory:

Linux/macOS:
```
python3 -m venv venv
source venv/bin/activate
```
Windows (PowerShell or CMD):
```
python -m venv venv
.\venv\Scripts\activate
```
If python3 doesn’t work, try python.

## 3. Install llama-cpp-python with GPU Support

Zyria uses llama-cpp-python to run local models. Install it with the proper GPU backend:

NVIDIA (CUDA)
```
CMAKE_ARGS="-DLLAMA_CUDA=ON" pip install llama-cpp-python --force-reinstall --no-cache-dir
```
AMD (ROCm)
```
CMAKE_ARGS="-DLLAMA_HIPBLAS=ON" pip install llama-cpp-python --force-reinstall --no-cache-dir
```
Note: AMD support can be a bit of a pain to get working. You may need to clone the full llama.cpp repo outside of the Zyria directory, and then install with more extensive flags like so:
```
CC=clang CXX=clang++ HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
         CMAKE_ARGS="-DLLAMA_HIP=ON -DCMAKE_PREFIX_PATH=/opt/rocm -DAMDGPU_TARGETS=gfx1030 -DGGML_HIP=ON -DCMAKE_BUILD_TYPE=Release" \
         pip install --no-cache-dir --verbose ../llama-cpp-python 2>&1
```
Apple (Metal)
```
CMAKE_ARGS="-DLLAMA_METAL=ON" pip install llama-cpp-python --force-reinstall --no-cache-dir
```
You can verify GPU support is working by launching Zyria and checking for llama.cpp messages related to CUDA, HIP, or Metal initialization.

## 4. Install the remaining required Python packages:

```
pip install -r requirements.txt
```
Note: If you installed the requirements before setting the correct CMAKE_ARGS, it may build llama-cpp-python without GPU support. In that case, uninstall it and reinstall with the correct flags:
```
pip uninstall llama-cpp-python
# then reinstall with GPU flags again
```

## 5. Download and Install a Model

Place your model in the models/ directory. Recommended:

[Llama-3.1-Techne-RP-8b-v1-GGUF](https://huggingface.co/athirdpath/Llama-3.1-Techne-RP-8b-v1-GGUF?not-for-all-audiences=true)

q4_K_M or q5_K_M quantized versions are a good performance/quality balance.

## 6. Configuration

**Main Config:** `zyria.conf`

You should edit:

- `model_file_path`
- `gpu_layers`
- `threads`

Everything else has sane defaults to get you started.

### Class Overrides

To enable custom class identities for characters:

1. Copy class_overrides.json.example → class_overrides.json
2. Edit it to match your character names:
```
{
    "Arthas": "Death Knight"
}
```
This makes your TBC Paladin named Arthas think he’s a Death Knight — and others will interact with him accordingly.

### Memory Files

Zyria automatically generates `.json` files in the `memory/` folder for any bot or player it interacts with. You can edit these files to customize each bot’s personality and relationships.
Adding entries to real player character files will also enrich your interactions with bots as they will know details about you.

Start by reviewing `Example.json` inside `memory/`

**Supported Tags:**

- `{player_info}` → expands to race/class/gender summary + present/online status
- `{player_status}` → expands to only present/online status

Example:
```
"relationships": {
    "boyfriend": [
        "Raxx {player_info}"
    ],
    "friends": [
        "Zorg {player_info}"
    ]
}
```
Avoid editing the `"character":` block — it's auto-populated by Zyria during prompt generation.

## 7. Running the Server

### With included script:
```
./zyria
```
(This automatically sources your venv. Edit the script if needed.)

### Or manually:
```
source venv/bin/activate
python server.py
```

### Hooking Up to Playerbots

Make sure your cMaNGOS server is build with [playerbots-zyria](https://github.com/miranda/playerbots-zyria), and update your `aiplayerbot.conf` if needed:
```
# Enables/disables Zyria LLM Server mode. Default is 1 (enabled)
AiPlayerbot.LLMUseZyriaServer = 1

# The api endpoint that should be called for chat generation.
AiPlayerbot.LLMApiEndpoint = http://127.0.0.1:5050/zyria/v1/generate
```
Zyria uses port 5050 by default. This can be changed in zyria.conf.

- You can update zyria.conf while cMaNGOS is running — just Ctrl+C to stop Zyria, edit the file, and restart. No need to restart the server.

### Live Logging

Zyria outputs detailed logs with colorized terminal output by default. You can disable colored logs (or all logs) in the config file.

### Troubleshooting

- Zyria crashes immediately → Check your model_file_path, and ensure your model exists and is readable.
- Zyria crashes after a request → Double-check gpu_layers, threads, or try a lower quantized model.
- Bots aren't responding → Make sure:
  - Zyria is running
  - `AiPlayerBot.LLMUseZyriaServer = 1` in `aiplayerbot.conf`
  - You're not accidentally using the upstream Playerbots module

---

## Future Plans

Many more features are in the works, but not fully implemented yet.
Here are some of the next features planned to be implemented:

**Natural Language Commands**
- “Meet me in Stormwind” → bot travels there
- “Wait here” or “Let’s pull carefully” → adjusts strategy

**Extensive In-Game Context Awareness**
- Bots discuss their current quest, location, or activity

**LLM-Driven Autonomy**
- Bots may log off from chat drama
- Say they'll go do something — and actually do it

**Dynamic Guild Conversations**
- Rotating guild chat themes or daily topics

**Editable Memory via Chat**
- Tell bots to remember or forget facts in-game

### And much more to come...

---

### License

Zyria is released under the Apache 2.0 License.
Contributions, suggestions, and PRs are welcome!

---

Make your WoW bots feel alive. Zyria gives them thoughts, memories, and personality — not just responses.

