Your task is to develop a telegram group chat bot in python that interacts with group members. Upon activation, it assembles the messages into a prompt, send the prompt to a backend openai compatible AI api, get response, optionally calls the corresponding tools, and send back the response message. There are following features to be implemented:

# Load config file

A config file is a yaml file. The default path should be `config/config.yaml`. It should be loaded at startup. It contains the following information:
1. Bot token
2. Connection API: openai api url, api key, model name
3. Model params: temperature, reasoning effort, other params that needs to be sent to the backend.
4. Context management: max context window, max ai response token
5. Chat white list: chat ids that the bot can be activated
6. Tool settings: memory capacity, short-term mem expiration days, etc

# System prompt

A system prompt file with default path `config/system_prompt.txt`. It should be loaded at startup time. As the name suggests, it contains the system prompt of the LLM request. It also support the following macro:
`{{short-mem}}`: The content of the short term memory
`{{long-mem}}`: The content of the long term memory
`{{user-info}}`: The content of the known user info

# Activation and Chat history management

The bot is activated under 2 conditions:
1. A user calls `/pepper`
2. A user replies to a previous bot message

A new chat history list is created when the bot is activated with command `/pepper`. 
Each message should be formatted in the following way:
`[msg {message-id}] {user-name} (reply to msg {ref-message-id}): {message-content}`
For example: 
```
(role: user:) [msg 01] Locky: Hello!
(role: assistant) [msg 02] Pepper (reply to msg 01): Hello!
(role: assistant) [msg 03] Locky (reply to msg 02): How's it going?
```

If a user message activates bot through `/pepper`, and it is replying to a message that's not included in a chat history, attach the referenced message as the first message.
For example:
```
(role: user) [msg 00] Ying: ah i'm tired today. \n[msg 01] Locky (reply to msg 00): 
```

Internally the message sender user id should be kept. When assembling a prompt to AI model, if the user id is from a known user, its name is placed at `{user-name}`. Otherwise, it should be formatted like `unknown-user {id}`. 

The bot should check whether the backend ai model accepts image input and attach the received image with message if supported.

The response from AI model is sent back to telegram as a reply to the activation user message.

Chat histories should have a configurable expiration period in config file. Currently active histories should be synced to file system at `data/chat-histories.json` for quick recovery.

# Short-term memory

Short term memory is provided as a tool to the ai model to take notes of what happened. The tool should take a single string as a parameter. Internally, the tool should manage a list of of events. When a preset contains `{{short-mem}}`, it is replaced with the memory content.

Each event should expire after a configurable time period. Upon expiration, a separate prompt should be used to ask the ai model to decide whether it should change the long-term memory, modify known user info, or simply drop (forget) the event.

Upon changes, the short term memory should be synced to the file system at `data/short-term.txt` for debugging and failure recovery. Load it at startup time.

# Long-term memory

Long term memory is a list of event replacing the `{{long-mem}}` macro in the system prompt. It is only modified at short-term memory expiration.

Upon changes, it should be synced to the file system at `data/long-term.txt`. It should also be loaded at startup.

# User info

A special memory system that contains user information that is not likely to change rapidly, including the personality, habits, hobbies, and the bot's own judgement. 

A user info entry contains the telegram user ID (e.g. `1012161237`), a description, and a name that the bot should call the user with.

A tool should be implemented that takes the user ID, description and name as parameters.

This should be synced with the file system at `data/known-users.yaml`.

# Other requirement

The bot features should be easily testable, for example, memory system should not be forced to wait for expiration for testing.
Code style should be clean, use type hints whenever it is feasible.
Include a dockerfile and a compose yaml for easy docker deployment
