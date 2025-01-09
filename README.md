# Overview

[!documentation/chat-ai.gif](!documentation/chat-ai.gif)

This project provides a client/UI for interacting with LLMs (e.g. OpenAI, Claude, or local models via LM Studio or other lamma.cpp-based services). It runs a gRPC inference and resource (e.g. file attachment) service.

- Chat with several models within the same conversation.
- "Attach" websites, local files, local directories
    - changes to local files/directories automatically update in the resource service
    - RAG is used when attached context is larger than specified threshold
- Predefined prompts, instructions
- Define "contexts" which are collections of attachments with optional instructions.
- Combine multiple contexts, instructions in a single conversation; dynamically add/remove.

# Running the Client/Server

- add `.env` file with `OPENAI_API_KEY` and `ATHROPIC_API_KEY` keys/tokens.
- run `make electron-setup`
- run `make run`
    - or start server with `make run-server` and start app with `make run-app`

# Caveats / Gotchas

- Cannot do `shift+enter` for new line in MacOS terminal because it doesn't support "extended key protocol".
    - Use `iterm`
- In `iterm` you have to hold down `option` in order to select tesxt to copy from messages. 
    - (`shift` in windows according to textual FAQ?; haven't tried)


## Testing

- `timeout = 5` is used to ensure github actions don't run indefinitely if a test hangs
    - As a result, debugging unit tests will also timeout after 5 seconds. To fix in VS Code, add `"args": [ "--timeout=0" ]` to your launch.json. For example,

        ```
        {
            "version": "0.2.0",
            "configurations": [
            
                {
                    "name": "Python: Debug Tests",
                    "type": "debugpy",
                    "request": "launch",
                    "program": "${file}",
                    "purpose": ["debug-test"],
                    "console": "integratedTerminal",
                    "justMyCode": false,
                    "env": {
                        "PYTEST_TIMEOUT": "0"
                    },
                    "args": [
                        "--timeout=0"
                    ]
                }
            ]
        }
        ```

## Running

- install brew
    - `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`
- install node
    - `brew install node`
- install `uv`
    - `pip install uv`
- run server: `make run-server`
- run electron app:
    - `make electron-setup`
    - `make run-electron`



# Electron

- `brew install node`
- .env file
- npm install  # This installs everything from package.json into node_modules
- npm start



# Gotchas

- Cancelling the request doesn't add tokens/costs to summary.



## TODO

- [ ] Clean up unused resources periodically (e.g. stored files and chunks)

**Models**

- [ ] Multiple Local Models
- [ ] Claude
- [ ] OpenAI images
- [ ] Claude images

**History**

- [ ] `Clear History` button
- [ ] A new item in history should only be added after the first message is sent and there is an actual history item. E.g. we could just call `Get History` every time instead of maintaining on both the client and server, but i'm not a huge fan of that, i'd rather call once at client startup. (and e.g. branch conversation)
- [ ] Semanatic/Keyword Search
- [ ] We are only storing model-config-id with message/history but if the configuration is deleted then the client won't know what model/parameters were used
    - [ ] Same if a ModelInfo is deleted or no longer supported

**Chat Messages**

- [ ] Expandable chat text box (vertically)
- [ ] `Branch Conversation` from Assitant Response
- [ ] `Regenerate` Assistant Response/Message
- [ ] `Edit` User/Assistant Message (which updates main conversation on server)
- [ ] Tool tips over icons (e.g. `Copy Full Message`, `Copy Code Snippet`)
- [ ] Format math equations
- [ ] Should `Summary` disappear if using OpenAI Server?
    - [ ] What happens in UI on error (e.g. exceeded context limit)?
    - [ ] Perhaps move summary into side bar with other log messages/events from server
- [ ] What happens if there is an error from api like OpenAI or grpc in general (can test by raising exception?)

**Prompt**

- [x] Clicking play button should append prompt to any existing text in chat box

**Resources**

- [ ] `Clear All Resources` button
- [ ] If you pass in a directory doesn't have `.gitignore` (e.g. subdirectory of the project) then it will still probably include a bunch of non-hidden files/directories that are not wanted (e.g. `__pycache__`, etc.). We probably want to create a list of common files/directories/patterns to ignore even without `.gitignore` file.

**Server**

- [ ] Get models (family e.g. OpenAI name e.g. gpt-4o)

- [ ] Get History
- [ ] Clear History

- [ ] Get prompts

- [ ] Get Resources
- [ ] Add Resource
- [ ] Delete Resource
- [ ] Need to decide when detect changes in files and make corresponding updates to e.g. vector database. For example it might be better to wait until the resource is needed because A) the resource may be changed frequently but rarely used and B) the same resource could either be used entirely or chunked depending on context type, or file type/size, etc.)  e.g. we would not want to chunk/rag on code files that are being directly used
- [ ] `_unique_conv_id_locks` will continue to fill up indefinitely until the server is restarted

**Sidebar**

- [ ] Sidebar collapse stops working after size is adjusted.
- [ ] Prompts/Instructions/Context
    - [ ] filter
    - [ ] sort
    - [ ] manual order
    - [ ] categorize

**Misc.**

- [ ] Stream Update events to client; client sidebar should have new tab for seeing updates

**Resources**

- [ ] Need a way to clean up resources (e.g. unused)
    - [ ] perhaps track operations in manager and clean up after every N operations in one of the workers.
- [ ] Need a way for client to view/delete resources?? Update resources (e.g. individually; resources not used in last N days; etc)
    - [ ] for web-page resources, we need a way to update the resource (for local files, we check if the contents have changed; we don't have an equivalent for web resource; perhaps check when it was last scrapped?)


## Ideas

- [ ] Chat with Github Repo (treated as special type of website resource?)
