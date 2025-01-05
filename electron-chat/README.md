# Chat App

## Run

```
make run-server
make electron-setup
make run-electron
```

## Code / Contributing

### Electron Chat App

Most of the javascript generated is from ChatGPT/Claude. It's mostly stuffed into one file. (Yeah... I know..) Feel free to refactor and send a PR.

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
