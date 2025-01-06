# Using

- add .env file with `OPENAI_API_KEY` and `ATHROPIC_API_KEY`.
- run `make electron-setup`

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