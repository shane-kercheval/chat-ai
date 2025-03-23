.PHONY: tests app proto

####
# Environment
####
build-env:
	uv sync --refresh

run: proto electron-proto
	uv run python launch.py

####
# Project
####
run-server: proto
	PYTHONPATH=$PYTHONPATH:.:./proto/generated uv run server/grpc_service.py $(ARGS)

run-server-default-mcp: proto
	PYTHONPATH=$PYTHONPATH:.:./proto/generated \
		uv run server/grpc_service.py \
		--mcp-config '/Users/shanekercheval/repos/chat-ai/server/mcp/mcp_fake_server_config.json' \
		$(ARGS)

run-server-jupyter-mcp: proto
	# uv pip install jupyterlab jupyter-collaboration ipykernel
	# uv run jupyter lab --port 8888 --IdentityProvider.token MY_TOKEN --ip 0.0.0.0
	# start docker e.g. docker desktop or daemon
	PYTHONPATH=$PYTHONPATH:.:./proto/generated \
		uv run server/grpc_service.py \
		--mcp-config '/Users/shanekercheval/repos/chat-ai/server/mcp/mcp_jupyter_config.json' \
		$(ARGS)

run-client: electron-proto
	cd client && npm start

proto: clean-proto
	uv run -m grpc_tools.protoc \
		-Iproto \
		--python_out=proto/generated \
		--grpc_python_out=proto/generated \
		proto/chat.proto

linting:
	uv run ruff check server
	uv run ruff check tests

test-context-agent:
	uv run pytest --durations=0 --durations-min=0.1 tests/test_context_strategy_agent.py

test-server: proto
	uv run pytest --durations=0 --durations-min=0.1 tests/test_grpc_service.py

unittests: proto
	uv run pytest --durations=0 --durations-min=0.1 tests

tests: linting unittests

clean-proto:
	rm -rf proto/generated/chat_pb2.py
	rm -rf proto/generated/chat_pb2_grpc.py
	rm -rf client/proto/generated/chat_pb.js
	rm -rf client/proto/generated/chat_grpc.js

clean: clean-proto
	rm -rf .venv
	rm -rf .ruff_cache
	rm -rf server/__pycache__
	rm -rf tests/__pycache__
	rm -rf client/node_modules
	rm -rf data/chat.db

####
# Electron
####
electron-setup: electron-clean
	cd client && rm -rf node_modules && npm install

electron-proto: proto
	mkdir -p client/proto/generated
	cp proto/chat.proto client/proto/
	cd client && \
		npx grpc_tools_node_protoc \
		--js_out=import_style=commonjs,binary:./proto/generated \
		--grpc_out=grpc_js:./proto/generated \
		--proto_path=./proto \
		./proto/chat.proto

electron-clean:
	rm -rf client/proto/generated/*
	rm -f client/proto/chat.proto
