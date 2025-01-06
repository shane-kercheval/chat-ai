.PHONY: tests app proto

####
# Environment
####
build-env:
	uv sync

run: proto electron-proto
	uv run python launch.py

####
# Project
####
run-server: proto
	PYTHONPATH=$PYTHONPATH:.:./proto/generated uv run server/grpc_service.py

run-app: electron-proto
	cd electron-chat && npm start

proto: clean-proto
	uv run -m grpc_tools.protoc \
		-Iproto \
		--python_out=proto/generated \
		--grpc_python_out=proto/generated \
		proto/chat.proto

linting:
	uv run ruff check server
	uv run ruff check tests

test-server: proto
	uv run pytest --durations=0 --durations-min=0.1 tests/test_grpc_service.py

unittests: proto
	uv run pytest --durations=0 --durations-min=0.1 tests

tests: linting unittests

clean-proto:
	rm -rf proto/generated/chat_pb2.py
	rm -rf proto/generated/chat_pb2_grpc.py
	rm -rf electron-chat/proto/generated/chat_pb.js
	rm -rf electron-chat/proto/generated/chat_grpc.js

clean: clean-proto
	rm -rf .venv
	rm -rf .ruff_cache
	rm -rf server/__pycache__
	rm -rf tests/__pycache__

####
# Electron
####
electron-setup: electron-clean
	cd electron-chat && npm install

electron-proto: proto
	mkdir -p electron-chat/proto/generated
	cp proto/chat.proto electron-chat/proto/
	cd electron-chat && \
		npx grpc_tools_node_protoc \
		--js_out=import_style=commonjs,binary:./proto/generated \
		--grpc_out=grpc_js:./proto/generated \
		--proto_path=./proto \
		./proto/chat.proto

electron-clean:
	rm -rf electron-chat/proto/generated/*
	rm -f electron-chat/proto/chat.proto
