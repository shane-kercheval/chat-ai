"""Launches the electron app and the gRPC server in parallel."""
import subprocess
import time
import sys
import grpc
import signal


def start_server() -> subprocess.Popen:
    """Starts the gRPC server."""
    return subprocess.Popen(
        ["make", "run-server"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def check_server_ready() -> bool:
    """Checks if the gRPC server has started."""
    try:
        channel = grpc.insecure_channel('localhost:50051')
        future = grpc.channel_ready_future(channel)
        future.result(timeout=1)
        return True
    except:  # noqa: E722
        return False


def main() -> None:
    """Launches the electron app and the gRPC server in parallel."""
        # Start electron app
    electron_process = subprocess.Popen(
        ["make", "run-client"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    server_process = start_server()
    # Wait for server to be ready
    max_attempts = 10
    for _ in range(max_attempts):
        if check_server_ready():
            break
        time.sleep(1)
    else:
        print("Server failed to start")
        server_process.kill()
        sys.exit(1)

    def cleanup(signum, frame) -> None:  # noqa: ANN001, ARG001
        """Cleans up the processes."""
        electron_process.terminate()
        server_process.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    electron_process.wait()
    server_process.terminate()

if __name__ == "__main__":
    main()