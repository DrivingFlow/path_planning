#!/usr/bin/env python3
import argparse
import socket
import sys
import termios
import time
import tty


def connect_with_retry(server_ip: str, port: int, initial_delay: float = 0.5) -> socket.socket:
    delay = initial_delay
    while True:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((server_ip, port))
            print("[CLIENT] Connection successful")
            return s
        except (ConnectionRefusedError, socket.error):
            time.sleep(delay)
            delay = min(delay * 2, 5)


def wait_for_keypress(target: str = "r") -> bool:
    """Block until target key is pressed. Returns False if 'q' is pressed."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while True:
            ch = sys.stdin.read(1)
            if ch == target:
                return True
            if ch == "q":
                return False
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def main() -> None:
    parser = argparse.ArgumentParser(description="Request remote recording/transcription.")
    parser.add_argument("--server-ip", default="10.43.153.168", help="Remote server IP")
    parser.add_argument("--port", type=int, default=8000, help="Remote server port")
    parser.add_argument("--timeout", type=float, default=60.0, help="Receive timeout in seconds")
    args = parser.parse_args()

    with connect_with_retry(args.server_ip, args.port) as s:
        s.settimeout(args.timeout)
        print("[CLIENT] Press 'r' to request transcription, 'q' to quit.")

        while True:
            should_request = wait_for_keypress("r")
            if not should_request:
                print("[CLIENT] Exiting.")
                return

            # Send transcribe request
            s.sendall(b"transcribe")
            print("[CLIENT] Sent 'transcribe' — waiting for server to record & respond...")

            # Wait for transcript (server will record via spacebar then transcribe)
            try:
                data = s.recv(4096)
                if not data:
                    print("[CLIENT] Server disconnected.")
                    return
                transcript = data.decode(errors="replace").strip()
                print(f"[CLIENT] Transcript: {transcript}\n")
                print("[CLIENT] Press 'r' to request again, 'q' to quit.")
            except socket.timeout:
                print("[CLIENT] Timed out waiting for response.")


if __name__ == "__main__":
    main()