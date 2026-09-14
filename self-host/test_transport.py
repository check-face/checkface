"""Fast real-socket transport tests; model correctness is covered by smoke.py."""
import contextlib
import io
import json
import socket
import struct
import threading
import time
import unittest
from urllib.request import urlopen
from api import BoundedHTTPServer, handler


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.server = BoundedHTTPServer(('127.0.0.1', 0), handler(None))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.address = self.server.server_address

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def health(self):
        with urlopen(f'http://127.0.0.1:{self.address[1]}/healthz', timeout=2) as response:
            return json.loads(response.read())

    def wait_slots(self, count):
        deadline = time.monotonic() + 2
        while self.server.slots._value != count:
            if time.monotonic() >= deadline:
                self.fail(f'Expected {count} available handler slots, got {self.server.slots._value}')
            time.sleep(0.005)

    def test_health_without_inference(self):
        self.assertTrue(self.health()['ready'])
        self.assertFalse(self.health()['inference_busy'])

    def test_handler_limit_and_recovery(self):
        sockets = []
        try:
            for _ in range(8):
                connection = socket.create_connection(self.address, timeout=2)
                connection.sendall(b'GET /healthz HTTP/1.1\r\n')  # Deliberately unfinished headers.
                sockets.append(connection)
            self.wait_slots(0)
            with socket.create_connection(self.address, timeout=2) as excess:
                response = excess.recv(1024)
                self.assertIn(b'503 Service Unavailable', response)
                self.assertIn(b'Retry-After: 1', response)
        finally:
            for connection in sockets:
                connection.close()
        self.wait_slots(8)
        self.assertTrue(self.health()['ready'])

    def test_disconnected_client_does_not_log_traceback(self):
        errors = io.StringIO()
        with contextlib.redirect_stderr(errors):
            connection = socket.create_connection(self.address, timeout=2)
            connection.sendall(b'GET /healthz HTTP/1.1\r\nHost: localhost\r\n\r\n')
            connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack('ii', 1, 0))
            connection.close()  # Reset while the server is preparing/writing its response.
            self.assertTrue(self.health()['ready'])
            self.wait_slots(8)
        self.assertNotIn('Traceback', errors.getvalue())


if __name__ == '__main__':
    unittest.main()
