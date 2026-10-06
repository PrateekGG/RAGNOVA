"""
The network guard behind scripts/verify_offline.py has to be trustworthy: a
"no step needed the network" result means nothing if the guard silently lets
traffic through, or blocks the local Ollama server it must permit.
Real sockets on loopback only; nothing leaves the machine.
"""

from __future__ import annotations

import errno
import importlib.util
import socket
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "verify_offline.py"


@pytest.fixture(scope="module")
def guard_cls():
    spec = importlib.util.spec_from_file_location("verify_offline", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.NetworkGuard


def test_guard_fails_a_dns_lookup_for_a_remote_name_and_records_it(guard_cls):
    with guard_cls() as guard:
        with pytest.raises(socket.gaierror):
            socket.getaddrinfo("huggingface.co", 443)
    assert ("dns lookup", "huggingface.co") in guard.blocked


# 192.0.2.0/24 is reserved for documentation (RFC 5737): even if the guard were
# broken, a connection attempt to it can never reach a real server.
REMOTE = ("192.0.2.1", 80)


def test_guard_fails_connect_to_a_remote_address_with_its_own_error_and_records_it(guard_cls):
    with guard_cls() as guard:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        try:
            with pytest.raises(OSError) as raised:
                sock.connect(REMOTE)
        finally:
            sock.close()
    # the guard's own error, not a real network timing out
    assert raised.value.errno == errno.ENETUNREACH
    assert guard.blocked == [("connect", "192.0.2.1:80")]


def test_guard_fails_connect_ex_to_a_remote_address_with_its_own_error_and_records_it(guard_cls):
    with guard_cls() as guard:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        try:
            result = sock.connect_ex(REMOTE)
        finally:
            sock.close()
    assert result == errno.ENETUNREACH
    assert guard.blocked == [("connect", "192.0.2.1:80")]


def test_guard_allows_loopback_so_the_local_ollama_server_still_works(guard_cls):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    try:
        with guard_cls() as guard:
            client = socket.create_connection(("127.0.0.1", port), timeout=2)
            client.close()
            assert socket.getaddrinfo("localhost", port)
    finally:
        server.close()
    assert guard.blocked == []


def test_guard_puts_the_socket_module_back_exactly_as_it_found_it(guard_cls):
    connect, connect_ex, getaddrinfo = socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo
    with guard_cls():
        assert socket.socket.connect is not connect
    assert socket.socket.connect is connect
    assert socket.socket.connect_ex is connect_ex
    assert socket.getaddrinfo is getaddrinfo
