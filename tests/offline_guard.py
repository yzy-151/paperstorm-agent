"""Explicit network guard for hermetic test runs."""

import os
import ipaddress
import socket


def install_offline_test_guard():
    for name, value in {
        "PAPERSTORM_CHAT_LLM": "0",
        "PAPERSTORM_JUDGE_LLM": "0",
        "PAPERSTORM_ROUTER_LLM": "0",
        "PAPERSTORM_RETRIEVAL_EMBEDDING": "hash",
    }.items():
        os.environ.setdefault(name, value)

    original = socket.create_connection
    original_connect = socket.socket.connect

    def blocked_create_connection(address, *args, **kwargs):
        if _is_loopback(address):
            return original(address, *args, **kwargs)
        raise RuntimeError("offline test blocked network connection")

    def blocked_connect(client, address):
        if _is_loopback(address):
            return original_connect(client, address)
        raise RuntimeError("offline test blocked network connection")

    socket.create_connection = blocked_create_connection
    socket.socket.connect = blocked_connect

    def restore():
        socket.create_connection = original
        socket.socket.connect = original_connect

    return restore


def _is_loopback(address):
    host = address[0] if isinstance(address, tuple) and address else address
    if str(host).lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False
