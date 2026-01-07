"""
WebSocket base classes for services
"""

import asyncio
import json
import logging
from typing import Callable, Optional, Set
from abc import ABC, abstractmethod

import websockets
from websockets.server import WebSocketServerProtocol
from websockets.client import WebSocketClientProtocol

from .protocol import Message, PORTS, get_ws_url

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')


class WSServer:
    """WebSocket server base class"""

    def __init__(self, name: str, port: int = None, host: str = "0.0.0.0"):
        self.name = name
        self.port = port or PORTS.get(name, 5550)
        self.host = host
        self.clients: Set[WebSocketServerProtocol] = set()
        self.logger = logging.getLogger(name)
        self._server = None
        self._on_message: Optional[Callable] = None

    def on_message(self, handler: Callable):
        """Decorator to set message handler"""
        self._on_message = handler
        return handler

    async def _handle_client(self, websocket: WebSocketServerProtocol):
        """Handle a client connection"""
        self.clients.add(websocket)
        client_addr = websocket.remote_address
        self.logger.info(f"Client connected: {client_addr}")

        try:
            async for raw in websocket:
                try:
                    msg = Message.from_json(raw)
                    if self._on_message:
                        result = self._on_message(msg)
                        if asyncio.iscoroutine(result):
                            await result
                except json.JSONDecodeError:
                    self.logger.warning(f"Invalid JSON: {raw[:100]}")
                except Exception as e:
                    self.logger.error(f"Handler error: {e}")
        except websockets.ConnectionClosed:
            pass
        finally:
            self.clients.discard(websocket)
            self.logger.info(f"Client disconnected: {client_addr}")

    async def broadcast(self, msg: Message):
        """Send message to all connected clients"""
        if not self.clients:
            return
        data = msg.to_json()
        await asyncio.gather(
            *[client.send(data) for client in self.clients],
            return_exceptions=True
        )

    async def send(self, websocket: WebSocketServerProtocol, msg: Message):
        """Send message to specific client"""
        try:
            await websocket.send(msg.to_json())
        except websockets.ConnectionClosed:
            self.clients.discard(websocket)

    async def start(self):
        """Start the server"""
        self._server = await websockets.serve(
            self._handle_client,
            self.host,
            self.port
        )
        self.logger.info(f"Server started on ws://{self.host}:{self.port}")
        return self._server

    async def run_forever(self):
        """Start server and run forever"""
        await self.start()
        await self._server.wait_closed()

    def run(self):
        """Blocking run"""
        asyncio.run(self.run_forever())


class WSClient:
    """WebSocket client for connecting to other services"""

    def __init__(self, name: str, target_service: str, target_host: str = "localhost"):
        self.name = name
        self.target = target_service
        self.url = get_ws_url(target_service, target_host)
        self.logger = logging.getLogger(f"{name}→{target_service}")
        self._ws: Optional[WebSocketClientProtocol] = None
        self._on_message: Optional[Callable] = None
        self._reconnect_delay = 1.0
        self._running = False

    def on_message(self, handler: Callable):
        """Decorator to set message handler"""
        self._on_message = handler
        return handler

    async def connect(self) -> bool:
        """Connect to target service"""
        try:
            self._ws = await websockets.connect(self.url)
            self.logger.info(f"Connected to {self.url}")
            return True
        except Exception as e:
            self.logger.warning(f"Connection failed: {e}")
            return False

    async def send(self, msg: Message):
        """Send message to target service"""
        if self._ws is None:
            if not await self.connect():
                return False
        try:
            await self._ws.send(msg.to_json())
            return True
        except websockets.ConnectionClosed:
            self._ws = None
            return False

    async def receive(self) -> Optional[Message]:
        """Receive a message"""
        if self._ws is None:
            return None
        try:
            raw = await self._ws.recv()
            return Message.from_json(raw)
        except websockets.ConnectionClosed:
            self._ws = None
            return None

    async def run_forever(self):
        """Connect and handle messages forever with auto-reconnect"""
        self._running = True
        while self._running:
            if not await self.connect():
                await asyncio.sleep(self._reconnect_delay)
                continue

            try:
                async for raw in self._ws:
                    try:
                        msg = Message.from_json(raw)
                        if self._on_message:
                            result = self._on_message(msg)
                            if asyncio.iscoroutine(result):
                                await result
                    except Exception as e:
                        self.logger.error(f"Handler error: {e}")
            except websockets.ConnectionClosed:
                self.logger.warning("Connection lost, reconnecting...")
                self._ws = None
                await asyncio.sleep(self._reconnect_delay)

    async def close(self):
        """Close connection"""
        self._running = False
        if self._ws:
            await self._ws.close()
            self._ws = None


class Service(ABC):
    """Base class for pipeline services"""

    def __init__(self, name: str, port: int = None):
        self.name = name
        self.server = WSServer(name, port)
        self.clients: dict[str, WSClient] = {}
        self.logger = logging.getLogger(name)

    def connect_to(self, service: str, host: str = "localhost") -> WSClient:
        """Create client connection to another service"""
        client = WSClient(self.name, service, host)
        self.clients[service] = client
        return client

    @abstractmethod
    async def setup(self):
        """Setup service (override in subclass)"""
        pass

    async def run(self):
        """Run the service"""
        await self.setup()

        # Start server
        server_task = asyncio.create_task(self.server.run_forever())

        # Start client connections
        client_tasks = [
            asyncio.create_task(client.run_forever())
            for client in self.clients.values()
        ]

        # Wait for all tasks
        await asyncio.gather(server_task, *client_tasks)

    def start(self):
        """Blocking start"""
        asyncio.run(self.run())
