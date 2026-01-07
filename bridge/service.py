"""
Bridge WebSocket Service
Central hub that aggregates all service messages
External clients (TouchDesigner, Web) connect here
"""

import argparse
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import WSServer, WSClient, Message, MessageType, Source, PORTS


class BridgeService:
    def __init__(self, args):
        self.args = args
        self.server = WSServer("bridge", args.port)
        self.clients = {}  # service_name -> WSClient
        self.running = False
        self.message_count = 0

    def create_forwarder(self, service_name: str):
        """Create a message forwarder for a service"""
        async def forward(msg: Message):
            self.message_count += 1
            # Add bridge metadata
            msg.data["_bridge_seq"] = self.message_count
            # Broadcast to all external clients
            await self.server.broadcast(msg)

            # Readable console output
            source = msg.source.upper() if msg.source else "UNKNOWN"
            msg_type = msg.type.upper() if msg.type else "UNKNOWN"
            print(f"╔═══ Bridge [{self.message_count}] ═══")
            print(f"║ From: {source} | Type: {msg_type}")
            print(f"║ Broadcast to {len(self.server.clients)} clients")
            print(f"╚═══════════════════════════════")
        return forward

    async def connect_service(self, name: str, host: str = "localhost"):
        """Connect to a pipeline service"""
        client = WSClient("bridge", name, host)

        @client.on_message
        async def on_msg(msg):
            await self.create_forwarder(name)(msg)

        self.clients[name] = client
        print(f"Bridge: will connect to {name} at {client.url}")
        return client

    async def run(self):
        """Run the service"""
        self.running = True

        # Connect to all pipeline services (ASR and SLM by default)
        if not self.args.no_asr:
            await self.connect_service("asr", self.args.asr_host)
        if not self.args.no_slm:
            await self.connect_service("slm", self.args.slm_host)
        if self.args.enable_state:
            await self.connect_service("state", self.args.state_host)
        if not self.args.no_t2i:
            await self.connect_service("t2i", self.args.t2i_host)

        # Handle messages from external clients
        @self.server.on_message
        async def on_external(msg: Message):
            """Handle commands from external clients (TouchDesigner, Web)"""
            msg_type = msg.type.value if hasattr(msg.type, "value") else str(msg.type)
            print(f"Bridge: external command: {msg_type}")

            if msg_type == MessageType.PING.value:
                pong = Message(
                    type=MessageType.PONG,
                    source=Source.BRIDGE,
                    data={"services": list(self.clients.keys())}
                )
                await self.server.broadcast(pong)
                return

            # Forward config updates to the target service (if connected)
            if msg_type == MessageType.CONFIG_UPDATE.value:
                target = msg.data.get("service")
                if target in self.clients:
                    ok = await self.clients[target].send(msg)
                    print(f"Bridge: forwarded config_update to {target} ({'ok' if ok else 'fail'})")
                else:
                    print(f"Bridge: target service not connected: {target}")

        # Start server
        await self.server.start()

        # Send ready status
        status_msg = Message.status(Source.BRIDGE, "ready", {
            "services": list(self.clients.keys()),
            "port": self.args.port
        })
        await self.server.broadcast(status_msg)

        print(f"Bridge ready on ws://0.0.0.0:{self.args.port}")
        print(f"External clients (TouchDesigner/Web) connect here")

        # Run all service clients concurrently
        tasks = [
            asyncio.create_task(client.run_forever())
            for client in self.clients.values()
        ]

        if tasks:
            await asyncio.gather(*tasks)
        else:
            # No services to connect to, just keep server running
            while self.running:
                await asyncio.sleep(1)

    def start(self):
        """Blocking start"""
        try:
            asyncio.run(self.run())
        except KeyboardInterrupt:
            print("\nBridge service stopped")
            self.running = False


def main():
    parser = argparse.ArgumentParser(description='Bridge WebSocket Service')
    parser.add_argument('--port', type=int, default=PORTS["bridge"])

    # Service connections (enable/disable)
    parser.add_argument('--no-asr', action='store_true',
                        help='Disable ASR service connection')
    parser.add_argument('--no-slm', action='store_true',
                        help='Disable SLM service connection')
    parser.add_argument('--enable-state', action='store_true',
                        help='Enable State service connection')
    # 默认连接 T2I，如需禁用使用 --no-t2i
    parser.add_argument('--no-t2i', action='store_true',
                        help='Disable T2I service connection')

    # Service hosts
    parser.add_argument('--asr-host', default='localhost')
    parser.add_argument('--slm-host', default='localhost')
    parser.add_argument('--state-host', default='localhost')
    parser.add_argument('--t2i-host', default='localhost')

    args = parser.parse_args()
    service = BridgeService(args)
    service.start()


if __name__ == "__main__":
    main()
