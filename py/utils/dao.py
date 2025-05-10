import json
import requests
from utils.base import ConnType
from pydantic import BaseModel
import typing


class ModelVital(BaseModel):
    moe_layers: typing.Tuple[int, int]
    routed_experts: int
    hidden_dim: int
    inter_dim: int


class WeightServerClient:

    def __init__(self, addr: str):
        self.base = addr

    def vital(self, model_name: str):
        url = f"{self.base}/meta/vital/{model_name}"
        res = requests.get(url)
        if res.status_code != 200:
            raise Exception(f"failed to get vital from {url}")
        return res.json()


class ModelDAO:
    def __init__(self, conn: ConnType):
        self.conn = conn

    async def by_name(self, name: str):
        res = await self.conn.execute("SELECT * FROM model WHERE name = %s", (name,))
        res = await res.fetchone()
        return res

    async def upsert(self, *, name: str, config: dict):
        res = await self.conn.execute(
            """INSERT INTO model (name, config) VALUES (%s,%s )
            ON CONFLICT (name) DO UPDATE SET
                config = EXCLUDED.config
            RETURNINg id
            """,
            (name, json.dumps(config)),
        )
        id = await res.fetchone()
        return id["id"]


class InstanceDAO:
    def __init__(self, conn: ConnType):
        self.conn = conn

    async def by_name(self, name: str):
        res = await self.conn.execute("SELECT * FROM instance WHERE name = %s", (name,))
        res = await res.fetchone()
        return res

    async def upsert(self, model_id: int, name: str):
        res = await self.conn.execute(
            """INSERT INTO instance (model_id, name)
              VALUES (%s,%s)
              ON CONFLICT (name) DO NOTHING
              RETURNING id
            """,
            (model_id, name),
        )
        res = await res.fetchone()
        if res is None:
            res = await self.by_name(name)
        return res["id"]


class ExpertDAO:
    def __init__(self, conn: ConnType):
        self.conn = conn

    async def upsert(
        self, instance_id: int, node_id: int, expert_id: str, replica: int
    ):
        res = await self.conn.execute(
            """INSERT INTO expert 
                (instance_id, node_id, expert_id, replica,state)
                VALUES (%s,%s,%s,%s,%s)
                ON CONFLICT (instance_id,node_id,expert_id) DO UPDATE SET
                    replica = EXCLUDED.replica
                    RETURNING id
                """,
            (instance_id, node_id, expert_id, replica, "{}"),
        )
        id = await res.fetchone()
        return id["id"]


class NodeDAO:
    def __init__(self, conn: ConnType):
        self.conn = conn

    async def upsert(self, hostname: str, device: str, config: dict):
        res = await self.conn.execute(
            """INSERT INTO node 
                (hostname,device,config) 
                    VALUES (%s,%s,%s) 
                ON CONFLICT (hostname) DO UPDATE SET
                    device = EXCLUDED.device,
                    config = EXCLUDED.config
                RETURNING id
                """,
            (hostname, device, json.dumps(config)),
        )
        res = await res.fetchone()

        return res["id"]


class DAO:
    def __init__(self, conn: ConnType):
        self.conn = conn
        self.model = ModelDAO(conn)
        self.instance = InstanceDAO(conn)
        self.expert = ExpertDAO(conn)
        self.node = NodeDAO(conn)
