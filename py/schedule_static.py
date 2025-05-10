import argparse
from tqdm import tqdm
import asyncio
import random

from pydantic import BaseModel
import yaml
from utils.dao import DAO, WeightServerClient
from utils.storage import setup_storage, with_storage_args
from utils.base import get_db_pool, getLogger
from utils.dao import ModelVital

logger = getLogger(__name__)


class Node(BaseModel):
    # gRPC only now
    hostname: str
    address: str
    channel: str = "grpc"


class Inventory(BaseModel):
    nodes: list[Node]


class StaticScheduler:
    def __init__(self, inventory: Inventory, split_plan: ModelVital):
        self.inventory = inventory
        self.plan = split_plan

    def schedule(self, task):

        return self.inventory.nodes[0]


def parse_args():
    parser = argparse.ArgumentParser(description="create model")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--instance_name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--inventory",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    return args


async def upsert_nodes(inventory: str, dao: DAO):
    inventory = yaml.safe_load(open(inventory, "r"))
    inv = Inventory(**inventory)
    logger.info(f"inventory loaded, total_nodes={len(inv.nodes)}")
    nodes = inv.nodes
    res = {}
    for node in nodes:
        nid = await dao.node.upsert(
            hostname=node.hostname,
            device="cpu",
            config={"channel": node.channel, "addr": node.address},
        )
        res[node.hostname] = nid
    return res


async def main():
    args = parse_args()
    p = await get_db_pool()
    ws = ""
    async with p.connection() as conn:
        q = DAO(conn)
        model = await q.model.by_name(args.model_name)
        if model is None:
            raise ValueError(f"model {args.model_name} not found")
        if model["config"] is None or model["config"]["weight_server"] is None:
            raise ValueError(f"model {args.model_name} is not proper configured")
        ws = model["config"]["weight_server"]
    wscli = WeightServerClient(ws)
    model_name = args.model_name
    vital = wscli.vital(model_name=args.model_name)
    experts = []
    for layer in range(vital.moe_layers[0], vital.moe_layers[1]):
        for idx in range(vital.routed_experts):
            experts.append(f"{model_name}/l{layer}-e{idx}")
    logger.info(f"total experts {len(experts)}")
    async with p.connection() as conn:
        q = DAO(conn)
        nodes = await upsert_nodes(args.inventory, q)
        model = await q.model.by_name(args.model_name)
        instance_id = await q.instance.upsert(
            model_id=model["id"], name=args.instance_name
        )
        hostnames = list(nodes.keys())
        with tqdm(total=len(experts), desc="scheduling") as pbar:
            for expert in experts:
                selected = random.choice(hostnames)
                node_id = nodes[selected]
                pbar.update(1)
                await q.expert.upsert(
                    instance_id=instance_id,
                    node_id=node_id,
                    expert_id=expert,
                    replica=1,
                )
    return ws


if __name__ == "__main__":
    asyncio.run(main())
