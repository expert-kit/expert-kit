#!/usr/bin/env bash
# provision-aliyun.sh — Launch Aliyun ECS Spot worker nodes for Expert Kit.
#
# Called by the ElasticManager provisioner when a capacity shortfall is detected:
#   provision-aliyun.sh --count <N> --model <model> --instance <instance>
#
# Required environment variables (set in the controller environment):
#   ALIYUN_REGION          e.g. cn-hangzhou
#   ALIYUN_ZONE            e.g. cn-hangzhou-h
#   ALIYUN_SECURITY_GROUP  Security group ID
#   ALIYUN_VSWITCH         VSwitch ID
#   ALIYUN_INSTANCE_TYPE   e.g. ecs.gn6i-c4g1.xlarge (T4 GPU)
#   ALIYUN_IMAGE_ID        ECS image with EK worker pre-installed
#   ALIYUN_KEY_PAIR        SSH key pair name
#   EK_CONTROLLER_ADDR     Controller intra-node gRPC address (host:5001)
#   EK_WEIGHT_SERVER_ADDR  Weight server HTTP address (http://host:6543)
#   EK_DB_DSN              Postgres DSN for the worker config
#   EK_CONFIG_TEMPLATE     Path to aliyun.config.yaml template (default: dev/aliyun.config.yaml)
#   ANSIBLE_PLAYBOOK       Path to ansible playbook (default: ek-solution/ansible/index.yaml)
#   EK_CLI_BIN             Path to ek-cli binary (default: ./target/release/ek-cli)

set -euo pipefail

# ── Parse arguments ────────────────────────────────────────────────────────────
COUNT=1
MODEL=""
INSTANCE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --count)    COUNT="$2";    shift 2 ;;
        --model)    MODEL="$2";    shift 2 ;;
        --instance) INSTANCE="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$MODEL" || -z "$INSTANCE" ]]; then
    echo "Usage: $0 --count N --model MODEL --instance INSTANCE" >&2
    exit 1
fi

echo "[provision] count=$COUNT model=$MODEL instance=$INSTANCE"

# ── Defaults ───────────────────────────────────────────────────────────────────
ALIYUN_REGION="${ALIYUN_REGION:-cn-hangzhou}"
ALIYUN_ZONE="${ALIYUN_ZONE:-cn-hangzhou-h}"
ALIYUN_INSTANCE_TYPE="${ALIYUN_INSTANCE_TYPE:-ecs.gn6i-c4g1.xlarge}"
EK_CONFIG_TEMPLATE="${EK_CONFIG_TEMPLATE:-dev/aliyun.config.yaml}"
ANSIBLE_PLAYBOOK="${ANSIBLE_PLAYBOOK:-ek-solution/ansible/index.yaml}"
EK_CLI_BIN="${EK_CLI_BIN:-./target/release/ek-cli}"

# ── Launch ECS Spot instances ──────────────────────────────────────────────────
echo "[provision] launching $COUNT ECS Spot instance(s) in $ALIYUN_REGION/$ALIYUN_ZONE"

INSTANCE_IDS=$(aliyun ecs RunInstances \
    --RegionId "$ALIYUN_REGION" \
    --ZoneId "$ALIYUN_ZONE" \
    --InstanceType "$ALIYUN_INSTANCE_TYPE" \
    --ImageId "${ALIYUN_IMAGE_ID:?ALIYUN_IMAGE_ID not set}" \
    --SecurityGroupId "${ALIYUN_SECURITY_GROUP:?ALIYUN_SECURITY_GROUP not set}" \
    --VSwitchId "${ALIYUN_VSWITCH:?ALIYUN_VSWITCH not set}" \
    --KeyPairName "${ALIYUN_KEY_PAIR:?ALIYUN_KEY_PAIR not set}" \
    --SpotStrategy SpotAsPriceGo \
    --SpotDuration 0 \
    --Amount "$COUNT" \
    --output cols=InstanceIdSets.InstanceIdSet[] rows | tr -d '[],"' | tr ' ' '\n' | grep -v '^$')

if [[ -z "$INSTANCE_IDS" ]]; then
    echo "[provision] ERROR: RunInstances returned no instance IDs" >&2
    exit 1
fi

echo "[provision] launched instances: $(echo $INSTANCE_IDS | tr '\n' ' ')"

# ── Wait for instances to reach Running state ──────────────────────────────────
for IID in $INSTANCE_IDS; do
    echo "[provision] waiting for $IID to reach Running state..."
    for attempt in $(seq 1 30); do
        STATUS=$(aliyun ecs DescribeInstances \
            --RegionId "$ALIYUN_REGION" \
            --InstanceIds "[\"$IID\"]" \
            --output cols=Instances.Instance[0].Status rows 2>/dev/null | tail -1 || echo "unknown")
        if [[ "$STATUS" == "Running" ]]; then
            echo "[provision] $IID is Running"
            break
        fi
        if [[ "$attempt" -eq 30 ]]; then
            echo "[provision] ERROR: $IID did not reach Running within 5 minutes" >&2
            exit 1
        fi
        sleep 10
    done
done

# ── Collect public IPs ─────────────────────────────────────────────────────────
declare -a PUBLIC_IPS=()
for IID in $INSTANCE_IDS; do
    IP=$(aliyun ecs DescribeInstances \
        --RegionId "$ALIYUN_REGION" \
        --InstanceIds "[\"$IID\"]" \
        --output cols=Instances.Instance[0].PublicIpAddress.IpAddress[0] rows 2>/dev/null | tail -1)
    if [[ -z "$IP" || "$IP" == "null" ]]; then
        echo "[provision] ERROR: could not get public IP for $IID" >&2
        exit 1
    fi
    PUBLIC_IPS+=("$IP")
    echo "[provision] $IID → $IP"
done

# ── Wait for SSH to become available ──────────────────────────────────────────
for IP in "${PUBLIC_IPS[@]}"; do
    echo "[provision] waiting for SSH on $IP..."
    for attempt in $(seq 1 18); do
        if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 \
               -o BatchMode=yes "root@$IP" "exit" 2>/dev/null; then
            echo "[provision] SSH ready on $IP"
            break
        fi
        if [[ "$attempt" -eq 18 ]]; then
            echo "[provision] ERROR: SSH not available on $IP after 3 minutes" >&2
            exit 1
        fi
        sleep 10
    done
done

# ── Generate per-node inventory and run Ansible ────────────────────────────────
INVENTORY_FILE="$(mktemp /tmp/ek-provision-XXXX.yaml)"
trap "rm -f $INVENTORY_FILE" EXIT

{
    echo "all:"
    echo "  vars:"
    echo "    ek_controller_addr: ${EK_CONTROLLER_ADDR:?EK_CONTROLLER_ADDR not set}"
    echo "    ek_weight_server_addr: ${EK_WEIGHT_SERVER_ADDR:?EK_WEIGHT_SERVER_ADDR not set}"
    echo "    ek_db_dsn: ${EK_DB_DSN:?EK_DB_DSN not set}"
    echo "    ek_model_name: $MODEL"
    echo "    ek_instance_name: $INSTANCE"
    echo "  hosts:"
    for IP in "${PUBLIC_IPS[@]}"; do
        echo "    $IP:"
        echo "      ansible_user: root"
        echo "      ansible_ssh_common_args: '-o StrictHostKeyChecking=no'"
    done
} > "$INVENTORY_FILE"

echo "[provision] running ansible playbook: $ANSIBLE_PLAYBOOK"
ansible-playbook -i "$INVENTORY_FILE" "$ANSIBLE_PLAYBOOK"

# ── Register new nodes in EK scheduler ────────────────────────────────────────
# Build a static inventory snippet and call ek-cli schedule static to register
# the new workers. The workers will connect back via heartbeat automatically,
# but pre-registering lets the scheduler assign experts immediately.
NODE_INVENTORY="$(mktemp /tmp/ek-nodes-XXXX.yaml)"
trap "rm -f $INVENTORY_FILE $NODE_INVENTORY" EXIT

{
    echo "nodes:"
    for IP in "${PUBLIC_IPS[@]}"; do
        echo "  - addr: $IP:51234"
        echo "    channel: grpc"
    done
} > "$NODE_INVENTORY"

echo "[provision] registering nodes with ek-cli"
"$EK_CLI_BIN" schedule static --inventory "$NODE_INVENTORY"

echo "[provision] done: $COUNT node(s) provisioned and registered"
