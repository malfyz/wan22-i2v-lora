#Pod Start Command
===========================
/bin/bash -lc '
set -euo pipefail
URL="https://raw.githubusercontent.com/malfyz/wan22-i2v-lora/refs/heads/main/wan22_bootstrap.sh"
mkdir -p /workspace/logs
echo "[ENTRY] fetching bootstrap..."
curl -fsSL "$URL" -o /tmp/wan22_bootstrap.sh
chmod +x /tmp/wan22_bootstrap.sh

# run once only, then mark as done
if [ ! -f /workspace/.bootstrapped ]; then
  echo "[ENTRY] running bootstrap once..."
  /bin/bash /tmp/wan22_bootstrap.sh 2>&1 | tee /workspace/logs/bootstrap.log || true
  touch /workspace/.bootstrapped
  echo "[ENTRY] bootstrap complete"
else
  echo "[ENTRY] bootstrap already done — skipping"
fi

# keep the container alive, but DO NOT re-run bootstrap
sleep infinity
'


#Training Commands
===========================
bash /workspace/scripts/train_i2v_high.sh

bash /workspace/scripts/train_i2v_low.sh
