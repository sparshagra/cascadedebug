#!/bin/bash
set -e

echo "=============================================="
echo "🚀 CascadeDebug GRPO Training — HF Spaces T4"
echo "=============================================="

# Pull latest code
cd /app
git pull origin main 2>/dev/null || true

# Start a simple status server in background (HF needs port 7860)
python3 -c "
import http.server
import threading
import json
import os
import time

class StatusHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        status_file = '/app/results/training_status.json'
        if os.path.exists(status_file):
            with open(status_file) as f:
                data = f.read()
        else:
            data = json.dumps({'status': 'starting', 'message': 'Training is initializing...'})
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(data.encode())
    def log_message(self, format, *args):
        pass  # suppress logs

server = http.server.HTTPServer(('0.0.0.0', 7860), StatusHandler)
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
print('📡 Status server running on :7860')
" &

# Wait for status server
sleep 2

# Create results dir
mkdir -p /app/results

# Write initial status
python3 -c "
import json
with open('/app/results/training_status.json', 'w') as f:
    json.dump({'status': 'running', 'message': 'Training starting...'}, f)
"

# Run training
echo ""
echo "🏋️ Starting GRPO training..."
python3 /app/training/train_grpo_colab.py

# Write completion status
python3 -c "
import json
with open('/app/results/training_status.json', 'w') as f:
    json.dump({'status': 'complete', 'message': 'Training finished!'}, f)
"

echo ""
echo "✅ Training complete! Results saved."
echo "Keeping container alive for result inspection..."

# Keep alive so Space doesn't restart
sleep infinity
