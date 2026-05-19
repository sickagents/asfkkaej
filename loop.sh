#!/bin/bash
# Auto-relog loop for 1-hour session limit.
# Just run: bash loop.sh
# It will keep running pipeline until all steps complete.

cd "$(dirname "$0")"

while true; do
    echo ""
    echo "=========================================="
    echo "  Starting session at $(date)"
    echo "=========================================="
    
    python run.py
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "Pipeline complete! Check results/ for submission."
        break
    fi
    
    echo ""
    echo "Session ended. Relogging in 60 seconds..."
    sleep 60
done
