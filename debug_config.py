#!/usr/bin/env python3
"""
Debug script to check configuration loading
"""

import sys
import os

# Add the app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from config.liveness_config import get_active_config, CHALLENGE_TIMEOUT

def main():
    print("=" * 50)
    print("LIVENESS CONFIGURATION DEBUG")
    print("=" * 50)
    
    # Check global constant
    print(f"Global CHALLENGE_TIMEOUT: {CHALLENGE_TIMEOUT}")
    
    # Get active config
    config = get_active_config()
    print(f"Active config type: {type(config)}")
    
    # Check if it's a globals dict
    if isinstance(config, dict):
        print(f"Config is globals() dict with {len(config)} keys")
        challenge_timeout = config.get('CHALLENGE_TIMEOUT', 'NOT_FOUND')
        print(f"CHALLENGE_TIMEOUT in config: {challenge_timeout}")
    else:
        # It's a class instance
        print(f"Config is class instance: {config.__class__.__name__}")
        challenge_timeout = getattr(config, 'CHALLENGE_TIMEOUT', 'NOT_FOUND')
        print(f"Config.CHALLENGE_TIMEOUT: {challenge_timeout}")
    
    # Test getattr pattern used by liveness detector
    timeout_via_getattr = getattr(config, 'CHALLENGE_TIMEOUT', 3.0)
    print(f"getattr(config, 'CHALLENGE_TIMEOUT', 3.0): {timeout_via_getattr}")
    
    print("=" * 50)

if __name__ == "__main__":
    main()
