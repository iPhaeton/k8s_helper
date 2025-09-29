#!/usr/bin/env python3
"""
Complete Grafana Cookie Authentication Example

This demonstrates how to use session cookies for Grafana API authentication.
Perfect for when you're using Google OAuth or other identity providers.
"""

import asyncio
import os
from tools import (
    grafana_api_request_impl,
)
from dotenv import load_dotenv
import json


load_dotenv(override=True)


async def main():
    # Configuration
    grafana_url = os.getenv("GRAFANA_URL", "https://your-grafana-instance.com")

    print(f"🍪 Using session cookie authentication")
    print(f"📍 Grafana URL: {grafana_url}")

    # Test 1: Basic authentication test
    print("\n1️⃣ Testing authentication...")
    result = await grafana_api_request_impl(endpoint="/api/datasources", method="GET")

    print('!!!!!!!!!!!!!!!!!!!!!', json.dumps(result.json, indent=2))

    if result.returncode == 0:
        print(f"✅ Authenticated successfully!")
    else:
        print(f"❌ Authentication failed: {result.stderr}")
        if "401" in result.stderr:
            print("💡 Your session cookie may have expired. Get a fresh one!")
        return

    print("\n🎉 Session cookie authentication test complete!")


if __name__ == "__main__":
    asyncio.run(main())
