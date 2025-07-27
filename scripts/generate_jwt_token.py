#!/usr/bin/env python3
import jwt
import argparse
import sys
from datetime import datetime, timedelta

def generate_token(user_id="test_user", username="test", permissions=None, test=False, quiet=False):
    if permissions is None:
        permissions = ["predict", "batch", "admin"] if test else ["predict"]
    
    secret = "test-secret-key-change-in-production"
    payload = {
        "user_id": user_id,
        "username": username,
        "permissions": permissions,
        "exp": datetime.utcnow() + timedelta(hours=24),
        "iat": datetime.utcnow()
    }
    
    token = jwt.encode(payload, secret, algorithm="HS256")
    
    if quiet:
        print(token)
    else:
        print(f"Generated JWT token for {username}:")
        print(f"Permissions: {', '.join(permissions)}")
        print(f"Token: {token}")
    
    return token

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate JWT token for API authentication")
    parser.add_argument("--user-id", default="test_user", help="User ID")
    parser.add_argument("--username", default="test", help="Username")
    parser.add_argument("--permissions", nargs="+", default=["predict"], help="Permissions")
    parser.add_argument("--test", action="store_true", help="Generate test token with all permissions")
    parser.add_argument("--quiet", action="store_true", help="Only output token")
    
    args = parser.parse_args()
    generate_token(args.user_id, args.username, args.permissions, args.test, args.quiet)