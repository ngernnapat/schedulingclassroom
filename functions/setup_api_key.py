#!/usr/bin/env python3
"""
Setup script for configuring the OpenAI API key for the Planner Content Generation API.
This script helps users set up their environment properly.
"""

import os
import sys
from pathlib import Path

def create_env_file():
    """Create a .env file with the API key"""
    env_file = Path(".env")
    
    if env_file.exists():
        print("📄 .env file already exists.")
        response = input("Do you want to update it? (y/n): ").lower().strip()
        if response != 'y':
            print("Keeping existing .env file.")
            return
    
    print("🔑 Please enter your OpenAI API key:")
    print("(You can get one from: https://platform.openai.com/api-keys)")
    api_key = input("API Key: ").strip()
    
    if not api_key:
        print("❌ No API key provided. Exiting.")
        return
    
    # Write to .env file
    with open(env_file, 'w') as f:
        f.write(f"OPENAI_API_KEY={api_key}\n")
    
    print(f"✅ API key saved to {env_file.absolute()}")
    print("🔒 The .env file is automatically ignored by git for security.")

def set_environment_variable():
    """Set the environment variable for current session"""
    print("🔑 Please enter your OpenAI API key:")
    print("(You can get one from: https://platform.openai.com/api-keys)")
    api_key = input("API Key: ").strip()
    
    if not api_key:
        print("❌ No API key provided. Exiting.")
        return
    
    # Set environment variable
    os.environ["OPENAI_API_KEY"] = api_key
    print("✅ API key set for current session.")
    print("⚠️  Note: This will only last for the current terminal session.")
    print("   For permanent setup, use the .env file option instead.")

def check_current_setup():
    """Check current API key setup"""
    print("🔍 Checking current API key setup...")
    
    # Check environment variable
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key:
        masked_key = env_key[:8] + "..." + env_key[-4:] if len(env_key) > 12 else "***"
        print(f"✅ Environment variable OPENAI_API_KEY is set: {masked_key}")
    else:
        print("❌ Environment variable OPENAI_API_KEY is not set")
    
    # Check .env file
    env_file = Path(".env")
    if env_file.exists():
        print("✅ .env file exists")
        try:
            with open(env_file, 'r') as f:
                content = f.read()
                if "OPENAI_API_KEY" in content:
                    print("✅ .env file contains OPENAI_API_KEY")
                else:
                    print("❌ .env file exists but doesn't contain OPENAI_API_KEY")
        except Exception as e:
            print(f"❌ Error reading .env file: {e}")
    else:
        print("❌ .env file does not exist")
    
    print()

def main():
    """Main setup function"""
    print("🚀 OpenAI API Key Setup for Planner Content Generation API")
    print("=" * 60)
    
    # Check current setup
    check_current_setup()
    
    # If API key is already set, ask if user wants to change it
    if os.getenv("OPENAI_API_KEY"):
        print("✅ API key is already configured!")
        response = input("Do you want to update it? (y/n): ").lower().strip()
        if response != 'y':
            print("Keeping current configuration.")
            return
    
    print("Choose setup method:")
    print("1. Create .env file (recommended for permanent setup)")
    print("2. Set environment variable (temporary, current session only)")
    print("3. Exit")
    
    choice = input("Enter your choice (1-3): ").strip()
    
    if choice == "1":
        create_env_file()
    elif choice == "2":
        set_environment_variable()
    elif choice == "3":
        print("Exiting setup.")
        return
    else:
        print("❌ Invalid choice. Exiting.")
        return
    
    print("\n🎉 Setup complete!")
    print("You can now start the API server with:")
    print("  python generate_planner_content_api.py")
    print("  or")
    print("  ./start_planner_api.sh")

if __name__ == "__main__":
    main()
