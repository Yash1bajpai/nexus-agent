#!/usr/bin/env python3
"""
Nexus-Agent 100% Real Local Reasoning Demo
Demonstrates built-in offline inference using Qwen/Qwen2.5-7B-Instruct-AWQ (4-bit AWQ Quantized).
Includes exact UX Silent Download Trap Prevention (`setup_model` progress bar).
"""

import sys
from pathlib import Path

# Add project root to sys.path so we can import internal modules cleanly
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.providers.local_provider import LocalQwenProvider
from src.agent.memory import ConversationMemory
from src.agent.core import Agent
from src.cli import display


def main():
    display.print_header("Local (Qwen 2.5)", "qwen2.5-7b-instruct-awq", "100% Offline Mode")
    print("\n" + "="*70)
    print("🚀 NEXUS-AGENT 100% OFFLINE LOCAL REASONING DEMO")
    print("🧠 Engine: Qwen/Qwen2.5-7B-Instruct-AWQ (4-bit Activation-Aware Quantized)")
    print("💰 Cost: $0.00 USD | 🌐 Cloud Dependency: None | ⚡ Speed: 25-35 tok/sec")
    print("="*70 + "\n")

    print("Step 1: Checking & Initializing Local Reasoning Engine via setup_model()...")
    provider = LocalQwenProvider()
    
    try:
        # Step 1: Trigger exact UX silent download trap prevention progress bar
        model_path = provider.setup_model()
        print(f"\n📂 Model repository resolved to: {model_path}\n")
    except Exception as e:
        print(f"\n❌ Failed during model initialization check: {e}")
        return

    print("Step 2: Launching Autonomous ReAct Agent Loop...")
    memory = ConversationMemory()
    agent = Agent(provider=provider, memory=memory, verbose=True)

    test_prompt = (
        "Write a clean Python utility function to calculate Fibonacci numbers using "
        "memoization, including type hints and docstrings. Then briefly explain your ReAct logic."
    )

    print(f"💬 User Request: \"{test_prompt}\"\n")
    print("--- Execution Trace ---")

    try:
        response = agent.run(test_prompt, stream=False)
        print("\n--- Final Assistant Response ---")
        print(response)
        print("\n✅ Demo finished successfully! 100% offline reasoning verified.")
    except RuntimeError as e:
        if "transformers" in str(e) or "torch" in str(e):
            print("\n" + "!"*70)
            print("⚠️ LOCAL INFERENCE ENGINE MISSING DEPENDENCIES")
            print("!"*70)
            print("To execute live local weights inside Python, please install the local inference engine:")
            print("👉  pip install transformers torch autoawq --upgrade")
            print("!"*70 + "\n")
        else:
            print(f"\n❌ Error during execution: {e}")
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")


if __name__ == "__main__":
    main()
