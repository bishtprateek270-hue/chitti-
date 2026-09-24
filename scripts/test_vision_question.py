"""
Test full vision question query.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.main import ChittiController

print("Initializing ChittiController...")
controller = ChittiController()
controller.initialize()

print("\n--- Asking: 'who is in front of u' ---")
controller.process_user_input("who is in front of u")
