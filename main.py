# main.py
from pynput import keyboard

def on_press(key):
    """Callback function when a key is pressed."""
    try:
        # key.char is used for alphanumeric keys (a, b, 1, 2, etc.)
        print(f'Key pressed: {key.char}')
    except AttributeError:
        # Non-alphanumeric keys (Space, Ctrl, Shift, F1, etc.)
        print(f'Special key pressed: {key}')

def on_release(key):
    """Callback function when a key is released."""
    # We don't necessarily need to do anything on release for this project,
    # but it's good to include for a complete listener example.
    print(f'Key released: {key}')
    # If you want to stop the listener after a specific key (e.g., 'esc'), uncomment:
    # if key == keyboard.Key.esc:
    #     return False # Stop listener

# Create a listener instance
# The 'with' statement ensures the listener is properly started and stopped.
with keyboard.Listener(
        on_press=on_press,
        on_release=on_release) as listener:
    print("Starting keyboard listener...")
    print("Press any key. Press Ctrl+C in this terminal to stop.")
    listener.join() # This keeps the script running indefinitely
