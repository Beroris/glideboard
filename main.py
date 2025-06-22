# main.py
from pynput import keyboard
from pynput.keyboard import Controller, Key  # NEW: Used for typing back
import time
import math
import os

# --- Configuration ---
KEYBOARD_LAYOUT_COORDINATES = {
    # This is a conceptual grid. (x, y) coordinates.
    # Top row (QWERTY)
    'q': (0, 2), 'w': (1, 2), 'e': (2, 2), 'r': (3, 2), 't': (4, 2), 'y': (5, 2), 'u': (6, 2), 'i': (7, 2), 'o': (8, 2), 'p': (9, 2),
    # Middle row (ASDFG) - shifted right
    'a': (0.5, 1), 's': (1.5, 1), 'd': (2.5, 1), 'f': (3.5, 1), 'g': (4.5, 1), 'h': (5.5, 1), 'j': (6.5, 1), 'k': (7.5, 1), 'l': (8.5, 1),
    # Bottom row (ZXCVB) - shifted right
    'z': (1.5, 0), 'x': (2.5, 0), 'c': (3.5, 0), 'v': (4.5, 0), 'b': (5.5, 0), 'n': (6.5, 0), 'm': (7.5, 0),
    # Spacebar (arbitrary lower coordinate)
    ' ': (5, -1)
}

# COSINE_THRESHOLD_FOR_PIVOT: This determines how sharp a turn must be
# for a key to be considered a "pivot" point.
# Ranges from -1 (180 degrees, perfectly back) to 1 (0 degrees, perfectly straight).
# -0.5 corresponds to a 120-degree turn. You'll fine-tune this later!
COSINE_THRESHOLD_FOR_PIVOT = -0.5

# WORD_END_PAUSE_SECONDS: If the user pauses typing for this long,
# we consider the current word segment complete and process the buffer.
WORD_END_PAUSE_SECONDS = 0.5

# MAX_KEY_BUFFER_SIZE: Limits how many recent key presses we store.
# Prevents the buffer from growing indefinitely if the user types very long "glides".
MAX_KEY_BUFFER_SIZE = 20

# --- Global State Variables ---
g_pressed_keys_buffer = []
g_last_key_press_time = time.time()
# Initialize the keyboard controller for typing back
g_keyboard_controller = Controller()

# --- Dictionary Loading ---


def load_dictionary(filepath="words.txt"):
    words = set()
    try:
        with open(filepath, 'r') as f:
            for line in f:
                word = line.strip().lower()
                # Only add words if all their characters exist in our keyboard layout
                if word.isalpha() and all(c in KEYBOARD_LAYOUT_COORDINATES for c in word):
                    words.add(word)
    except FileNotFoundError:
        print(f"Error: Dictionary file '{filepath}' not found.")
        print("Please download a word list (e.g., from https://raw.githubusercontent.com/dwyl/english-words/master/words.txt)")
        print("Using an empty dictionary.")
    return sorted(list(words))


DICTIONARY = load_dictionary()  # Load dictionary once at startup
print(f"Loaded {len(DICTIONARY)} words for prediction.")


# --- Helper Functions (for vector math) ---
def get_vector(p1, p2):
    """Calculates a vector from point p1 to p2."""
    return (p2[0] - p1[0], p2[1] - p1[1])


def dot_product(v1, v2):
    """Calculates the dot product of two vectors."""
    return v1[0] * v2[0] + v1[1] * v2[1]


def magnitude(v):
    """Calculates the magnitude (length) of a vector."""
    return math.sqrt(v[0]**2 + v[1]**2)


# --- Keyboard Listener Callbacks ---

def on_press(key):
    global g_pressed_keys_buffer, g_last_key_press_time

    current_time = time.time()

    if current_time - g_last_key_press_time > WORD_END_PAUSE_SECONDS:
        print(
            f"\n--- Pause detected ({current_time - g_last_key_press_time:.2f}s). Processing previous buffer. ---")
        process_current_buffer(g_pressed_keys_buffer)
        g_pressed_keys_buffer = []  # Reset buffer

    g_last_key_press_time = current_time

    char = None
    try:
        char = key.char
    except AttributeError:
        if key == keyboard.Key.space:
            char = ' '
        else:
            return True  # Don't process other special keys for glide logic

    if char and char in KEYBOARD_LAYOUT_COORDINATES:
        key_pos = KEYBOARD_LAYOUT_COORDINATES[char]
        g_pressed_keys_buffer.append((char, key_pos, current_time))

        if len(g_pressed_keys_buffer) > MAX_KEY_BUFFER_SIZE:
            g_pressed_keys_buffer.pop(0)

        print(
            f"Key pressed: '{char}' at {key_pos}. Buffer size: {len(g_pressed_keys_buffer)}")

    return True


def on_release(key):
    return True  # No specific action on key release needed


# --- Core Processing Function (Now with Pivot Detection and Prediction!) ---
def process_current_buffer(buffer):
    if not buffer:
        print("Received an empty buffer for processing.")
        return

    # Step 1: Detect pivot points from the raw key buffer
    pivot_chars = []
    # The first key pressed is always considered a pivot (start of word)
    pivot_chars.append(buffer[0][0])

    # Need at least 3 keys (A -> B -> C) to detect a pivot at B
    if len(buffer) >= 3:
        for i in range(1, len(buffer) - 1):  # Iterate through potential middle pivots
            A_char, A_pos, A_time = buffer[i-1]
            B_char, B_pos, B_time = buffer[i]
            C_char, C_pos, C_time = buffer[i+1]

            v1 = get_vector(A_pos, B_pos)
            v2 = get_vector(B_pos, C_pos)

            mag_v1 = magnitude(v1)
            mag_v2 = magnitude(v2)

            # Avoid division by zero if keys are identical (no movement)
            if mag_v1 == 0 or mag_v2 == 0:
                continue

            cos_theta = dot_product(v1, v2) / (mag_v1 * mag_v2)

            # If the angle is sharp enough (cosine is low/negative), B is a pivot
            if cos_theta < COSINE_THRESHOLD_FOR_PIVOT:
                # Add B as a pivot only if it's different from the last added pivot
                if pivot_chars[-1] != B_char:
                    pivot_chars.append(B_char)

    # The very last key pressed is also considered a pivot (end of the word)
    # Add if it's different from the last pivot already captured
    if len(buffer) > 0 and pivot_chars[-1] != buffer[-1][0]:
        pivot_chars.append(buffer[-1][0])

    detected_sequence = "".join(pivot_chars)
    print(
        f"Detected pivot sequence: '{detected_sequence}' (from {len(buffer)} raw keys)")

    # Step 2: Predict word based on detected pivot sequence
    predicted_word = None
    if detected_sequence in DICTIONARY:
        predicted_word = detected_sequence  # Exact match is highest priority
    else:
        # Simple startswith match for suggestions. You can make this more complex later.
        candidates = [
            word for word in DICTIONARY if word.startswith(detected_sequence)]
        # Prioritize longer, more specific matches, assuming more letters -> more precise
        candidates.sort(key=len, reverse=True)
        if candidates:
            predicted_word = candidates[0]  # Pick the longest starting match

    # Step 3: Type the predicted word into the active application
    if predicted_word:
        print(f"Predicted word: '{predicted_word}'. Attempting to type...")

        # Determine how many characters to backspace.
        # This is a critical and potentially tricky part.
        # For simplicity in this example, we backspace the number of raw keys in the buffer.
        # This assumes the user's intent is to replace everything they just typed for this word.
        num_keys_to_delete = len(buffer)
        print(f"Simulating {num_keys_to_delete} backspaces...")

        # Backspace the raw input
        for _ in range(num_keys_to_delete):
            g_keyboard_controller.press(Key.backspace)
            g_keyboard_controller.release(Key.backspace)
            time.sleep(0.01)  # Small delay for reliability in quick operations

        # Type the predicted word
        g_keyboard_controller.type(predicted_word)
        # Add a space after the word, as is common in typing
        g_keyboard_controller.press(Key.space)
        g_keyboard_controller.release(Key.space)
        print(f"Successfully typed '{predicted_word}' and a space.")
    else:
        print(
            f"No dictionary prediction found for sequence: '{detected_sequence}'. Leaving raw input.")
        # If no word is predicted, you might choose to leave the raw input
        # or implement a fallback (e.g., just type the detected pivot sequence).


# --- Main Listener Setup ---
if __name__ == '__main__':
    print("Starting directional keyboard listener...")
    print("Instructions:")
    print("1. Type by 'gliding' your fingers across the keyboard.")
    print("2. Change direction sharply at each intended letter (e.g., for 'test', T -> E -> S -> T).")
    print(
        f"3. A pause of more than {WORD_END_PAUSE_SECONDS} seconds will trigger word processing.")
    print("4. Press Ctrl+C in this terminal to stop the script.")
    print("\nIMPORTANT:")
    print(" - Ensure 'words.txt' (dictionary) is in the same directory.")
    print(" - For WSL2 Linux GUI apps, ensure your WSLg setup is functional (which it now seems to be!).")

    with keyboard.Listener(
            on_press=on_press,
            on_release=on_release) as listener:
        listener.join()
