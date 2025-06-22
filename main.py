# main.py
from pynput import keyboard
from pynput.keyboard import Controller, Key  # New: for typing back
import time
import math  # New: for vector math
import os  # New: for dictionary file path

# --- Configuration (unchanged) ---
KEYBOARD_LAYOUT_COORDINATES = {
    'q': (0, 2), 'w': (1, 2), 'e': (2, 2), 'r': (3, 2), 't': (4, 2), 'y': (5, 2), 'u': (6, 2), 'i': (7, 2), 'o': (8, 2), 'p': (9, 2),
    'a': (0.5, 1), 's': (1.5, 1), 'd': (2.5, 1), 'f': (3.5, 1), 'g': (4.5, 1), 'h': (5.5, 1), 'j': (6.5, 1), 'k': (7.5, 1), 'l': (8.5, 1),
    'z': (1.5, 0), 'x': (2.5, 0), 'c': (3.5, 0), 'v': (4.5, 0), 'b': (5.5, 0), 'n': (6.5, 0), 'm': (7.5, 0),
    ' ': (5, -1)
}

COSINE_THRESHOLD_FOR_PIVOT = -0.5  # A placeholder value, you'll tune this!
WORD_END_PAUSE_SECONDS = 0.5
MAX_KEY_BUFFER_SIZE = 20

# --- Global State (unchanged) ---
g_pressed_keys_buffer = []
g_last_key_press_time = time.time()
g_keyboard_controller = Controller()  # Initialize the keyboard controller

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
    # Sort for consistency (though not strictly necessary)
    return sorted(list(words))


DICTIONARY = load_dictionary()  # Load dictionary once at startup
print(f"Loaded {len(DICTIONARY)} words for prediction.")

# --- Vector Helper Functions (add these) ---


def get_vector(p1, p2):
    return (p2[0] - p1[0], p2[1] - p1[1])


def dot_product(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1]


def magnitude(v):
    return math.sqrt(v[0]**2 + v[1]**2)

# --- Key Listener Callback (on_press, unchanged) ---


def on_press(key):
    global g_pressed_keys_buffer, g_last_key_press_time
    # ... (same as before) ...
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
            return  # Ignore other special keys

    if char and char in KEYBOARD_LAYOUT_COORDINATES:
        key_pos = KEYBOARD_LAYOUT_COORDINATES[char]
        g_pressed_keys_buffer.append((char, key_pos, current_time))
        if len(g_pressed_keys_buffer) > MAX_KEY_BUFFER_SIZE:
            g_pressed_keys_buffer.pop(0)
        print(
            f"Key: '{char}' ({key_pos}). Buffer size: {len(g_pressed_keys_buffer)}")


def on_release(key):
    return  # No changes here

# --- Main Processing Logic (updated) ---


def process_current_buffer(buffer):
    if not buffer:
        print("Buffer was empty.")
        return

    # Step 1: Detect pivot points
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
                if pivot_chars[-1] != B_char:  # Prevent duplicate consecutive chars
                    pivot_chars.append(B_char)

    # The very last key pressed is also considered a pivot (end of the word)
    if len(buffer) > 0 and pivot_chars[-1] != buffer[-1][0]:
        pivot_chars.append(buffer[-1][0])

    detected_sequence = "".join(pivot_chars)
    print(
        f"Detected pivot sequence: '{detected_sequence}' (from {len(buffer)} raw keys)")

    # Step 2: Predict word based on sequence
    predicted_word = None
    if detected_sequence in DICTIONARY:
        predicted_word = detected_sequence
    else:
        # Simple startswith match (can be improved later)
        candidates = [
            word for word in DICTIONARY if word.startswith(detected_sequence)]
        # Prioritize longer, more specific matches
        candidates.sort(key=len, reverse=True)
        if candidates:
            predicted_word = candidates[0]

    # Step 3: Type the predicted word
    if predicted_word:
        print(f"Predicted word: '{predicted_word}'. Typing...")

        # Simulate backspaces to delete the "raw" keys the user pressed
        # This is the tricky part!
        # Delete all keys in the current buffer segment
        num_keys_to_delete = len(buffer)
        print(f"Attempting to backspace {num_keys_to_delete} characters.")

        # Temporarily block other key events while we type (important!)
        # pynput Listener has an `on_press` return value, but for injecting keys,
        # you need to be careful not to trigger *your own* listener.
        # `pynput` generally handles this internally for `Controller.type()`.

        # Simulate backspaces
        for _ in range(num_keys_to_delete):
            g_keyboard_controller.press(Key.backspace)
            g_keyboard_controller.release(Key.backspace)
            time.sleep(0.01)  # Small delay between key presses for reliability

        # Type the predicted word
        g_keyboard_controller.type(predicted_word)
        g_keyboard_controller.press(Key.space)  # Auto-add a space
        g_keyboard_controller.release(Key.space)
        print(f"Successfully typed '{predicted_word}' and space.")
    else:
        print(f"No prediction found for sequence: '{detected_sequence}'.")
        # You might want to re-type the original keys here if no match,
        # or just leave them for the user to manually correct.
        # For now, we do nothing, meaning the user's raw input remains.


# --- Main Listener Setup (unchanged) ---
with keyboard.Listener(
        on_press=on_press,
        on_release=on_release) as listener:
    print("Starting directional keyboard listener...")
    print("Type by 'gliding' your fingers across the keyboard.")
    print(
        f"A pause of more than {WORD_END_PAUSE_SECONDS} seconds will trigger word processing.")
    print("Press Ctrl+C in this terminal to stop.")
    print("Remember to grant Accessibility/Input Monitoring permissions if prompted by your OS.")
    listener.join()
