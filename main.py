# main.py
from pynput import keyboard
from pynput.keyboard import Controller, Key
import time
import math
import os
from Levenshtein import distance  # Ensure 'python-Levenshtein' is installed!

# --- Configuration ---
KEYBOARD_LAYOUT_COORDINATES = {
    'q': (0, 2), 'w': (1, 2), 'e': (2, 2), 'r': (3, 2), 't': (4, 2), 'y': (5, 2), 'u': (6, 2), 'i': (7, 2), 'o': (8, 2), 'p': (9, 2),
    'a': (0.5, 1), 's': (1.5, 1), 'd': (2.5, 1), 'f': (3.5, 1), 'g': (4.5, 1), 'h': (5.5, 1), 'j': (6.5, 1), 'k': (7.5, 1), 'l': (8.5, 1),
    'z': (1.5, 0), 'x': (2.5, 0), 'c': (3.5, 0), 'v': (4.5, 0), 'b': (5.5, 0), 'n': (6.5, 0), 'm': (7.5, 0),
    ' ': (5, -1)
}

# --- Adjustable Constants for Tuning ---
# COSINE_THRESHOLD_FOR_PIVOT: Higher = more forgiving angle detection (more pivots). Lower = stricter.
# Based on your tests, 0.48 was best. Try 0.5 or 0.53 next if pivots are still missed.
COSINE_THRESHOLD_FOR_PIVOT = 0.5  # Keeping this at 0.5 as per your last code

WORD_END_PAUSE_SECONDS = 0.5
MAX_KEY_BUFFER_SIZE = 20

# Levenshtein distance thresholds for forgiving matching
# Allow up to 2 edits (insertions, deletions, substitutions)
MAX_LEVENSHTEIN_DISTANCE = 2
# Max allowed length difference (e.g., 'helo' (4) vs 'hello' (5) is diff 1)
MAX_LENGTH_DIFF = 3

# Minimum movement magnitude to register a segment in pivot detection.
# Helps filter out tiny jitters as pivots. Adjust as needed.
# Example: a movement less than 0.1 units is ignored.
MIN_SEGMENT_MAGNITUDE = 0.1


# --- Global State Variables ---
g_pressed_keys_buffer = []
g_last_key_press_time = time.time()
g_keyboard_controller = Controller()
g_keyboard_listener = None
g_is_injecting_keys = False

# Global dictionary for word frequencies
WORD_FREQUENCIES = {}
# Path to word frequency file (you need to create/download this: word_frequencies.txt)
FREQUENCY_FILEPATH = "word_frequencies.txt"


# --- Trie Data Structure for Efficient Prefix Search ---
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def starts_with(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []  # No words start with this prefix

            node = node.children[char]

        results = []
        self._collect_words(node, prefix, results)
        return results

    def _collect_words(self, node, current_prefix, results):
        if node.is_end_of_word:
            results.append(current_prefix)
        for char, child_node in node.children.items():
            self._collect_words(child_node, current_prefix + char, results)
        return results


# --- Dictionary Loading (Updated to use Trie) ---
def load_dictionary(filepath="words.txt"):
    words = set()
    trie = Trie()

    # List of words to explicitly exclude (add more if you find them undesirable)
    EXCLUDE_WORDS = {'tis', 'heloe', 'tst', 'psha',
                     # Add common short words here if they mis-predict
                     'glen', 'mal', 'gal', 'te', 'aa', 'sanyo'}

    # Load word frequencies
    try:
        with open(FREQUENCY_FILEPATH, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2 and parts[0].isalpha():
                    WORD_FREQUENCIES[parts[0].lower()] = int(parts[1])
    except FileNotFoundError:
        print(
            f"Warning: Word frequency file '{FREQUENCY_FILEPATH}' not found. Prediction will not use frequency ranking.")
    except Exception as e:
        print(f"Error loading frequencies: {e}")

    try:
        # Specify encoding for broader compatibility
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip().lower()
                # Exclude words that are explicitly undesirable
                if word in EXCLUDE_WORDS:
                    continue

                if word.isalpha() and all(c in KEYBOARD_LAYOUT_COORDINATES for c in word):
                    words.add(word)
                    trie.insert(word)
    except FileNotFoundError:
        print(f"Error: Dictionary file '{filepath}' not found.")
        print("Please download a word list (e.g., from https://raw.githubusercontent.com/first20hours/google-10000-english/blob/master/20k.txt)")
        print("Using an empty dictionary.")
        return [], Trie()

    return sorted(list(words)), trie


# Load dictionary and build Trie at startup
DICTIONARY, DICTIONARY_TRIE = load_dictionary()
print(f"Loaded {len(DICTIONARY)} words and built Trie for prediction.")


# --- Helper Functions (for vector math) ---
def get_vector(p1, p2):
    return (p2[0] - p1[0], p2[1] - p1[1])


def dot_product(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1]


def magnitude(v):
    return math.sqrt(v[0]**2 + v[1]**2)


# --- Keyboard Listener Callbacks ---

def on_press(key):
    global g_pressed_keys_buffer, g_last_key_press_time, g_is_injecting_keys

    # Skip processing if we are currently injecting keys
    if g_is_injecting_keys:
        return True

    current_time = time.time()

    # *** REVISED SPACEBAR HANDLING ***
    if key == keyboard.Key.space:
        # If space is pressed and there's a buffer to process, trigger prediction.
        if g_pressed_keys_buffer:
            print(
                f"\n--- Space pressed. Processing buffer (length: {len(g_pressed_keys_buffer)}). ---")
            # Pass the buffer for prediction. triggered_by_space indicates it was triggered this way.
            buffer_for_prediction = list(g_pressed_keys_buffer)  # Make a copy

            # Reset buffer immediately AFTER getting content for prediction
            g_pressed_keys_buffer = []

            process_current_buffer(buffer_for_prediction,
                                   triggered_by_space=True)
            g_last_key_press_time = current_time
            return True  # Allow the space key to pass through and be typed by the user

        # If space is pressed with an empty buffer, just let it pass through.
        return True

    # Regular pause detection for non-space key presses
    if current_time - g_last_key_press_time > WORD_END_PAUSE_SECONDS:
        if g_pressed_keys_buffer:  # Only process if there's actual buffer content
            print(
                f"\n--- Pause detected ({current_time - g_last_key_press_time:.2f}s). Processing previous buffer. ---")
            buffer_for_prediction = list(g_pressed_keys_buffer)  # Make a copy
            g_pressed_keys_buffer = []  # Reset buffer
            process_current_buffer(buffer_for_prediction,
                                   triggered_by_space=False)
        else:
            print(
                f"\n--- Pause detected ({current_time - g_last_key_press_time:.2f}s). Received an empty buffer for processing. ---")

    g_last_key_press_time = current_time

    char = None
    try:
        char = key.char
    except AttributeError:
        # Ignore other special keys (e.g., Ctrl, Alt, Shift, Esc, Enter)
        # unless they are explicitly handled like Space (or if you add Enter).
        return True

    if char and char in KEYBOARD_LAYOUT_COORDINATES:
        key_pos = KEYBOARD_LAYOUT_COORDINATES[char]
        g_pressed_keys_buffer.append((char, key_pos, current_time))

        if len(g_pressed_keys_buffer) > MAX_KEY_BUFFER_SIZE:
            g_pressed_keys_buffer.pop(0)

        print(
            f"Key pressed: '{char}' at {key_pos}. Buffer size: {len(g_pressed_keys_buffer)}")

    return True


def on_release(key):
    return True


# --- Core Processing Function ---
def process_current_buffer(buffer, triggered_by_space):
    if not buffer:
        print("Received an empty buffer for processing in process_current_buffer.")
        return

    # Step 1: Detect pivot points from the raw key buffer
    pivot_chars = []
    pivot_coords = []

    if buffer:
        pivot_chars.append(buffer[0][0])
        pivot_coords.append(buffer[0][1])
    else:
        print("Buffer became empty during pivot detection setup. Skipping.")
        return

    if len(buffer) >= 3:
        for i in range(1, len(buffer) - 1):
            A_char, A_pos, A_time = buffer[i-1]
            B_char, B_pos, B_time = buffer[i]
            C_char, C_pos, C_time = buffer[i+1]

            v1 = get_vector(A_pos, B_pos)
            v2 = get_vector(B_pos, C_pos)

            mag_v1 = magnitude(v1)
            mag_v2 = magnitude(v2)

            # Filter out very small movements to reduce noise in pivot detection
            if mag_v1 < MIN_SEGMENT_MAGNITUDE or mag_v2 < MIN_SEGMENT_MAGNITUDE:
                continue

            # Avoid division by zero if vectors are zero length
            if mag_v1 == 0 or mag_v2 == 0:
                continue

            cos_theta = dot_product(v1, v2) / (mag_v1 * mag_v2)

            if cos_theta < COSINE_THRESHOLD_FOR_PIVOT:
                if pivot_chars[-1] != B_char:
                    pivot_chars.append(B_char)
                    pivot_coords.append(B_pos)

    # The very last key pressed is also considered a pivot (end of the word)
    if buffer and pivot_chars and pivot_chars[-1] != buffer[-1][0]:
        pivot_chars.append(buffer[-1][0])
        pivot_coords.append(buffer[-1][1])

    detected_sequence = "".join(pivot_chars)
    print(
        f"Detected pivot sequence: '{detected_sequence}' (from {len(buffer)} raw keys)")

    # Step 2: Predict word based on detected pivot sequence
    predicted_word = None
    min_distance_overall = float('inf')

    if not detected_sequence:
        print(f"No detected sequence to predict from.")
    elif detected_sequence in DICTIONARY:  # Priority 1: Exact match of the detected sequence
        predicted_word = detected_sequence
        min_distance_overall = 0
    else:  # Priority 2: Find best Levenshtein match from the dictionary
        candidate_matches_with_dist = []  # Store (distance, word) tuples
        for word in DICTIONARY:
            # Optimization 1: First character must match detected_sequence's first character
            if word[0] != detected_sequence[0]:
                continue

            # Optimization 2: Length difference must be within acceptable range
            if abs(len(detected_sequence) - len(word)) > MAX_LENGTH_DIFF:
                continue

            dist = distance(detected_sequence, word)

            if dist <= MAX_LEVENSHTEIN_DISTANCE:  # Only consider matches within allowed distance
                candidate_matches_with_dist.append((dist, word))

        # Sort candidates: First by Levenshtein distance (ascending),
        # then by word frequency (descending, more common first),
        # then by length (descending, for longer words first)
        candidate_matches_with_dist.sort(key=lambda x: (
            x[0], -WORD_FREQUENCIES.get(x[1], 0), -len(x[1])))

        if candidate_matches_with_dist:
            predicted_word = candidate_matches_with_dist[0][1]
            min_distance_overall = candidate_matches_with_dist[0][0]
        else:
            print(
                f"No strong candidates found in dictionary for '{detected_sequence}'.")
            predicted_word = None

    # Step 3: Type the predicted word into the active application
    if predicted_word:
        print(
            f"Predicted word: '{predicted_word}' (Levenshtein Dist: {min_distance_overall}). Attempting to type...")

        # *** NEW DELETION STRATEGY: Ctrl + Backspace ***
        print("Simulating Ctrl+Backspace to delete the last word/sequence.")

        # Set injection flag to True
        global g_is_injecting_keys
        g_is_injecting_keys = True
        # Give OS a moment to process original key presses before deletion
        time.sleep(0.05)

        try:
            # Press Ctrl+Backspace to delete the last typed "word"
            # Use _l for left Ctrl, often more reliable
            g_keyboard_controller.press(Key.ctrl_l)
            g_keyboard_controller.press(Key.backspace)
            time.sleep(0.05)  # Small delay for the combo to register
            g_keyboard_controller.release(Key.backspace)
            g_keyboard_controller.release(Key.ctrl_l)
            time.sleep(0.05)  # Give OS time to process the deletion

            # Type the predicted word
            g_keyboard_controller.type(predicted_word)
            # Add a slight delay here before space for better visual flow
            time.sleep(0.01)

            # Only auto-insert space if the user did NOT trigger with a space
            if not triggered_by_space:
                g_keyboard_controller.press(Key.space)
                g_keyboard_controller.release(Key.space)
                print(f"Successfully typed '{predicted_word}' and a space.")
            else:
                print(
                    f"Successfully typed '{predicted_word}'. Space already provided by user.")

        finally:
            # Set injection flag back to False
            g_is_injecting_keys = False
            # Reset g_last_key_press_time AFTER injection completes AND flag is off.
            global g_last_key_press_time
            g_last_key_press_time = time.time()
            print("Injection complete. Listener reactivated for user input.")

    else:
        print(
            f"No confident dictionary prediction found for sequence: '{detected_sequence}'. Leaving raw input.")


# --- Main Listener Setup ---
if __name__ == '__main__':
    g_keyboard_listener = keyboard.Listener(
        on_press=on_press, on_release=on_release)
    g_keyboard_listener.start()
    print("Listener started in background. Press Ctrl+C in this terminal to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping listener...")
        if g_keyboard_listener:
            g_keyboard_listener.stop()
            g_keyboard_listener.join()
        print("Listener stopped gracefully.")
