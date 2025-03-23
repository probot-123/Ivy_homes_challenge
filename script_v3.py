"""
Improved Autocomplete Scraper for v3 Endpoint

Key Features and Insights for v3:
- API returns at most 15 words per request (MAX_RESULTS set to 15).
- Maximum word length is 10 characters.
- Words are lowercase; they may include digits (even at the beginning) 
  and can contain plus(+), minus(-), space(' '), and period('.') in positions 
  other than the start.
- No "Retry-After" header is provided; exponential backoff is used for 429 responses.
- Average response time is 40-60 ms, so a 10-second timeout is sufficient.
- Server limits to 80 requests per minute are enforced using a token bucket.
- Progress is saved every 100 API requests in files with a v3 suffix.
- Good new words/request sent ratio is maintained via dynamic prefix expansion.
- Graceful shutdown is handled using threading.Event for Ctrl+C.
"""

import requests
import time
import logging
import threading
from queue import Queue, Empty
from threading import Thread, RLock  # RLock allows nested locking

# Configure logging to output timestamps, log level, and message into a log file for v3.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='autocomplete_extraction_v3.log'
)

# Configuration constants based on our discovered API constraints for v3.
API_VERSION = "v3"  # Updated API version.
API_BASE = f"http://35.200.185.69:8000/{API_VERSION}/autocomplete"  # v3 endpoint URL.
MAX_WORKERS = 3         # Use 3 worker threads for concurrent processing.
MAX_RETRIES = 5         # Maximum number of retries per request (for handling 429 responses).
MAX_PREFIX_LENGTH = 10  # Maximum length of a word/prefix.
MAX_RESULTS = 15        # v3 returns at most 15 words per request.
RETRY_DELAY = 1         # Base delay (not directly used as we implement exponential backoff).

# Rate limiting: v3 allows 80 requests per minute.
class TokenBucket:
    """Token bucket to enforce a maximum of 80 requests per minute."""
    def __init__(self, capacity, refill_period):
        self.capacity = capacity         # Maximum tokens available per period.
        self.tokens = capacity           # Current available tokens.
        self.refill_period = refill_period  # Refill period in seconds.
        self.last_refill = time.monotonic()
        self.lock = RLock()              # Reentrant lock for thread safety.

    def consume(self):
        with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            # Refill tokens if the refill period has elapsed.
            if elapsed > self.refill_period:
                self.tokens = self.capacity
                self.last_refill = now
            if self.tokens > 0:
                self.tokens -= 1
                return True
            # No tokens available: compute precise wait time and sleep.
            sleep_time = self.refill_period - elapsed + 0.1  # Add a small buffer.
            logging.info(f"Token bucket empty. Sleeping for {sleep_time:.2f} seconds")
        time.sleep(sleep_time)
        return self.consume()

# Initialize the token bucket for 80 requests per 60 seconds.
bucket = TokenBucket(capacity=80, refill_period=60)

# Shared resources for managing state across threads.
all_names = set()           # Set to store all unique names extracted.
explored_prefixes = set()   # Set to track which prefixes have been queried.
prefix_queue = Queue()      # Queue for dynamic prefix exploration.
data_lock = RLock()         # Reentrant lock to protect shared data.
request_counter = 0         # Counter for total API requests made.
last_saved_request = 0      # Last request count when progress was saved.
stop_event = threading.Event()  # Event to signal threads to stop (e.g., on Ctrl+C).

def save_progress():
    """
    Save current progress (unique names and stats) to disk.
    Files are saved with a v3 suffix (e.g., names_v3.txt, stats_v3.txt).
    This function is triggered every 100 API requests.
    """
    suffix = f"_{API_VERSION}"
    try:
        # Save unique names to file.
        with open(f'unique_names{suffix}.txt', 'w') as f:
            f.write('\n'.join(sorted(all_names)))
        # Save progress statistics to file.
        with open(f'stats{suffix}.txt', 'w') as f:
            f.write(f"Total unique names: {len(all_names)}\n")
            f.write(f"Total requests sent: {request_counter}\n")
        logging.info("Progress saved to names and stats files.")
    except Exception as e:
        logging.error(f"Save error: {str(e)}")

def query_api(prefix):
    """
    Query the v3 autocomplete API with a given prefix.
    Uses exponential backoff for 429 responses, as the API does not provide a Retry-After header.
    """
    global request_counter, last_saved_request
    retries = 0
    while retries <= MAX_RETRIES and not stop_event.is_set():
        try:
            # Enforce rate limiting by consuming a token.
            bucket.consume()
            with data_lock:
                request_counter += 1
                # Save progress every 100 requests.
                if request_counter - last_saved_request >= 100:
                    last_saved_request = request_counter
                    save_progress()
            # Make the GET request with the given prefix.
            response = requests.get(API_BASE, params={'query': prefix}, timeout=10)
            
            if response.status_code == 429:
                # If rate limited, use exponential backoff (no Retry-After header available).
                delay = (2 ** retries) + 0.1
                logging.warning(f"429 on {prefix}. Retry {retries} in {delay:.2f} seconds")
                time.sleep(delay)
                retries += 1
                continue
                
            response.raise_for_status()
            # Return the suggestions; v3 returns at most 15 words per request.
            return response.json().get('results', [])
            
        except Exception as e:
            logging.error(f"Error on {prefix}: {str(e)}")
            if retries >= MAX_RETRIES:
                return []
            retries += 1
            time.sleep(0.5)  # Small delay before retrying.
    return []

def worker():
    """
    Worker thread function:
    - Retrieves a prefix from the queue.
    - Queries the API and collects new names.
    - If the API returns the maximum number of suggestions (15) and the prefix is not too long,
      new prefixes are generated by appending allowed characters.
      
    Allowed characters for extending prefixes in v3:
    - For non-initial positions, allowed characters are lowercase letters, digits, plus(+), minus(-), space(' '), and period('.').
    - Words can begin with a digit or letter but cannot begin with plus, minus, space, or period.
    """
    # Allowed characters for extending prefixes.
    allowed_chars = "abcdefghijklmnopqrstuvwxyz0123456789+- ."
    while not stop_event.is_set():
        try:
            prefix = prefix_queue.get(timeout=1)
        except Empty:
            continue

        with data_lock:
            # Skip if prefix is already processed or if its length exceeds maximum allowed.
            if prefix in explored_prefixes or len(prefix) > MAX_PREFIX_LENGTH:
                prefix_queue.task_done()
                continue
            explored_prefixes.add(prefix)

        suggestions = query_api(prefix)
        new_names = []
        with data_lock:
            # Add new names to our global set, filtering duplicates.
            for name in suggestions:
                if name not in all_names:
                    all_names.add(name)
                    new_names.append(name)
        
        # If the API returns the maximum suggestions and prefix length is below limit,
        # generate new prefixes by appending allowed characters.
        if len(suggestions) >= MAX_RESULTS and len(prefix) < MAX_PREFIX_LENGTH:
            new_prefixes = [prefix + char for char in allowed_chars]
            with data_lock:
                for np in new_prefixes:
                    if np not in explored_prefixes:
                        prefix_queue.put(np)
        
        prefix_queue.task_done()
        logging.info(f"Processed {prefix} | New names: {len(new_names)} | Queue size: {prefix_queue.qsize()}")

def main():
    """
    Main function:
    - Seeds the prefix queue with initial prefixes (allowed: letters and digits only, as words can't start with punctuation).
    - Starts worker threads for concurrent processing.
    - Monitors the prefix queue and handles graceful shutdown on KeyboardInterrupt.
    - Saves final progress and prints summary statistics upon completion.
    """
    start_time = time.time()
    # Initialize the queue with all single lowercase letters and digits.
    initial_prefixes = "abcdefghijklmnopqrstuvwxyz0123456789"
    for c in initial_prefixes:
        prefix_queue.put(c)
    
    # Start worker threads.
    threads = []
    for _ in range(MAX_WORKERS):
        t = Thread(target=worker)
        t.start()
        threads.append(t)
    
    try:
        # Monitor the prefix queue until it's empty.
        while True:
            time.sleep(1)
            if prefix_queue.empty():
                time.sleep(2)  # Additional wait to ensure no new items are added.
                if prefix_queue.empty():
                    logging.info("Queue empty. No more prefixes to process.")
                    break
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt detected. Shutting down...")
        stop_event.set()
        # Drain the queue to allow workers to finish gracefully.
        while not prefix_queue.empty():
            try:
                prefix_queue.get_nowait()
                prefix_queue.task_done()
            except Exception:
                break
    finally:
        # Wait for all worker threads to finish.
        for t in threads:
            t.join(timeout=5)
        stop_event.set()
        save_progress()  # Final progress save.
        elapsed_time = time.time() - start_time
        # Log and print final statistics.
        logging.info("Final progress saved.")
        logging.info(f"Total names extracted: {len(all_names)}")
        logging.info(f"Total API requests made: {request_counter}")
        logging.info(f"Total execution time: {elapsed_time:.2f} seconds")
        print(f"Total new names extracted: {len(all_names)}")
        print(f"Total API requests sent: {request_counter}")
        print(f"Total execution time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()
