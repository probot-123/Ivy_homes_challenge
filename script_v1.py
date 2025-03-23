"""
Improved Autocomplete Scraper with Insights Incorporated
Key Features and Insights:
- API returns at most 10 words per request (MAX_RESULTS set to 10).
- Maximum word length is 10 characters (MAX_PREFIX_LENGTH set to 10).
- Returned words are lowercase and do not include numbers or special symbols.
- API does not include a "Retry-After" header, so our exponential backoff is used for 429 responses.
- Average API response time is 40-60 ms, so our timeout of 10 seconds is ample.
- Server limits to 100 requests per minute are enforced via a token bucket.
- Periodic progress saving is triggered every 100 API requests.
- Proper handling of Ctrl+C using threading.Event for graceful shutdown.
"""

import requests
import time
import logging
import threading
from queue import Queue, Empty
from threading import Thread, RLock  # RLock allows nested locking

# Configure logging to output timestamps, log level, and message into a file.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='autocomplete_extraction_v1.log'
)

# Configuration constants based on our discovered API constraints.
API_VERSION = "v1"
API_BASE = f"http://35.200.185.69:8000/{API_VERSION}/autocomplete"
MAX_WORKERS = 3         # Use 3 worker threads for concurrent processing (good for I/O-bound tasks)
MAX_RETRIES = 5         # Maximum retries for each request upon failure (e.g., 429 response)
MAX_PREFIX_LENGTH = 10  # Maximum length of word/prefix as per API constraint (words are <=10 letters)
MAX_RESULTS = 10        # Maximum count of words returned per request (as discovered)
RETRY_DELAY = 1         # Base delay in seconds after a 429 error (we use exponential backoff)

# Rate limiting: server allows 100 requests per minute.
class TokenBucket:
    """Token bucket to enforce a maximum of 100 requests per minute."""
    def __init__(self, capacity, refill_period):
        self.capacity = capacity         # Maximum tokens (requests) available per period.
        self.tokens = capacity           # Current available tokens.
        self.refill_period = refill_period  # Refill period in seconds.
        self.last_refill = time.monotonic()
        self.lock = RLock()              # Use reentrant lock for thread safety.

    def consume(self):
        with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            # Refill the bucket if the period has passed.
            if elapsed > self.refill_period:
                self.tokens = self.capacity
                self.last_refill = now
            if self.tokens > 0:
                self.tokens -= 1
                return True
            # No tokens available: compute precise wait time and sleep.
            sleep_time = self.refill_period - elapsed + 0.1  # small buffer added
            logging.info(f"Token bucket empty. Sleeping for {sleep_time:.2f} seconds")
        time.sleep(sleep_time)
        return self.consume()

# Create a token bucket allowing 100 requests per 60 seconds.
bucket = TokenBucket(capacity=100, refill_period=60)

# Shared resources for managing state across threads.
all_names = set()           # Stores unique names returned by the API.
explored_prefixes = set()   # Tracks which prefixes have been queried.
prefix_queue = Queue()      # Queue for dynamic prefix exploration.
data_lock = RLock()         # Protects shared data during concurrent access.
request_counter = 0         # Total number of API requests made.
last_saved_request = 0      # Tracks request count when progress was last saved.
stop_event = threading.Event()  # Signals threads to stop (e.g., on Ctrl+C).

def save_progress():
    """Save current progress (names and stats) to disk after every 100 requests."""
    suffix = f"_{API_VERSION}"
    try:
        # Save unique names to file.
        with open(f'unique_names{suffix}.txt', 'w') as f:
            f.write('\n'.join(sorted(all_names)))
        # Save statistics about progress.
        with open(f'stats{suffix}.txt', 'w') as f:
            f.write(f"Total unique names: {len(all_names)}\n")
            f.write(f"Total requests sent: {request_counter}\n")
            # f.write(f"Total prefixes explored: {len(explored_prefixes)}\n")
        logging.info("Progress saved to names and stats files.")
    except Exception as e:
        logging.error(f"Save error: {str(e)}")

def query_api(prefix):
    """
    Query the autocomplete API with a given prefix.
    Implements exponential backoff for 429 responses (since no Retry-After header is provided).
    """
    global request_counter, last_saved_request
    retries = 0
    while retries <= MAX_RETRIES and not stop_event.is_set():
        try:
            # Ensure we don't exceed the rate limit.
            bucket.consume()
            with data_lock:
                request_counter += 1
                # Save progress every 100 requests.
                if request_counter - last_saved_request >= 100:
                    last_saved_request = request_counter
                    save_progress()

            # Make the GET request to the API with the prefix.
            response = requests.get(API_BASE, params={'query': prefix}, timeout=10)
            
            if response.status_code == 429:
                # API hit rate limit; since no Retry-After header is given, we use exponential backoff.
                delay = (2 ** retries) + 0.1  
                logging.warning(f"429 on {prefix}. Retry {retries} in {delay:.2f} seconds")
                time.sleep(delay)
                retries += 1
                continue
                
            response.raise_for_status()
            # Return the list of suggestions; note: maximum 10 words per request.
            return response.json().get('results', [])
            
        except Exception as e:
            logging.error(f"Error on {prefix}: {str(e)}")
            if retries >= MAX_RETRIES:
                return []
            retries += 1
            time.sleep(0.5)  # Small delay before retrying.
    return []

def worker():
    """Worker thread: processes prefixes from the queue and explores further prefixes if needed."""
    while not stop_event.is_set():
        try:
            prefix = prefix_queue.get(timeout=1)
        except Empty:
            continue

        with data_lock:
            # Skip processing if prefix is already explored or if it exceeds the maximum length.
            if prefix in explored_prefixes or len(prefix) > MAX_PREFIX_LENGTH:
                prefix_queue.task_done()
                continue
            explored_prefixes.add(prefix)

        suggestions = query_api(prefix)
        new_names = []
        with data_lock:
            # Add new names to the global set, filtering duplicates.
            for name in suggestions:
                if name not in all_names:
                    all_names.add(name)
                    new_names.append(name)
        
        # If the API returns the maximum number of suggestions and the prefix is not too long,
        # generate new prefixes by appending each lowercase letter (a-z).
        if len(suggestions) >= MAX_RESULTS and len(prefix) < MAX_PREFIX_LENGTH:
            new_prefixes = [prefix + chr(c) for c in range(97, 123)]
            with data_lock:
                for np in new_prefixes:
                    if np not in explored_prefixes:
                        prefix_queue.put(np)
        
        prefix_queue.task_done()
        logging.info(f"Processed {prefix} | New names: {len(new_names)} | Queue size: {prefix_queue.qsize()}")

def main():
    start_time = time.time()
    # Initialize the prefix queue with all single lowercase letters.
    for c in "abcdefghijklmnopqrstuvwxyz":
        prefix_queue.put(c)
    
    # Start worker threads for concurrent processing.
    threads = []
    for _ in range(MAX_WORKERS):
        t = Thread(target=worker)
        t.start()
        threads.append(t)
    
    try:
        # Monitor the prefix queue until all prefixes have been processed.
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
        # Log final statistics.
        logging.info("Final progress saved.")
        logging.info(f"Total names extracted: {len(all_names)}")
        logging.info(f"Total API requests made: {request_counter}")
        logging.info(f"Total execution time: {elapsed_time:.2f} seconds")
        # Print summary details.
        print(f"Total new names extracted: {len(all_names)}")
        print(f"Total API requests sent: {request_counter}")
        print(f"Total execution time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()
