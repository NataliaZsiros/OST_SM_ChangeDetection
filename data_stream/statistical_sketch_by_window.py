from collections import deque
from confluent_kafka import Consumer, KafkaError

# Sliding window size
WINDOW_SIZE = 10

# Kafka Consumer Configuration
consumer_conf = {
    'bootstrap.servers': 'kafka:9092',
    'group.id': 'stats-group',
    'auto.offset.reset': 'earliest'
}
consumer = Consumer(consumer_conf)
topic = 'input'
consumer.subscribe([topic])

# Define a dictionary to store sliding windows and statistics for each field
sliding_windows = {}

def update_statistics(field, value):
    """Update count and variance in the sliding window."""
    if field not in sliding_windows:
        sliding_windows[field] = {'window': deque(maxlen=WINDOW_SIZE), 'count': 0, 'variance': 0.0}

    # Add the current value to the sliding window
    sliding_windows[field]['window'].append(value)
    sliding_windows[field]['count'] += 1
    
    # Update variance using Welford's algorithm
    window_values = list(sliding_windows[field]['window'])
    if len(window_values) > 1:
        mean = sum(window_values) / len(window_values)
        variance = sum((x - mean) ** 2 for x in window_values) / len(window_values)
        sliding_windows[field]['variance'] = variance
    else:
        sliding_windows[field]['variance'] = 0

def process_field(field, value):
    """Process a field by updating its sliding window and statistics."""
    # Initialize sliding window for the field if not already present
    if field not in sliding_windows:
        sliding_windows[field] = {'window': deque(maxlen=WINDOW_SIZE), 'count': 0, 'variance': 0.0}

    # Add the current value to the sliding window
    sliding_windows[field]['window'].append(value)

    # Update the sliding window statistics (count, variance)
    update_statistics(field, value)

    # Calculate the statistics for the current sliding window
    window_mean = sum(sliding_windows[field]['window']) / len(sliding_windows[field]['window'])
    window_min = min(sliding_windows[field]['window'])
    window_max = max(sliding_windows[field]['window'])
    window_count = sliding_windows[field]['count']
    window_variance = sliding_windows[field]['variance']

    # Display the results including numeric value
    time.sleep(2)
    print(f"Field: {field}")
    print(f"Numeric Value: {value}")
    print(f"Sliding Window - Mean: {window_mean:.2f}, Min: {window_min}, Max: {window_max}, Count: {window_count}, Variance: {window_variance:.2f}")
    print("-" * 40)

try:
    print(f"Consuming messages from topic: {topic}")
    while True:
        msg = consumer.poll(1.0)
        if msg is None:
            print("No message received. Waiting...")
            continue
        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                print(f"End of partition reached {msg.topic()} [{msg.partition()}]")
            else:
                print(f"Consumer error: {msg.error()}")
            continue

        # Parse the message
        raw_msg = msg.value().decode('utf-8')
        try:
            data = dict(item.split('=') for item in raw_msg.split(','))
        except Exception as e:
            print(f"Error parsing message: {raw_msg}, Error: {e}")
            continue

        # Process numeric fields
        for field_name, value in data.items():
            try:
                numeric_value = float(value)  # Convert to numeric value
                process_field(field_name, numeric_value)  # Process the field and show the statistics

            except ValueError:
                print(f"Field: {field_name} is not numeric. Skipping.")

except KeyboardInterrupt:
    print("Consumer interrupted.")

finally:
    consumer.close()
