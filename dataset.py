import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Set seed for reproducibility
np.random.seed(42)

# Number of data points
n = 100000

# Generate ItemID (Unique)
ItemID = np.arange(1, n+1)

# Generate Timestamp (start from current time and add random seconds)
start_time = datetime.now()
Timestamp = [start_time + timedelta(seconds=int(random.uniform(0, 1000000))) for _ in range(n)]

# Convert Timestamps to human-readable format (YYYY-MM-DD HH:MM:SS)
Timestamp = [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in Timestamp]

# Generate AccessFrequency (random values between 1 and 100)
AccessFrequency = np.random.randint(1, 101, n)

# Generate RecencyOfAccess (random values between 1 and 50)
RecencyOfAccess = np.random.randint(1, 51, n)

# Generate BurstPattern (random values between 1 and 3)
BurstPattern = np.random.randint(1, 4, n)

# Generate ItemSize (random values between 10 and 100)
ItemSize = np.random.randint(10, 101, n)

# Generate Priority (random values between 1 and 5)
Priority = np.random.randint(1, 6, n)

# Create a DataFrame
df = pd.DataFrame({
    'ItemID': ItemID,
    'Timestamp': Timestamp,
    'AccessFrequency': AccessFrequency,
    'RecencyOfAccess': RecencyOfAccess,
    'BurstPattern': BurstPattern,
    'ItemSize': ItemSize,
    'Priority': Priority
})

# Save to Excel
df.to_excel('lru_cache_data.xlsx', index=False)

print(f"Dataset of {n} entries generated and saved as 'lru_cache_data.xlsx'")
