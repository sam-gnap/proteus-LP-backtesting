import json
import os
from datetime import datetime
from src.data_processing.fetch_data import fetch_top_pools
from config.api_config import API_KEY

def main():
    # Fetch top 100 pools
    top_pools = fetch_top_pools(API_KEY, num_pools=100)

    # Create a directory for the data if it doesn't exist
    os.makedirs("data/processed", exist_ok=True)

    # Generate a filename with the current date and time
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data/processed/top_pools_{timestamp}.json"

    # Save the data to a JSON file
    with open(filename, "w") as f:
        json.dump(top_pools, f, indent=2)

    print(f"Top pools data saved to {filename}")


if __name__ == "__main__":
    main()
