import os
import csv
import random

def generate_mock_diabetes_dataset(filename, num_records=1000):
    """
    Generates a realistic mock dataset for diabetes diagnosis.
    Features: Glucose, BloodPressure, Insulin, BMI, Age, Outcome (0 or 1)
    """
    header = ["Glucose", "BloodPressure", "Insulin", "BMI", "Age", "Outcome"]
    
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        for _ in range(num_records):
            outcome = random.choice([0, 1])
            if outcome == 1:
                glucose = random.randint(120, 199)
                bp = random.randint(70, 110)
                insulin = random.randint(80, 300)
                bmi = round(random.uniform(25.0, 45.0), 1)
                age = random.randint(30, 70)
            else:
                glucose = random.randint(70, 125)
                bp = random.randint(60, 85)
                insulin = random.randint(15, 120)
                bmi = round(random.uniform(18.5, 29.9), 1)
                age = random.randint(18, 50)
                
            writer.writerow([glucose, bp, insulin, bmi, age, outcome])
            
    print(f"Generated {num_records} mock records in {filename}")

def partition_dataset(raw_file, output_dir, num_clients=5):
    """
    Partitions the raw CSV dataset into non-IID subsets for simulated federated clients.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(raw_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
        
    random.shuffle(rows)
    
    # Non-IID split: Client 0-1 get mostly healthy records, 3-4 get mostly diabetic records
    client_buckets = [[] for _ in range(num_clients)]
    
    for row in rows:
        outcome = int(row[-1])
        if outcome == 0:
            # Route healthy cases primarily to clients 0, 1, 2
            target_client = random.choices([0, 1, 2, 3, 4], weights=[0.4, 0.3, 0.2, 0.05, 0.05])[0]
        else:
            # Route diabetic cases primarily to clients 3, 4, 2
            target_client = random.choices([0, 1, 2, 3, 4], weights=[0.05, 0.05, 0.2, 0.35, 0.35])[0]
            
        client_buckets[target_client].append(row)
        
    for i in range(num_clients):
        client_filename = os.path.join(output_dir, f"client_{i}.csv")
        with open(client_filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(client_buckets[i])
        print(f"Client {i} dataset saved: {client_filename} ({len(client_buckets[i])} rows)")

if __name__ == "__main__":
    dataset_dir = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.join(dataset_dir, "diabetes_raw.csv")
    
    generate_mock_diabetes_dataset(raw_path, num_records=1200)
    partition_dataset(raw_path, dataset_dir)
