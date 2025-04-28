import os
import csv

def store_data_csv(data, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

def load_data_csv(filename):
    if not os.path.exists(filename):
        print(f"{filename} does not exist!")
        return None
    with open(filename, mode='r', newline='') as file:
        reader = csv.reader(file)
        data = [list(map(int, row)) for row in reader]
    return data
