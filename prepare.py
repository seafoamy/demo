import os
import sys
import pandas as pd

def gather_data(train_path, test_path):
    
    # Get data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Clean data
    # Fill in missing Cabin as Missing
    train_df["Cabin"] = train_df["Cabin"].fillna("Missing")
    test_df["Cabin"] = test_df["Cabin"].fillna("Missing")

    # Drop Ticket
    train_df = train_df.drop(["Ticket"], axis = 1)
    test_df = test_df.drop(["Ticket"], axis = 1)

    # Store data
    os.makedirs(os.path.join(os.getcwd(), "data"), exist_ok=True)

    train_df.to_csv(os.path.join(os.getcwd(), "data", "train.csv"))
    test_df.to_csv(os.path.join(os.getcwd(), "data", "test.csv"))
    return

if len(sys.argv) != 3:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython prepare.py train-file test-file\n")
    sys.exit(1)

train = sys.argv[1]
test = sys.argv[2]
gather_data(train, test)