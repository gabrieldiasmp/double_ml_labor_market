import json
from pathlib import Path

HYPERPARAMETERS_PATH = Path(__file__).parent / "hyperparameters"

def import_hyperparameters():
    
    json_params = {}
    
    # Open the JSON file
    with open(HYPERPARAMETERS_PATH + "random_forest.json", 'r') as file:
        # Load the JSON data
        json_params["random_forest_hp"] = json.load(file)

    return json_params