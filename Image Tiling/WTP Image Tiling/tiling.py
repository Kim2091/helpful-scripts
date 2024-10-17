import json
from src import Tiler
with open(r"config.json") as f:
    config = json.load(f)
process = Tiler(config)
process.run()
