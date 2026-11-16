"""Call locally running API using Python standard library."""
import json
from pathlib import Path
from urllib.request import Request, urlopen
req = Request('http://127.0.0.1:8000/predict',
              data=(Path(__file__).resolve().parent / 'request.json').read_bytes(),
              headers={'Content-Type': 'application/json'}, method='POST')
with urlopen(req, timeout=15) as response:
    print(json.dumps(json.load(response), indent=2))
