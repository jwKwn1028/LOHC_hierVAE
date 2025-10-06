import logging
import sys
from props import similarity, penalized_logp

for line in sys.stdin:
    x,y = line.split()[:2]
    if y == "None": 
        y = None
    sim = similarity(x, y)
    try:
        prop = penalized_logp(y) - penalized_logp(x)
        print(x, y, sim, prop)
    except Exception as e:
        logging.warning(f"logp_score script failed for {e}")
        print(x, y, sim, 0.0)

