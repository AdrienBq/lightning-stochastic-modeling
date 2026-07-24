import os
import sys
import logging

# we add to PATH the root folder
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))      # path to directory containing the script
sys.path.append(root_path)
logging.basicConfig(filename=os.path.join(root_path, 'output.log'), level=logging.INFO)

# create console handler and set level to info
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# create formatter and add it to the handler
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")
console_handler.setFormatter(formatter)
