import logging
import os
from datetime import datetime
import sys

LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path=os.path.join(os.getcwd(),"logs",LOG_FILE)
os.makedirs(logs_path,exist_ok=True)

LOG_FILE_PATH=os.path.join(logs_path,LOG_FILE)

logging.basicConfig(
    format="[%(asctime)s] - %(levelname)s \nFile    : [%(module)s] File \nLineNum : %(lineno)d - %(name)s \nMessage : %(message)s \n-----------------------",
    level=logging.INFO,
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler(sys.stdout)
    ]
)

logger=logging.getLogger("BBC_News_Classification_LSTM")
