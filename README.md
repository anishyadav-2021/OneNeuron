# OneNeuron
OneNeuron | Perceptron
## Useful Link
[git handbook] (https://guides.github.com/introduction/git-handbook/)
## add image files
![and output](E:\DL\iNeuron\Practice\OneNeuron\plots\and.png)

## Add image with html format
<img src="E:\DL\iNeuron\Practice\OneNeuron\plots\and.png" alt="AND o/p Image" width="200" height="200">

```python
from utils.model import Perceptron
from utils.all_utils import prepare_data, save_plot, save_model
import pandas as pd
import logging
import os

logging_str="[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
log_dir="logs"
os.makedirs(log_dir,exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir,"running_logs.log"),level=logging.INFO, format=logging_str,filemode="a")



def main(data, modelName, plotName, eta, epochs):
    df = pd.DataFrame(data)
    logging.info(f"This is actual dataframe{df}")
    X, y = prepare_data(df)
    model = Perceptron(eta=eta, epochs=epochs)
    model.fit(X, y)
    _ = model.total_loss()
    save_model(model, filename=modelName)
    save_plot(df, plotName, model)


if __name__ == '__main__':
    OR = {
        "x1": [0,0,1,1],
        "x2": [0,1,0,1],
        "y": [0,1,1,1],
    }
    ETA = 0.3 # 0 and 1
    EPOCHS = 100
    try:
        logging.info(">>>>>>>Training is going to start>>>>>>>")
        main(data=OR, modelName="or.model", plotName="or.png", eta=ETA, epochs=EPOCHS)
        logging.info("<<<<<<<<Training is completed<<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e
```
```bash
git add .
git commit -m "Type message"
git push origin main
```
#headings
##sub-headings
###sub-sub-headings
## make points
1. point 1
* point1
## MAke tables
x1|x2|x3
-|-|-
0|0|0
0|1|0
