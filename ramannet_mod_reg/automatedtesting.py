import pandas as pd
import numpy as np 
import diversipy as dp
import scipy as sp
import random

import train_model_reg as tm
import Train_reg as tr
import test as tst
from doepy import build
""" 
Automated testing script to run training and evaluation of the regression model. 

doing DEO on:

- loss weights for multi-output model
- number of PCA components used
- data range

loss weights:
rpet and PCA range from 0 to 0.7 in increments of 0.1


"""
factors = {
    "rpet_loss_weights": [0, 1, 2, 3, 4, 5, 6, 7],
    "PCA_loss_weights":  [0, 1, 2, 3, 4, 5, 6, 7],

    "PCA_vals": [3,4,5],

    "data_lower": [1100, 600, 400, 200, 1200],
    "data_upper": [3000, 1800,1300,1000, 800],
}

# design = build.full_fact(factors)

# print(design)

design = build.full_fact(factors)

sample = random.sample(range(1, 3000),50)

# results = {"MAE" : 1000}
for x in sample:
    try:
        rpet_loss = design['rpet_loss_weights'][x] * 0.1
        PCA_loss = design['PCA_loss_weights'][x] * 0.1
        PCA_val = int(design['PCA_vals'][x])

        data_upper = design['data_upper'][x]
        data_lower = design['data_lower'][x]


        print(f"Running test {x} with rpet loss weight: {rpet_loss}, PCA loss weight: {PCA_loss}, PCA range: {PCA_val}, data range: {data_lower} to {data_upper}")
        
        PCA_train = f"PCA_model/PCA_training{PCA_val}.csv"
        PCA_test = f"PCA_model/PCA_test{PCA_val}.csv"
        factors = [rpet_loss, PCA_loss, PCA_val, data_upper, data_lower]

        tr.train_reg(PCA_val, data_upper, data_lower, rpet_loss, PCA_loss)
        metrics = tst.test(PCA_val, data_upper, data_lower)
        mae = metrics["MAE"]

        # summary = {
        #     "test_number": x,
        #     "rpet_loss_weight": rpet_loss,
        #     "PCA_loss_weight": PCA_loss,
        #     "PCA_val": PCA_val,
        #     "data_upper": data_upper,
        #     "data_lower": data_lower,
        #     "MAE": mae
        # }
        
        # if summary["MAE"] < results["MAE"]:
        #     results = summary

        print(f"Test {x} complete. Metrics: {metrics}")
        print("writing to file")
        
        from pathlib import Path
        out = Path(__file__).with_name("DOE.txt")
        with out.open("a", encoding="utf-8") as f:
            f.write(f"Test {x} with rpet loss weight: {rpet_loss}, PCA loss weight: {PCA_loss}, PCA range: {PCA_val}, data range: {data_lower} to {data_upper}, MAE: {mae}\n")
        print("Wrote to", out)
        # try:
        #     out = (Path(__file__).with_name("DOE.txt")).resolve()
        #     with out.open("a", encoding="utf-8") as f:
        #         f.write(
        #             f"Test {x} with rpet loss weight: {rpet_loss}, PCA loss weight: {PCA_loss}, "
        #             f"PCA range: {PCA_val}, data range: {data_lower} to {data_upper}, MAE: {mae}\n"
        #         )
        #     print("Wrote to", out)
        # except Exception as e:
        #     print("Failed to write DOE:", e)

    except Exception as e:
        print(f"Test {x} failed due to error: {e}")
        with open("DOE_errors.txt", "a") as error_log:
            error_log.write(f"Test {x} with rpet loss weight: {rpet_loss}, PCA loss weight: {PCA_loss}, PCA range: {PCA_val}, data range: {data_lower} to {data_upper} failed due to error: {e}\n")
        continue

# out = Path(__file__).with_name("DOE.txt")
# with out.open("a", encoding="utf-8") as f:
#     f.write("\nBest Result:\n")
#     f.write(f"Test {summary['test_number']} with rpet loss weight: {summary['rpet_loss_weight']}, PCA loss weight: {summary['PCA_loss_weight']}, PCA range: {summary['PCA_val']}, data range: {summary['data_lower']} to {summary['data_upper']}\n")