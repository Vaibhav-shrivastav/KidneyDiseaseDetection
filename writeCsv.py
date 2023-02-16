import csv
import pandas as pd
import numpy as np



def write_to_csv(time,specificGravity, albumin, redBloodCellCount, serumCreatinine, sugar, diabetesMellitus, hemoglobin,packedCellVolume,hypertension,result):

    with open('dataset/records.csv', 'r') as f:
        reader = csv.reader(f)
        for header in reader:
            break
    with open('dataset/records.csv', "a", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        dict = {'time':time,'specificGravity':specificGravity,'albumin':albumin,'redBloodCellCount':redBloodCellCount,'serumCreatinine':serumCreatinine,
                'sugar':sugar,'diabetesMellitus':diabetesMellitus,'hemoglobin':hemoglobin,'packedCellVolume':packedCellVolume,
                'hypertension':hypertension,'result':result}
        writer.writerow(dict)