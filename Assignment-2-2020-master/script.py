import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import json
import collections

import csv
import os
import re



if __name__ == "__main__":
    with open('out/countlines2.json', mode='r') as file:
        countlines = json.loads(file.read())

    with open('out/test2.json', mode='r') as file:
        jsinspect = json.loads(file.read())

    filelines = {}

    for file in countlines.keys():
        if file != 'header' and file != 'SUM':
            version = re.search(r"[\d.]+", file).group(0)
            if version == '':
                print("blabla")
            if version in filelines:
                filelines[version] += countlines[file]['blank'] + countlines[file]['comment'] + countlines[file]['code']
            else:
                filelines[version] = countlines[file]['blank'] + countlines[file]['comment'] + countlines[file]['code']

    sim = {}

    for match in jsinspect:
        match_sim = {}
        for inst in match['instances']:
            version = re.search(r"[\d.]+", inst['path'][14:]).group(0)
            no_lines = inst['lines'][1] - inst['lines'][0] + 1

            if version in match_sim:
                match_sim[version] += no_lines
            else:
                match_sim[version] = no_lines

        cache = []
        
        for version1 in sorted(match_sim.keys(), key=lambda s: list(map(int, s.split('.')))):
            cache.append(version1)
            for version2 in match_sim.keys():
                if version2 not in cache:
                    if (version1, version2) in sim:
                        sim[(version1, version2)] += match_sim[version1]
                    else:
                        sim[(version1, version2)] = match_sim[version1]

    for (version1, version2) in sim.keys():
        max_lines = max(filelines[version1], filelines[version2])
        sim[(version1, version2)] /= max_lines
        sim[(version1, version2)] /= 2

    od = sorted(sim.items(), key=lambda s: list(map(int, s[0][0].split('.'))))

    for i in od:
        print(i)
    
    ser = pd.Series(list(sim.values()),
                    index=pd.MultiIndex.from_tuples(sim.keys()))
    df = ser.unstack().fillna(0)
    df = df.reindex(sorted(df.columns, key=lambda s: list(map(int, s.split('.')))))
    df = df.reindex(sorted(df, key=lambda s: list(map(int, s.split('.')))), axis=1)
    df = df.transpose()
    ax = sns.heatmap(df, xticklabels=1, yticklabels=1)
    last = "0"
    count = 0
    sep = []
    lastBig = "1"
    sepBig = []
    for i, my_df in df.items():
        
        if i.split('.')[0] != lastBig:
            print(i)
            lastBig = i.split('.')[0]
            sepBig.append(count)
        count += 1
        sep.append(count)
        
    d1, d2 = ax.get_xlim()
    d2 -= 1
    ax.hlines(sep, *(d1, d2), color='white', linewidth=0.5)
    ax.vlines(sep, *ax.get_ylim(), color='white', linewidth=0.5)

    ax.hlines(sepBig, *(d1, d2), color='white', linewidth=1.5)
    ax.vlines(sepBig, *ax.get_ylim(), color='white', linewidth=1.5)

    # od = sorted(filelines.items(), key=lambda s: list(map(int, s[0].split('.'))))

    # print(od)

    # keys = [x[0] for x in od]
    # vals = [x[1] for x in od]

    # for ind, label in enumerate(plot_.get_xticklabels()):
    #     if ind % 10 == 0: 
    #         label.set_visible(True)
    #     else:
    #         label.set_visible(False)

    plt.show()