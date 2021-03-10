import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import json
import collections

import csv
import os
import re

def simHeatMap(simDeg):
    # arrange simDeg dict into a dataframe, then heatmap
    ser = pd.Series(list(simDeg.values()),
                    index=pd.MultiIndex.from_tuples(simDeg.keys()))
    df = ser.unstack().fillna(0)
    # resorting in order to have versions 1.10.0...1.12.4 after version 1.9 and before version 2.0
    df = df.reindex(sorted(df.columns, key=lambda s: list(map(int, s.split('.')))))
    df = df.reindex(sorted(df, key=lambda s: list(map(int, s.split('.')))), axis=1)
    df = df.transpose()
    ax = sns.heatmap(df, xticklabels=1, yticklabels=1)

    # set up separators for major versions
    count = 0
    sep = []
    lastBig = "1"
    sepBig = []
    for i, _ in df.items():
        if i.split('.')[0] != lastBig:
            lastBig = i.split('.')[0]
            sepBig.append(count)
        count += 1
        sep.append(count)
        
    # draw thin separator between all versions
    d1, d2 = ax.get_xlim()
    d2 -= 1
    ax.hlines(sep, *(d1, d2), color='white', linewidth=0.5)
    ax.vlines(sep, *ax.get_ylim(), color='white', linewidth=0.5)

    # draw thick separator between major versions
    ax.hlines(sepBig, *(d1, d2), color='white', linewidth=1.5)
    ax.vlines(sepBig, *ax.get_ylim(), color='white', linewidth=1.5)

    return ax

def simBarPlot(noLines):
    # resorting in order to have versions 1.10.0...1.12.4 after version 1.9 and before version 2.0
    sortedNoLines = sorted(noLines.items(), key=lambda s: list(map(int, s[0].split('.'))))

    keys = [x[0] for x in sortedNoLines]
    vals = [x[1] for x in sortedNoLines]

    plot_ = sns.barplot(x = keys, y = vals)

    # only show every 10th label in order to not get crowded
    for ind, label in enumerate(plot_.get_xticklabels()):
        if ind % 10 == 0: 
            label.set_visible(True)
        else:
            label.set_visible(False)
    
    return plot_

if __name__ == "__main__":
    # built using cloc, command: cloc --skip-uniqueness --by-file --fullpath --not-match-d="build|dist|test|Test|speed|external" --not-match-f="intro.js|outro.js|.min.js|.slim.js|Test" --include-lang=JavaScript --json jquery-data > /out/countlines.json
    with open('out/countlines.json', mode='r') as file:
        noLinesFile = json.loads(file.read())

    # built using jsinspect, command: jsinspect jquery-data -I -L -C --ignore "build|dist|test|Test|speed|external|intro.js|outro.js|.min.js|.slim.js" -r "json" --debug > /out/similarity.json
    with open('out/similarity.json', mode='r') as file:
        similarityFiles = json.loads(file.read())

    # total line count per version
    noLines = {}

    for file in noLinesFile.keys():
        if file != 'header' and file != 'SUM':
            # extract version number
            version = re.search(r"[\d.]+", file).group(0)
            # add number of lines in file to total line count of version
            if version in noLines:
                noLines[version] += noLinesFile[file]['blank'] + noLinesFile[file]['comment'] + noLinesFile[file]['code']
            else:
                noLines[version] = noLinesFile[file]['blank'] + noLinesFile[file]['comment'] + noLinesFile[file]['code']

    # similarity between versions, access as sim[(version1, version2)]
    sim = {}

    # loop through all similarity matches given by jsinspect
    for match in similarityFiles:
        match_sim = {}
        for inst in match['instances']:
            # extract version number
            version = re.search(r"[\d.]+", inst['path'][14:]).group(0)
            # count number of lines in match instance
            no_lines = inst['lines'][1] - inst['lines'][0] + 1

            # store number of lines matched in this version in the current similarity match
            if version in match_sim:
                match_sim[version] += no_lines
            else:
                match_sim[version] = no_lines

        cache = []
        
        # loop through all pairs of versions in the current match
        for version1 in sorted(match_sim.keys(), key=lambda s: list(map(int, s.split('.')))):
            cache.append(version1)
            for version2 in match_sim.keys():
                if version2 not in cache:
                    # add number of lines for version 1 that are similar to some in version 2
                    if (version1, version2) in sim:
                        sim[(version1, version2)] += match_sim[version1]
                    else:
                        sim[(version1, version2)] = match_sim[version1]

    # divide each number of lines by max number of lines between version1 and version2
    # output should range from 0 to 1, representing the degree of similarity between version1 and version2
    for (version1, version2) in sim.keys():
        max_lines = max(noLines[version1], noLines[version2])
        sim[(version1, version2)] /= max_lines
        sim[(version1, version2)] /= 2

    simHeatMap(sim)

    # simBarPlot(noLines)

    plt.show()