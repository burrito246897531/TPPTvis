'''
Notes for use:
 - Place any coinc.dat files in the same directory, it will automatically create the ccounts, lcounts, and toplors files
 - ccounts.bin - channel counts: counts per channel
 - lcounts.bin - LOR counts: counts per LOR (not currently used)
 - toplors.bin - top 1000 LORs to be drawn in TPPTvis
'''

import pandas as pd
import numpy as np
import os

# Find all files in the directory ending in .dat
current_dir = os.path.dirname(os.path.abspath(__file__))
coinc_dat_files = [os.path.join(current_dir, f) for f in os.listdir(current_dir) if f.endswith('.dat')]
print(coinc_dat_files)

channelcounts = [np.zeros(6144, dtype=np.int32) for _ in coinc_dat_files]
lorcounts = [np.zeros((3072, 3072), dtype=np.int16) for _ in coinc_dat_files]
toplors = [np.zeros((1000, 3), dtype=np.int16) for _ in coinc_dat_files]

for i, file_name in enumerate(coinc_dat_files):
    for data in pd.read_csv(file_name, sep='\t', header=None, usecols=[4, 9], chunksize=1000000):
        idls = (data.iloc[:, 0].values.astype(np.int32) - 131072) % 3072
        idrs = data.iloc[:, 1].values.astype(np.int32) % 3072 + 3072
        np.add.at(channelcounts[i], idls, 1)
        np.add.at(channelcounts[i], idrs, 1)
        idrs = idrs - 3072
        np.add.at(lorcounts[i], (idls, idrs), 1)

    #print(channelcounts[i])
    # Find the top 1000 LORs by count from lorcounts[i]
    flat_indices = np.argpartition(lorcounts[i].ravel(), -1000)[-1000:]
    sorted_indices = flat_indices[np.argsort(lorcounts[i].ravel()[flat_indices])[::-1]]
    idls, idrs = np.unravel_index(sorted_indices, lorcounts[i].shape)
    counts = lorcounts[i][idls, idrs]
    toplors[i][:, 0] = idls
    toplors[i][:, 1] = idrs
    toplors[i][:, 2] = counts

    # Save channelcounts and lorcounts after processing each file
    # Save channelcounts
    ch_filename = os.path.splitext(file_name)[0] + '_ccounts.bin'
    with open(ch_filename, 'wb') as ch_fp:
        channelcounts[i].tofile(ch_fp)
    # Save lorcounts
    lor_filename = os.path.splitext(file_name)[0] + '_lcounts.bin'
    with open(lor_filename, 'wb') as lor_fp:
        lorcounts[i].tofile(lor_fp)
    # Save toplors
    topl_filename = os.path.splitext(file_name)[0] + '_toplors.bin'
    with open(topl_filename, 'wb') as topl_fp:
        toplors[i].tofile(topl_fp)



