import os
import glob
import pandas as pd

cwd = os.getcwd()
raw_data_path = os.path.join(cwd,'data/raw_data/')

extension = 'csv'
all_filenames = [i for i in glob.glob(raw_data_path+'*.{}'.format(extension))]

#combine all files in the list
header=['id','scantype','scandesc','report']
combined_csv = pd.concat([pd.read_csv(f, header=None, names=header) for f in all_filenames ])


#export to csv
out_path = os.path.join(cwd,'data/combined_csv.csv')
combined_csv.to_csv(out_path, index=False, encoding='utf-8-sig')