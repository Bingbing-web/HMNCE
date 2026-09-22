import numpy as np
import argparse
import pickle
import pandas as pd
from pathlib import Path

parser = argparse.ArgumentParser(description='Create literals')
parser.add_argument('--dataset', default='credit', metavar='',
                    help='Dataset folder name')
parser.add_argument('--data-root', type=Path, default=Path(__file__).resolve().parent / 'datasets',
                    help='Directory containing dataset folders')
args = parser.parse_args()
dataset_dir = (args.data_root / args.dataset).resolve()
literal_dir = dataset_dir / 'literals'
literal_dir.mkdir(parents=True, exist_ok=True)

train=pd.read_csv(dataset_dir / 'train.txt', sep="\t",header=None,names=['head','relation','tail'])
valid=pd.read_csv(dataset_dir / 'valid.txt', sep="\t",header=None,names=['head','relation','tail'])
test=pd.read_csv(dataset_dir / 'test.txt', sep="\t",header=None,names=['head','relation','tail'])
neg=pd.read_csv(dataset_dir / 'neg.txt', sep="\t",header=None,names=['head','relation','tail'])

all_df=pd.concat([train, valid, test], ignore_index=True)

print("# of Triplets", len(all_df))
print('# of Triplets (train)', len(train))
print('# of Triplets (valid)', len(valid))
print('# of Triplets (test)', len(test))

# Entity dictionary
ent=set(all_df['head'].unique()) | set(all_df['tail'].unique())
entities = sorted({str(e) for e in ent})
entity_dict = {v: k for k, v in enumerate(entities)}

print("# of Entites:", len(entity_dict))

with (dataset_dir / 'entities.dict').open('wb') as fw:
    pickle.dump(entity_dict, fw)

# Relation dictionary
relations=all_df['relation'].unique()
relation_dict = {v: k for k, v in enumerate(relations)}


print("# of Entity Relations:",len(relation_dict))
with (dataset_dir / 'relations.dict').open('wb') as fw:
    pickle.dump(relation_dict, fw)

# Load raw literals
df = pd.read_csv(literal_dir / 'numerical_literals.txt', header=None, sep='\t')

numrel_dict = {v: k for k, v in enumerate(df[1].unique())}

print("# of Attribute Triples: ", len(df))

# Resulting file
num_lit = np.zeros([len(entity_dict), len(numrel_dict)], dtype=np.float32)

for i, (s, p, lit) in enumerate(df.values):
    try:
        entity_key = str(s)
        if "id" in p:
            num_lit[entity_dict[entity_key], numrel_dict[p]] = 1.0
        else:
            num_lit[entity_dict[entity_key], numrel_dict[p]] = lit

    except KeyError:
        continue

np.save(literal_dir / 'numerical_literals.npy', num_lit)


M = train.shape[0]
X = np.zeros([M, 3], dtype=int)
for i, row in train.iterrows():
    X[i, 0] = entity_dict[str(row[0])]
    X[i, 1] = relation_dict[row[1]]
    X[i, 2] = entity_dict[str(row[2])]

np.save(dataset_dir / 'train.npy', X.astype(np.int32))







