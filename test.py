import json
from lsa import preprocess, compute

with open('server/lsa_config.json', 'r') as f:
    config = json.load(f)
preprocess_cfg = config['preprocess']
compute_cfg = config['compute']

print('start', flush=True)
df_tf_idf = preprocess('server/data', 'tmp', **preprocess_cfg)
compute(df_tf_idf, cache_dir='tmp', **compute_cfg)
print('end', flush=True)
