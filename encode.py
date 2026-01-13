
# %%

import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import pickle

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer


# %%

root = "./data"
dataset = "Amazon2014Beauty_550_LOU"
# dataset = "Amazon2014Toys_550_LOU"
# dataset = "Amazon2014Sports_550_LOU"
# dataset = "Yelp2018_10100_LOU"
# dataset = "OnlineRetail"
path = os.path.join(root, "Processed", dataset)
item_df = pd.read_csv(os.path.join(path, "item.txt"), sep='\t')

if 'Amazon' in dataset:
    fields = ('TITLE', 'CATEGORIES', 'BRAND')
elif 'Yelp' in dataset:
    fields = ('ITEM_NAME', 'CATEGORIES', 'CITY')
elif 'Retail' in dataset:
    fields = ('TEXT',)

for field in fields:
    item_df[field] = item_df[field].fillna('')

item_df['TEXT'] = item_df.apply(
    lambda row: "\n".join([f"{field}: {row[field]}." for field in fields]).strip(),
    axis=1
)

print(item_df['TEXT'].head(5))

# %%


def export_pickle(data, file: str):
    r"""
    Export data into pickle format.

    data: Any
    file: str
        The file (path/filename) to be saved
    """
    fh = None
    try:
        fh = open(file, "wb")
        pickle.dump(data, fh, pickle.HIGHEST_PROTOCOL)
    except (EnvironmentError, pickle.PicklingError) as err:
        ExportError_ = type("ExportError", (Exception,), dict())
        raise ExportError_(f"Export Error: {err}")
    finally:
        if fh is not None:
            fh.close()

def encode_textual_modality(
    item_df: pd.DataFrame,
    model: str = "sentence-t5-xl", model_dir: str = "./models",
    batch_size: int = 128
):
    saved_filename = f"{model}_{'_'.join(fields)}.pkl".lower()
    sentences = item_df['TEXT']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = SentenceTransformer(
        os.path.join(model_dir, model),
        device=device
    ).eval()

    with torch.no_grad():
        tFeats = encoder.encode(
            sentences, 
            convert_to_tensor=True,
            batch_size=batch_size, show_progress_bar=True
        ).cpu()
    assert tFeats.size(0) == len(item_df), f"Unknown errors happen ..."

    if 'Amazon' in dataset:
        tFeats = tFeats - tFeats.mean(0, keepdim=True)

    export_pickle(
        tFeats,
        os.path.join(path, saved_filename)
    )

    return tFeats


# %%

encode_textual_modality(item_df, model="sentence-t5-xl")