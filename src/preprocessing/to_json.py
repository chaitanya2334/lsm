from definitions import ROOT_DIR
import glob
import itertools as it
import json
import os

from src.corpus.proto_file import ProtoFile
from src.corpus.tokenizers import SplitTokenizer, WordPieceTokenizer
from tqdm import tqdm


def to_json(
    src, dest_dir, dest_name, firstn=None, kernel_size=None, stride=None
):
    files = [
        os.path.splitext(f)[0]
        for f in glob.iglob(f"{src}/*.ann", recursive=True)
    ]

    if firstn:
        files = sorted(files)[:firstn]

    docs = [
        ProtoFile(
            filename=filename,
            to_lower=True,
            replace_digits=True,
            gen_features=False,
            rel_undirected_classes=["Or", "Overlaps", "Coreference-Link"],
            tokenizer=WordPieceTokenizer(lowercase=True, replace_digits=True),
            process_titles=True
        ) for filename in tqdm(files)
    ]

    if kernel_size and stride:
        json_docs = list(
            it.chain.from_iterable(
                [
                    doc.tile_to_json(kernel_size=kernel_size, stride=stride)
                    for doc in docs
                ]
            )
        )

    else:
        # no tiling
        json_docs = [doc.to_json(0, prefix=False) for doc in docs]

    save_file = os.path.join(dest_dir, f"{dest_name}.json")

    json_docs = [j + "\n" for j in json_docs]

    with open(save_file, 'w') as f:
        f.writelines(json_docs)


def wlp_csr_dataset():
    json_dir = os.path.join(ROOT_DIR, f"./data_abb/wlp_csr_k5_s2")
    os.makedirs(json_dir, exist_ok=True)
    to_json(
        src="wlp-mstg-dataset/train",
        dest_dir=json_dir,
        dest_name="train",
        kernel_size=5,
        stride=2
    )
    to_json(
        src="wlp-mstg-dataset/dev",
        dest_dir=json_dir,
        dest_name="dev",
        kernel_size=5,
        stride=2
    )
    to_json(
        src="wlp-mstg-dataset/test",
        dest_dir=json_dir,
        dest_name="test",
        kernel_size=5,
        stride=2
    )

    json_dir = os.path.join(ROOT_DIR, f"./data_abb/wlp_csr_k5_s2")
    # os.makedirs(json_dir, exist_ok=True)
    # to_json(
    #     src="wlp-csr-dataset/train", dest=os.path.join(json_dir, "full_train")
    # )
    to_json(src="wlp-mstg-dataset/dev", dest_dir=json_dir, dest_name="full_dev")
    to_json(
        src="wlp-mstg-dataset/test", dest_dir=json_dir, dest_name="full_test"
    )


if __name__ == "__main__":
    wlp_csr_dataset()
