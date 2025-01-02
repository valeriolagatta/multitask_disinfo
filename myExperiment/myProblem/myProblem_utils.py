# Copyright (c) Microsoft. All rights reserved.
from random import shuffle
from data_utils.metrics import calc_metrics

def load_htiee(file, header=True):
    """Loading data of topic for classification"""
    rows = []
    cnt = 0
    with open(file, encoding="utf8") as f:
        for line in f:
            if header:
                header = False
                continue
            blocks = line.strip().split("\t")
            assert len(blocks) > 3
        
            lab = blocks[-1]
            if lab is None:
                import pdb

                pdb.set_trace()
            sample = {
                "uid": blocks[0],
                "premise": blocks[2],
                "label": lab,
            }
            rows.append(sample)
            cnt += 1
    return rows

def load_sqdc(file, header=True):
    """Loading data of stance for classification"""
    rows = []
    cnt = 0
    with open(file, encoding="utf8") as f:
        for line in f:
            if header:
                header = False
                continue
            blocks = line.strip().split("\t")
            assert len(blocks) > 6
        
            lab = blocks[-1]
            if lab is None:
                import pdb

                pdb.set_trace()
            sample = {
                "uid": blocks[0],
                "premise": blocks[4],
                "hypothesis": blocks[5],
                "label": lab,
            }
            rows.append(sample)
            cnt += 1
    return rows

def load_fnd(file, header=True):
    """Loading data of veracity for classification"""
    rows = []
    cnt = 0
    with open(file, encoding="utf8") as f:
        for line in f:
            if header:
                header = False
                continue
            blocks = line.strip().split("\t")
            assert len(blocks) > 3
            
            lab = blocks[-1]
            if lab is None:
                import pdb

                pdb.set_trace()
            sample = {
                "uid": blocks[0],
                "premise": blocks[2],
                "label": lab,
            }
            rows.append(sample)
            cnt += 1
    return rows

def load_s(file, header=True):
    """Loading data of sentiment for classification"""
    rows = []
    cnt = 0
    with open(file, encoding="utf8") as f:
        for line in f:
            if header:
                header = False
                continue
            blocks = line.strip().split("\t")
            assert len(blocks) > 3
            
            lab = blocks[-1]
            if lab is None:
                import pdb

                pdb.set_trace()
            sample = {
                "uid": blocks[0],
                "premise": blocks[2],
                "label": lab,
            }
            rows.append(sample)
            cnt += 1
    return rows

def submit(path, data, label_dict=None):
    header = "index\tprediction"
    with open(path, "w") as writer:
        predictions, uids = data["predictions"], data["uids"]
        writer.write("{}\n".format(header))
        assert len(predictions) == len(uids)
        # sort label
        paired = [(int(uid), predictions[idx]) for idx, uid in enumerate(uids)]
        paired = sorted(paired, key=lambda item: item[0])
        for uid, pred in paired:
            if label_dict is None:
                writer.write("{}\t{}\n".format(uid, pred))
            else:
                assert type(pred) is int
                writer.write("{}\t{}\n".format(uid, label_dict[pred]))
