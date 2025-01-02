import os
import argparse
import random
from sys import path

path.append(os.getcwd())
from myExperiment.common_utils import dump_rows
from data_utils.task_def import DataFormat
from data_utils.log_wrapper import create_logger
from myExperiment.myProblem.myProblem_utils import *

logger = create_logger(__name__, to_disk=True, log_file="myProblem_prepro.log")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocessing myProblem dataset."
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--root_dir", type=str, default="data")  
    args = parser.parse_args()
    return args


def main(args):
    root = args.root_dir
    assert os.path.exists(root)

    ######################################
    # myProblem tasks                    #
    ######################################
    
    s_train_path = os.path.join(root, "s/train_S.tsv")
    s_dev_path = os.path.join(root, "s/dev_S.tsv")
    s_test_path = os.path.join(root, "s/test_S.tsv")
    
    fnd_train_path = os.path.join(root, "fnd/train_TF.tsv")
    fnd_dev_path = os.path.join(root, "fnd/dev_TF.tsv")
    fnd_test_path = os.path.join(root, "fnd/test_TF.tsv")
    
    sqdc_train_path = os.path.join(root, "sqdc/train_SQDC.tsv")
    sqdc_dev_path = os.path.join(root, "sqdc/dev_SQDC.tsv") 
    sqdc_test_path = os.path.join(root, "sqdc/test_SQDC.tsv")
    
    htiee_train_path = os.path.join(root, "htiee/train_HTIEE.tsv")
    htiee_dev_path = os.path.join(root, "htiee/dev_HTIEE.tsv") 
    htiee_test_path = os.path.join(root, "htiee/test_HTIEE.tsv")
    
    ######################################
    # Loading DATA                       #
    ######################################
    
    sentimentDetection_train_data = load_s(s_train_path)
    sentimentDetection_dev_data = load_s(s_dev_path)
    sentimentDetection_test_data = load_s(s_test_path)
    
    logger.info("Loaded {} sentiment train samples".format(len(sentimentDetection_train_data)))
    logger.info("Loaded {} sentiment dev samples".format(len(sentimentDetection_dev_data)))
    logger.info("Loaded {} sentiment test samples".format(len(sentimentDetection_test_data)))
    
    fakeNewsDetection_train_data = load_fnd(fnd_train_path)
    fakeNewsDetection_dev_data = load_fnd(fnd_dev_path)
    fakeNewsDetection_test_data = load_fnd(fnd_test_path)
    
    logger.info("Loaded {} veracity train samples".format(len(fakeNewsDetection_train_data)))
    logger.info("Loaded {} veracity dev samples".format(len(fakeNewsDetection_dev_data)))
    logger.info("Loaded {} veracity test samples".format(len(fakeNewsDetection_test_data)))

    stanceDetection_train_data = load_sqdc(sqdc_train_path)
    stanceDetection_dev_data = load_sqdc(sqdc_dev_path)
    stanceDetection_test_data = load_sqdc(sqdc_test_path)
    
    logger.info("Loaded {} stance train samples".format(len(stanceDetection_train_data)))
    logger.info("Loaded {} stance dev samples".format(len(stanceDetection_dev_data)))
    logger.info("Loaded {} stance test samples".format(len(stanceDetection_test_data)))
    
    topicDetection_train_data = load_htiee(htiee_train_path)
    topicDetection_dev_data = load_htiee(htiee_dev_path)
    topicDetection_test_data = load_htiee(htiee_test_path)
    
    logger.info("Loaded {} topic train samples".format(len(topicDetection_train_data)))
    logger.info("Loaded {} topic dev samples".format(len(topicDetection_dev_data)))
    logger.info("Loaded {} topic test samples".format(len(topicDetection_test_data)))
    
    canonical_data_suffix = "canonical_data"
    canonical_data_root = os.path.join(root, canonical_data_suffix)
    if not os.path.isdir(canonical_data_root):
        os.mkdir(canonical_data_root)
    
    # BUILD S
    sentimentDetection_train_fout = os.path.join(canonical_data_root, "s_train.tsv")
    sentimentDetection_dev_fout = os.path.join(canonical_data_root, "s_dev.tsv")
    sentimentDetection_test_fout = os.path.join(canonical_data_root, "s_test.tsv")
    
    dump_rows(sentimentDetection_train_data, sentimentDetection_train_fout, DataFormat.PremiseOnly)
    dump_rows(sentimentDetection_dev_data, sentimentDetection_dev_fout, DataFormat.PremiseOnly)
    dump_rows(sentimentDetection_test_data, sentimentDetection_test_fout, DataFormat.PremiseOnly)
    
    logger.info("done with sentiment detection")

    # BUILD FND 
    fakeNewsDetection_train_fout = os.path.join(canonical_data_root, "fnd_train.tsv")
    fakeNewsDetection_dev_fout = os.path.join(canonical_data_root, "fnd_dev.tsv")
    fakeNewsDetection_test_fout = os.path.join(canonical_data_root, "fnd_test.tsv")
    
    dump_rows(fakeNewsDetection_train_data, fakeNewsDetection_train_fout, DataFormat.PremiseOnly)
    dump_rows(fakeNewsDetection_dev_data, fakeNewsDetection_dev_fout, DataFormat.PremiseOnly)
    dump_rows(fakeNewsDetection_test_data, fakeNewsDetection_test_fout, DataFormat.PremiseOnly)
    
    logger.info("done with fake news detection")
    
    # BUILD SQDC 
    stanceDetection_train_fout = os.path.join(canonical_data_root, "sqdc_train.tsv")
    stanceDetection_dev_fout = os.path.join(canonical_data_root, "sqdc_dev.tsv")
    stanceDetection_test_fout = os.path.join(canonical_data_root, "sqdc_test.tsv")
    
    dump_rows(stanceDetection_train_data, stanceDetection_train_fout, DataFormat.PremiseAndOneHypothesis)
    dump_rows(stanceDetection_dev_data, stanceDetection_dev_fout, DataFormat.PremiseAndOneHypothesis)
    dump_rows(stanceDetection_test_data, stanceDetection_test_fout, DataFormat.PremiseAndOneHypothesis)
    
    logger.info("done with stance detection")
    
    # BUILD HTIEE 
    topicDetection_train_fout = os.path.join(canonical_data_root, "htiee_train.tsv")
    topicDetection_dev_fout = os.path.join(canonical_data_root, "htiee_dev.tsv")
    topicDetection_test_fout = os.path.join(canonical_data_root, "htiee_test.tsv")
    
    dump_rows(topicDetection_train_data, topicDetection_train_fout, DataFormat.PremiseOnly)
    dump_rows(topicDetection_dev_data, topicDetection_dev_fout, DataFormat.PremiseOnly)
    dump_rows(topicDetection_test_data, topicDetection_test_fout, DataFormat.PremiseOnly)
    
    logger.info("done with topic detection")


if __name__ == "__main__":
    args = parse_args()
    main(args)
