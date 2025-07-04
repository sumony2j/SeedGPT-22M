from modelscope.msdatasets import MsDataset
import argparse
import os

parser = argparse.ArgumentParser("Download the dataset from Modelscope")
parser.add_argument("--dataset",type=str,default="OmniData/Pile-OpenWebText2",
                    help="Define the dataset from modelscope")
parser.add_argument("--output_dir",type=str,default="./data",
                    help="Location where the dataset should be saved")
arg = parser.parse_args()

if __name__ == "__main__":
    if not os.path.exists(arg.output_dir):
        os.makedirs(arg.output_dir,exist_ok=True)
        
        MsDataset.load(arg.dataset,trust_remote_code=True,
               cache_dir=arg.output_dir)
