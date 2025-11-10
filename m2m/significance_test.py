'''
Test the result significance between REMI-z and REMI+ model.
'''

import os
import sys
dirof = os.path.dirname
sys.path.insert(0, dirof(dirof(__file__)))
from scipy.stats import wilcoxon
from utils_common.utils import read_json


def main():
    test_significance()


def procedures():
    test_significance()


def test_significance():
    '''
    Test the significance with 
    '''
    from scipy.stats import wilcoxon
    remi_z_res = read_json('/data2/longshen/musecoco_data/results_new/band_obj/ours_no_pt/ep5_lr1e-4_linear/lightning_logs/version_0/remi_z_no_pt.json')
    remi_p_res = read_json('/data2/longshen/musecoco_data/results_new/band_obj/remi_plus/ep5_lr1e-4_linear/lightning_logs/version_2/remi_plus_no_pt.json')

    metric_names = remi_z_res.keys()
    for metric_name in metric_names:
        t1 = remi_z_res[metric_name]
        t2 = remi_p_res[metric_name]
        res = wilcoxon(t1, t2)
        print(f'{metric_name}: {res.pvalue:.4f} {res.pvalue < 0.05} {res.pvalue < 0.01} {res.pvalue < 0.001}')


if __name__ == "__main__":
    main()