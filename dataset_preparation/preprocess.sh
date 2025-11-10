# This file contains same content as util.py in the same directory.
# That preprocess the splitted data.

data_path='/data2/longshen/Datasets/slakh2100_flac_redux/musecoco_data/slakh_segmented_2bar_sss'
dict_path='/data2/longshen/Datasets/slakh2100_flac_redux/musecoco_data/dicts/dict_large_ft/dict.txt'

fairseq-preprocess \
  --only-source \
  --destdir ${data_path}/data-bin/ \
  --validpref ${data_path}/valid.txt  \
  --testpref ${data_path}/test.txt  \
  --trainpref ${data_path}/train.txt \
  --srcdict ${dict_path} \
  --workers 6