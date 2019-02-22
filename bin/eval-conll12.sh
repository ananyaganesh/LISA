#!/usr/bin/env bash
#
# Usage: ./eval-conll12.sh [model_dir] (regular|dm|gold)

model_dir=$1
parse_config=$2

data_dir=$DATA_DIR/conll-2012-sdeps-filt-new

valid_file=$data_dir/conll2012-dev.txt.bio
valid_props_file=$data_dir/conll2012-dev-gold-props.txt

valid_file=$data_dir/conll2012-dev.txt.bio
valid_props_file=$data_dir/conll2012-dev-gold-props.txt
valid_parse_file=$data_dir/conll2012-dev-gold-parse.txt

# Test
test_file=$data_dir/conll2012-test.txt.bio
test_props_file=$data_dir/conll2012-test-gold-props.txt
test_parse_file=$data_dir/conll2012-test-gold-parse.txt

# D&M injected
dm_valid_file=lstm_predicted_parses/parse_preds-conll2012-lstm-dev.tsv.replaced.filt
dm_test_file=lstm_predicted_parses/parse_preds-conll2012-lstm-test.tsv.replaced

if [[ "$parse_config" == "dm" || "$parse_config" == "gold" ]]; then
    gold_attn_at_train="False"
    inject_manual_attn="True"
    if [[ "$parse_config" == "dm" ]]; then
        echo "Doing D&M eval"
        valid_file=$dm_valid_file
        test_file=$dm_test_file
    else
        echo "Doing Gold eval"
    fi
else
    echo "Doing regular eval"
    gold_attn_at_train="True"
    inject_manual_attn="True"
fi

#    --test_eval \
python $DOZAT_ROOT/network.py \
    --load \
    --test \
    --load_dir $model_dir \
    --config_file $model_dir/config.cfg \
    --gold_attn_at_train $gold_attn_at_train \
    --inject_manual_attn $inject_manual_attn \
    --valid_file $valid_file \
    --test_file $test_file \
    --gold_test_props_file $test_props_file \
    --gold_dev_parse_file $valid_parse_file \
    --gold_test_parse_file $test_parse_file \
    --eval_by_domain True
#cp $model_dir/srl_preds.tsv $model_dir/${parse_config}_dev_srl_preds.tsv
#cp $model_dir/parse_preds.tsv $model_dir/${parse_config}_dev_parse_preds.tsv
