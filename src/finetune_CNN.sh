# run fine-tuning
DATA_DIR=../data/cnndm_data
OUTPUT_DIR=../model/cnn_finetuned/
MODEL_RECOVER_PATH=../model/unilmv1-large-cased.bin
export PYTORCH_PRETRAINED_BERT_CACHE=../model/bert-cased-pretrained-cache
export CUDA_VISIBLE_DEVICES=0,1
python biunilm/run_seq2seq.py --do_train --amp --num_workers 0 \
  --bert_model bert-large-cased --new_segment_ids --tokenized_input \
  --data_dir ${DATA_DIR} \
  --output_dir ${OUTPUT_DIR}/bert_save \
  --log_dir ${OUTPUT_DIR}/bert_log \
  --model_recover_path ${MODEL_RECOVER_PATH} \
  --max_seq_length 768 --max_position_embeddings 768 \
  --trunc_seg a --always_truncate_tail \
  --max_len_a 568 --max_len_b 200 \
  --mask_prob 0.7 --max_pred 140 \
  --train_batch_size 48 --gradient_accumulation_steps 2 \
  --learning_rate 0.00003 --warmup_proportion 0.1 --label_smoothing 0.1 \
  --num_train_epochs 30  
