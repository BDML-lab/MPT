
## 2025-11-29-20:53:57 


|  Attribute   |   Value   |
| :-------------: | :-----------: |
|  adaptor_only  |   True    |
|  batch_size  |   512    |
|  benchmark  |   False    |
|  BEST_FILENAME  |   best.pt    |
|  beta1  |   0.9    |
|  beta2  |   0.999    |
|  CHECKPOINT_FILENAME  |   checkpoint.tar    |
|  CHECKPOINT_FREQ  |   1    |
|  CHECKPOINT_MODULES  |   ['model', 'optimizer', 'lr_scheduler']    |
|  CHECKPOINT_PATH  |   ./infos/yelp-tokens/Yelp2018_10100_LOU/1    |
|  ckpt_epoch  |   4000    |
|  config  |   configs/finetune.yaml    |
|  dataset  |   Yelp2018_10100_LOU    |
|  DATA_DIR  |   data    |
|  ddp_backend  |   nccl    |
|  description  |   yelp-tokens    |
|  device  |   cuda:1    |
|  dropout_rate  |   0.2    |
|  early_stop_patience  |   1e+23    |
|  epochs  |   200    |
|  eval_freq  |   5    |
|  eval_test  |   False    |
|  eval_valid  |   True    |
|  gradient_accumulation_steps  |   1    |
|  hidden_size  |   256    |
|  id  |   1129205351    |
|  log2console  |   True    |
|  log2file  |   True    |
|  LOG_PATH  |   ./logs/yelp-tokens/Yelp2018_10100_LOU/1129205351    |
|  lora_alpha  |   16    |
|  lora_dropout  |   0.1    |
|  lora_rank  |   8    |
|  lr  |   0.001    |
|  maxlen  |   50    |
|  momentum  |   0.9    |
|  monitors  |   ['LOSS', 'HitRate@1', 'HitRate@10', 'HitRate@20', 'NDCG@10', 'NDCG@20']    |
|  MONITOR_BEST_FILENAME  |   best.pkl    |
|  MONITOR_FILENAME  |   monitors.pkl    |
|  nesterov  |   False    |
|  num_attention_heads  |   2    |
|  num_hidden_layers  |   4    |
|  num_workers  |   4    |
|  optimizer  |   AdamW    |
|  path  |   models/MPT/Amazon2014Beauty_550_LOU/1124131243    |
|  ranking  |   full    |
|  resume  |   False    |
|  retain_seen  |   False    |
|  root  |   ../data    |
|  SAVED_FILENAME  |   model.pt    |
|  seed  |   2    |
|  sem_feat_file  |   sentence-t5-xl_item_name_categories_city.pkl    |
|  SUMMARY_DIR  |   summary    |
|  SUMMARY_FILENAME  |   SUMMARY.md    |
|  T  |   0.07    |
|  tasktag  |   TaskTags.NEXTITEM    |
|  weight_decay  |   0.1    |
|  which4best  |   NDCG@10    |
