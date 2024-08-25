import os
import subprocess


def run_train(level=2,
              lr=3e-5,
              bs=8,
              gac=1,
              n_epochs=30,
              use_rels=True,
              use_nucsat=True,
              use_paths=True,
              use_start_end=True,
              save_eval_metric=None,
              device='cuda:0'):
    
    ending = f'lvl{level}--use_rels={use_rels}--use_nucsat={use_nucsat}--use_paths={use_paths}--use_start_end={use_start_end}_multi.json'
    train_file = "datasets/deberta_propaganda_full/" + [fn for fn in os.listdir("datasets/deberta_propaganda_full/") if
                                         fn.startswith('train') and fn.endswith(ending)][0]
    dev_file = "datasets/deberta_propaganda_full/" + [fn for fn in os.listdir("datasets/deberta_propaganda_full/") if
                                         fn.startswith('dev') and fn.endswith(ending)][0]
    extra_feature_size = int(train_file.split('/')[2].split('=')[1].split('_')[0])
    
    output_dir = f"checkpoint/deberta_ner_binary_noo_test_lr{lr}-{bs*gac}-{n_epochs}ep_w100_0lin__" + ending.replace('.json', '')
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = device.replace('cuda:', '')
    
    subprocess.call(
        ["python",
         "ner_deberta_multi/run_ner_no_trainer_custom.py",
         "--model_name_or_path", "microsoft/deberta-v3-base",
         "--train_file", train_file,
         "--validation_file", dev_file,
         "--text_column_name", 'token',
         "--label_column_name", 'label',
         "--feature_column_name", 'feature',
         "--max_length", "256",
         "--extra_feature_size", str(extra_feature_size),
         "--pad_to_max_length",
         "--per_device_train_batch_size", str(bs),
         "--per_device_eval_batch_size", str(bs),
         "--gradient_accumulation_steps", str(gac),
         "--learning_rate", str(lr),
         "--num_train_epochs", str(n_epochs),
         "--save_eval_metric", save_eval_metric,
         "--checkpointing_steps", "100000",
         "--output_dir", output_dir,
         "--with_tracking"]
    )
