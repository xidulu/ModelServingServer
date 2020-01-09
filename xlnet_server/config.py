import torch
checkpoint_dir = 'outputs/'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = {
        'data_dir': 'data/',
        'model_type':  'xlnet',
        'model_name': 'xlnet-base-cased',
        'task_name': 'binary',
        'output_dir': 'outputs/',
        'cache_dir': 'cache/',
        'do_train': False,
        'do_eval': True,
        'do_infer': True,
        'fp16': False,
        'fp16_opt_level': 'O1',
        'max_seq_length': 256,
        'output_mode': 'classification',
        'train_batch_size': 512,
        'eval_batch_size': 64,
        'infer_batch_size': 64,

        'gradient_accumulation_steps': 1,
        'num_train_epochs': 1,
        'weight_decay': 0,
        'learning_rate': 4e-5,
        'adam_epsilon': 1e-8,
        'warmup_steps': 0,
        'max_grad_norm': 1.0,

        'logging_steps': 50,
        'evaluate_during_training': False,
        'save_steps': 2000,
        'eval_all_checkpoints': True,

        'overwrite_output_dir': False,
        'reprocess_input_data': True,
        'notes': 'Using Yelp Reviews dataset'
    }