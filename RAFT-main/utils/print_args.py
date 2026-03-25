def print_args(args):
    print("\033[1m" + "Basic Config" + "\033[0m")
    print(f'  {"Task Name:":<20}{args.task_name:<20}{"Is Training:":<20}{args.is_training:<20}')
    print(f'  {"Model ID:":<20}{args.model_id:<20}{"Model:":<20}{args.model:<20}')
    print()

    print("\033[1m" + "Data Loader" + "\033[0m")
    print(f'  {"Data:":<20}{args.data:<20}{"Root Path:":<20}{args.root_path:<20}')
    print(f'  {"Data Path:":<20}{args.data_path:<20}{"Features:":<20}{args.features:<20}')
    print(f'  {"Target:":<20}{args.target:<20}{"Freq:":<20}{args.freq:<20}')
    print(f'  {"Checkpoints:":<20}{args.checkpoints:<20}')
    print()

    if args.task_name in ['long_term_forecast', 'short_term_forecast']:
        print("\033[1m" + "Forecasting Task" + "\033[0m")
        print(f'  {"Seq Len:":<20}{args.seq_len:<20}{"Label Len:":<20}{args.label_len:<20}')
        print(f'  {"Pred Len:":<20}{args.pred_len:<20}{"Seasonal Patterns:":<20}{args.seasonal_patterns:<20}')
        print(f'  {"Inverse:":<20}{args.inverse:<20}')
        print()

    if args.task_name == 'imputation':
        print("\033[1m" + "Imputation Task" + "\033[0m")
        print(f'  {"Mask Rate:":<20}{args.mask_rate:<20}')
        print()

    if args.task_name == 'anomaly_detection':
        print("\033[1m" + "Anomaly Detection Task" + "\033[0m")
        print(f'  {"Anomaly Ratio:":<20}{args.anomaly_ratio:<20}')
        print()

    print("\033[1m" + "Model Parameters" + "\033[0m")
    print(f'  {"Top k:":<20}{args.top_k:<20}{"Num Kernels:":<20}{args.num_kernels:<20}')
    print(f'  {"Enc In:":<20}{args.enc_in:<20}{"Dec In:":<20}{args.dec_in:<20}')
    print(f'  {"C Out:":<20}{args.c_out:<20}{"d model:":<20}{args.d_model:<20}')
    print(f'  {"n heads:":<20}{args.n_heads:<20}{"e layers:":<20}{args.e_layers:<20}')
    print(f'  {"d layers:":<20}{args.d_layers:<20}{"d FF:":<20}{args.d_ff:<20}')
    print(f'  {"Moving Avg:":<20}{args.moving_avg:<20}{"Factor:":<20}{args.factor:<20}')
    print(f'  {"Distil:":<20}{args.distil:<20}{"Dropout:":<20}{args.dropout:<20}')
    print(f'  {"Embed:":<20}{args.embed:<20}{"Activation:":<20}{args.activation:<20}')
    print(f'  {"Output Attention:":<20}{args.output_attention:<20}')
    if hasattr(args, "retrieval_alpha"):
        print(f'  {"Retr Alpha:":<20}{args.retrieval_alpha:<20}{"Coarse K:":<20}{args.retrieval_coarse_k:<20}')
    if hasattr(args, "meta_only_retrieval"):
        print(f'  {"Meta-Only Retr:":<20}{args.meta_only_retrieval:<20}{"Topm:":<20}{args.topm:<20}')
    if hasattr(args, "compare_retrieval_topm"):
        print(f'  {"Cmp Retr Topm:":<20}{args.compare_retrieval_topm:<20}')
    elif hasattr(args, "compare_retrieval_topk"):
        print(f'  {"Cmp Retr Topm:":<20}{args.compare_retrieval_topk:<20}')
    if hasattr(args, "context_dim"):
        print(f'  {"Context Dim:":<20}{args.context_dim:<20}{"Learnable Alpha:":<20}{args.learnable_alpha:<20}')
    if hasattr(args, "freeze_context_encoder"):
        print(f'  {"Freeze Context:":<20}{args.freeze_context_encoder:<20}{"Refresh Ctx Pool:":<20}{args.refresh_context_each_epoch:<20}')
    if hasattr(args, "online_retrieval"):
        print(f'  {"Online Retrieval:":<20}{args.online_retrieval:<20}')
    if hasattr(args, "text_encoder_name"):
        print(f'  {"Text Encoder:":<20}{args.text_encoder_name:<20}{"Text Proj Dim:":<20}{args.text_proj_dim:<20}')
    if hasattr(args, "require_text_encoder"):
        print(f'  {"Require Text Enc:":<20}{args.require_text_encoder:<20}')
    if hasattr(args, "save_meta_texts"):
        print(f'  {"Save Meta Texts:":<20}{args.save_meta_texts:<20}{"Text Dump Dir:":<20}{args.meta_text_dump_dir:<20}')
    if hasattr(args, "meta_text_max_samples"):
        print(f'  {"Dump Max Samples:":<20}{args.meta_text_max_samples:<20}')
    print()

    print("\033[1m" + "Run Parameters" + "\033[0m")
    print(f'  {"Num Workers:":<20}{args.num_workers:<20}{"Itr:":<20}{args.itr:<20}')
    print(f'  {"Train Epochs:":<20}{args.train_epochs:<20}{"Batch Size:":<20}{args.batch_size:<20}')
    print(f'  {"Patience:":<20}{args.patience:<20}{"Learning Rate:":<20}{args.learning_rate:<20}')
    print(f'  {"Des:":<20}{args.des:<20}{"Loss:":<20}{args.loss:<20}')
    print(f'  {"Lradj:":<20}{args.lradj:<20}{"Use Amp:":<20}{args.use_amp:<20}')
    print()

    print("\033[1m" + "GPU" + "\033[0m")
    print(f'  {"Use GPU:":<20}{args.use_gpu:<20}{"GPU:":<20}{args.gpu:<20}')
    print(f'  {"Use Multi GPU:":<20}{args.use_multi_gpu:<20}{"Devices:":<20}{args.devices:<20}')
    print()

    print("\033[1m" + "De-stationary Projector Params" + "\033[0m")
    p_hidden_dims_str = ', '.join(map(str, args.p_hidden_dims))
    print(f'  {"P Hidden Dims:":<20}{p_hidden_dims_str:<20}{"P Hidden Layers:":<20}{args.p_hidden_layers:<20}') 
    print()
