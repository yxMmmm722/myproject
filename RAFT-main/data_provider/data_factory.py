from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom
from torch.utils.data import DataLoader
import numpy as np
import torch

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'etth1': Dataset_ETT_hour,
    'etth2': Dataset_ETT_hour,
    'ettm1': Dataset_ETT_minute,
    'ettm2': Dataset_ETT_minute,
    'electricity': Dataset_Custom,
    'exchange': Dataset_Custom,
    'exchange_rate': Dataset_Custom,
    'illness': Dataset_Custom,
    'traffic': Dataset_Custom,
    'weather': Dataset_Custom,
    'Electricity': Dataset_Custom,
    'Exchange': Dataset_Custom,
    'Illness': Dataset_Custom,
    'Traffic': Dataset_Custom,
    'Weather': Dataset_Custom,
    'custom': Dataset_Custom,
}


def _resolve_data_key(data_name):
    if data_name in data_dict:
        return data_name
    lowered = data_name.lower()
    if lowered in data_dict:
        return lowered
    raise KeyError(f'Unknown dataset "{data_name}". Available: {sorted(data_dict.keys())}')


def _safe_float_array(value):
    arr = np.asarray(value)
    if arr.dtype == np.object_:
        try:
            arr = arr.astype(np.float32)
        except (TypeError, ValueError):
            arr = np.asarray(arr.tolist(), dtype=np.float32)
    else:
        arr = arr.astype(np.float32, copy=False)
    return arr


def hcar_collate_fn(batch):
    norm_batch = []
    for sample in batch:
        if len(sample) >= 6:
            norm_batch.append(sample[:6])
        elif len(sample) == 5:
            norm_batch.append((*sample, {}))
        else:
            raise ValueError(f"Unexpected sample size in batch: {len(sample)}")

    index, seq_x, seq_y, seq_x_mark, seq_y_mark, meta_data = zip(*norm_batch)

    index = torch.tensor(index, dtype=torch.long)
    seq_x = torch.from_numpy(np.stack([_safe_float_array(x) for x in seq_x], axis=0))
    seq_y = torch.from_numpy(np.stack([_safe_float_array(y) for y in seq_y], axis=0))
    seq_x_mark = torch.from_numpy(np.stack([_safe_float_array(xm) for xm in seq_x_mark], axis=0))
    seq_y_mark = torch.from_numpy(np.stack([_safe_float_array(ym) for ym in seq_y_mark], axis=0))

    local_state_batch = []

    for md in meta_data:
        md = md or {}
        local_state = np.asarray(md.get("local_state_by_period", np.zeros((3, 4), dtype=np.float32)))
        if local_state.ndim == 1:
            local_state = local_state.reshape(1, -1)
        local_state = _safe_float_array(local_state)
        if local_state.shape[0] < 3:
            local_state = np.concatenate(
                [local_state, np.zeros((3 - local_state.shape[0], local_state.shape[1]), dtype=np.float32)],
                axis=0,
            )
        if local_state.shape[1] < 4:
            local_state = np.concatenate(
                [local_state, np.zeros((local_state.shape[0], 4 - local_state.shape[1]), dtype=np.float32)],
                axis=1,
            )
        local_state_batch.append(local_state[:3, :4])

    meta_batch = {
        "local_state_by_period": torch.from_numpy(np.stack(local_state_batch, axis=0).astype(np.float32))
    }

    return index, seq_x, seq_y, seq_x_mark, seq_y_mark, meta_batch


def data_provider(args, flag):
    Data = data_dict[_resolve_data_key(args.data)]
    timeenc = 0 if args.embed != 'timeF' else 1

    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = False
    batch_size = args.batch_size
    freq = args.freq

    # We currently supports only forecasting

    data_set = Data(
        args = args,
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        seasonal_patterns=None # We do not use this option.
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        collate_fn=hcar_collate_fn)
    return data_set, data_loader
