# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, Optional

from swift.llm import DatasetMeta, ResponsePreprocessor, load_dataset, register_dataset


class GSM8KPreprocessor(ResponsePreprocessor):
    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        answer = row['response']
        ground_truth = row['response'].split("####")[-1].strip()
        row.update({'answer': answer, 'ground_truth': ground_truth})
        return super().preprocess(row)


register_dataset(
    DatasetMeta(
        ms_dataset_id='modelscope/gsm8k',
        subsets=['main'],
        split=['train'],
        tags=['qa', 'math'],
        preprocess_func=GSM8KPreprocessor(),
    ), 
    exist_ok=True
)

if __name__ == '__main__':
    dataset = load_dataset(['/apdcephfs_cq10/share_1150325/ztjiaweixu/huggingface/gsm8k'])[0]
    print(f'dataset: {dataset}')
    print(f'dataset[0]: {dataset[0]}')
