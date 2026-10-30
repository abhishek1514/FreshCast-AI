# FreshRetailNet-50K data attribution

The real-data mode downloads FreshRetailNet-50K from Dingdong Inc. through Hugging Face:
https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K

- Dataset license: Creative Commons Attribution 4.0 International (CC BY 4.0)
- License text: https://creativecommons.org/licenses/by/4.0/legalcode
- Paper: https://arxiv.org/abs/2505.16319
- Dataset developer: Dingdong Inc.

This source ZIP contains the adapter/downloader, not the third-party raw dataset itself.
The downloaded dataset is used by `scripts/prepare_real_data.py` and is stored locally under
`data/raw/`. If you redistribute the dataset or derivatives, review the license and retain
appropriate attribution and citation.

The dataset's `sale_amount` is a normalized sales measure, not literal physical cases. It
also does not contain item-level supplier costs, prices, shelf lives, or lead times. The
project's inventory cost inputs are illustrative assumptions and must not be represented as
actual financial data.
