---
language:
- vi
tags:
- capitalization
- punctuation
- token-classification
license: cc-by-sa-4.0
datasets:
- oscar-corpus/OSCAR-2109
metrics:
- accuracy
- precision
- recall
- f1
---
# ✨ xlm-roberta-capitalization-punctuation
This a [XLM-RoBERTa](https://huggingface.co/xlm-roberta-base) model finetuned for Vietnamese punctuation restoration on the [OSCAR-2109](https://huggingface.co/datasets/oscar-corpus/OSCAR-2109) dataset. 
The model predicts the punctuation and upper-casing of plain, lower-cased text. An example use case can be ASR output. Or other cases when text has lost punctuation.
This model is intended for direct use as a punctuation restoration model for the general Vietnamese language. Alternatively, you can use this for further fine-tuning on domain-specific texts for punctuation restoration tasks.
Model restores the following punctuations -- **[. , : ? ]**
The model also restores the complex upper-casing of words like *YouTube*, *MobiFone*.

-----------------------------------------------
## 🚋 Usage

**Below is a quick way to get up and running with the model.**
1. Download files from hub  
```python
import os
import shutil
import sys
from huggingface_hub import snapshot_download
cache_dir = "./capu"
def download_files(repo_id, cache_dir=None, ignore_regex=None):
    download_dir = snapshot_download(repo_id=repo_id, cache_dir=cache_dir, ignore_regex=ignore_regex)
    if cache_dir is None or download_dir == cache_dir:
        return download_dir
    file_names = os.listdir(download_dir)
    for file_name in file_names:
        shutil.move(os.path.join(download_dir, file_name), cache_dir)
    os.rmdir(download_dir)
    return cache_dir
cache_dir = download_files(repo_id="dragonSwing/xlm-roberta-capu", cache_dir=cache_dir, ignore_regex=["*.json", "*.bin"])
sys.path.append(cache_dir)
```
2. Sample python code  
```python
import os
from gec_model import GecBERTModel
model = GecBERTModel(
    vocab_path=os.path.join(cache_dir, "vocabulary"),
    model_paths="dragonSwing/xlm-roberta-capu",
    split_chunk=True
)
model("theo đó thủ tướng dự kiến tiếp bộ trưởng nông nghiệp mỹ tom wilsack bộ trưởng thương mại mỹ gina raimondo bộ trưởng tài chính janet yellen gặp gỡ thượng nghị sĩ patrick leahy và một số nghị sĩ mỹ khác")
# Always return list of outputs.
# ['Theo đó, Thủ tướng dự kiến tiếp Bộ trưởng Nông nghiệp Mỹ Tom Wilsack, Bộ trưởng Thương mại Mỹ Gina Raimondo, Bộ trưởng Tài chính Janet Yellen, gặp gỡ Thượng nghị sĩ Patrick Leahy và một số nghị sĩ Mỹ khác.']
model("những gói cước năm g mobifone sẽ mang đến cho bạn những trải nghiệm mới lạ trên cả tuyệt vời so với mạng bốn g thì tốc độ truy cập mạng 5 g mobifone được nhận định là siêu đỉnh với mức truy cập nhanh gấp 10 lần")
# ['Những gói cước 5G MobiFone sẽ mang đến cho bạn những trải nghiệm mới lạ trên cả tuyệt vời. So với mạng 4G thì tốc độ truy cập mạng 5G MobiFone được Nhận định là siêu đỉnh với mức truy cập nhanh gấp 10 lần.']
```
**This model can work on arbitrarily large text in Vietnamese language.**

-----------------------------------------------
## 📡 Training data
Here is the number of product reviews we used for fine-tuning the model:

| Language | Number of text samples |
| --- | --- |
| Vietnamese  | 5,600,000  |

-----------------------------------------------
## 🎯 Accuracy
Below is a breakdown of the performance of the model by each label on 10,000 held-out text samples:

|  label    |   precision  |  recall | f1-score  | support |
| --- | --- | --- | --- | --- |
|     **Upper**    |   0.89       | 0.90    |  0.89     |  56497   |
|     **Complex-Upper**    |   0.93       | 0.83    |  0.88     |   480   |
|     **.**    |   0.81       | 0.84    |  0.82     | 18139   |
|    **,**    |   0.69       | 0.75    |  0.72     | 22961   |
|     **:**    |   0.76       | 0.60    |  0.67     |   1432   |
|     **?**    |   0.82       | 0.75    |  0.78     |   1730   |
|     **none**    |   0.99       | 0.99    |  0.99     |475611   |
-----------------------------------------------