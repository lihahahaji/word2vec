# 维基百科语料数据处理

Project Tree
```
wiki_corpus2txt
├─ README.md
├─ processed_corpus
│  ├─ wiki_data.txt               wiki 英文语料
│  ├─ wiki_data_zh.txt            wiki 中文语料（未分词）
│  └─ wiki_data_zh_segmented.txt  wiki 中文语料（已分词）
├─ src
│  ├─ process_wiki.py             处理 xml 文件
│  └─ zh_segmentation.py          中文分词
└─ wiki_corpus                    原始语料数据
```

## 维基百科常用语料下载路径

英文语料路径：

https://dumps.wikimedia.org/enwiki/

中文语料路径：

https://dumps.wikimedia.org/zhwiki/

西班牙语料路径：

https://dumps.wikimedia.org/eswiki/latest/