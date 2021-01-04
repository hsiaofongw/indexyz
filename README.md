# 介绍

这是一款文本检索工具，它基于 LSA 与 SVD，中文分词模块引用了 [lancopku/PKUSeg-python](https://github.com/lancopku/PKUSeg-python) 项目．

特别鸣谢 [lancopku/PKUSeg-python](https://github.com/lancopku/PKUSeg-python) 项目！

## 使用方法

### 方式一

在该项目目录下的 data 目录下创建一个 articles 文件夹，将文章移移动进去，并且更改 data 目录下的配置文件 config.json ，置：

```
"load_articles_from_file_or_folder": "folder"
```

然后运行

```
python3 main.py --local data
```

就可以查询了．

### 方式二

往 data/articles.json 中添加文章与内容，然后直接运行

```
python3 main.py --local data
```

就可以查询了．