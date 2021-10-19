# patent-kg-tf
## patent-kg

## 执行
按照requirements安装环境，推荐conda/venv  

直接执行extract.py文件  

其中argparse已经给定default值，  

调整input_filename和output_filename至个人文件位置  

同样也可以
```
python extract.py examples/example.json examples/example_result.json  --use_cuda true

```

## 注意
tf版本为1.4  

未加入batch运算 （慢）

最后的筛选adverb_list, 和 confidence设置 不是最终版本

