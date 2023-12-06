import logging
import os.path
import sys
from gensim.corpora import WikiCorpus
import time

# 获取当前时间，用于后续计算程序运行的时间
begin = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

if __name__ == '__main__':
    # 设置 logger
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    # 定义输入和输出文件的路径
    inp, outp = "/Users/lihaji/Desktop/wiki_corpus2txt/wiki_corpus/zhwiki-articles-20230820.xml.bz2", "./processed_corpus/wiki_data_zh.txt"

    # 定义分隔符，这里使用空格
    space = ' '

    # 初始化计数器，用于统计处理的文章数量
    i = 0

    # 打开输出文件，写入处理后的文本内容
    output = open(outp, 'w', encoding='utf-8')

    # 创建 WikiCorpus 对象，用于处理维基百科的 XML 数据
    wiki = WikiCorpus(inp, dictionary={})

    # 遍历处理后的文本数据
    for text in wiki.get_texts():

        # 将每篇文章的词语用空格连接起来，并且添加换行符
        s = space.join(text)+"\n"
        # 将 s 写入输出文件
        output.write(s)

        i = i+1
        # 每处理 10000 篇文章打一次 log
        if (i % 10000 == 0):
            logger.info("Saved "+str(i) + " articles")

    # 关闭输出文件
    output.close()
    # 记录处理完成的日志信息
    logger.info("Finished Saved " + str(i) + " articles")

    end = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    # 输出程序开始和结束的时间
    print("begin", begin)
    print("end  ", end)
