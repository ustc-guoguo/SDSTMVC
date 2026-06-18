
import os
import sys
import logging
from datetime import datetime



class NoNewlineStreamHandler(logging.StreamHandler):
    """不自动加换行的终端处理器。A terminal Handler that does not automaticaly add line breaks."""
    def emit(self, record):
        try:
            msg = self.format(record)
            self.stream.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


class NoNewlineFileHandler(logging.FileHandler):
    """不自动加换行的文件处理器。 A file Handler that does not automatically add line breaks."""
    def emit(self, record):
        try:
            msg = self.format(record)
            self.stream.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


class DualLogger:
    """
    :param log_name: 日志名标识符
    :param filename: 日志文件路径
    :param show_time: 是否显示时间戳
    :param time_format: 时间戳格式
    """
    def __init__(self,
                 log_name=None,
                 root='logs',
                 filename=None,
                 show_time=True,
                 time_format="%Y-%m-%d %H:%M:%S"):
        self.log_name = log_name
        self.show_time = show_time
        self.time_format = time_format
        self.new_line = True  # 记录是否处于新行开头

        if self.log_name is None:
            self.logger = logging.getLogger()
        else:
            self.logger = logging.getLogger(self.log_name)
        self.logger.propagate = False  # 禁止日志冒泡, 只打印自己的日志
        self.logger.setLevel(logging.DEBUG)

        if not self.logger.handlers:
            # 只显示 message，时间戳我们自己加
            formatter = logging.Formatter('%(message)s')

            # 终端输出
            ch = NoNewlineStreamHandler(sys.stdout)
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

            # 文件输出（追加模式）
            if filename is not None:
                if not os.path.exists(root):
                    os.makedirs(root)
                fh = NoNewlineFileHandler(os.path.join(root, filename),
                                          mode="a",
                                          encoding="utf-8")
                fh.setFormatter(formatter)
                self.logger.addHandler(fh)

    def _add_timestamp_if_needed(self, message):
        if self.show_time and self.new_line:
            ts = datetime.now().strftime(self.time_format)
            return f"[{ts}] {message}"
        return message

    def write(self, message="", end="\n"):
        msg = str(message)
        msg = self._add_timestamp_if_needed(msg)
        self.logger.info(msg + end)
        self.new_line = (end == "\n")  # 如果结束符是换行，下一次输出才加时间戳

    def flush(self):
        """兼容 print 的 file 接口"""
        pass


# ================== Examples ==================
if __name__ == "__main__":
    # print to terminal and save as ./logs/log.txt file
    logger = DualLogger(root="./logs", filename="log.txt", show_time=True)

    # print to terminal but do not save as file
    # log = DualLogger(root="./logs", show_time=True)

    logger.write("Begin ...")                       # 自动带时间戳 + 换行
    logger.write("Handling ... ", end="")           # 不换行
    logger.write("Done.", end="\n")                 # 接上面一行
    logger.write("Muti-line log: first line, \nnext line")        # 多行也会带时间戳
    logger.write(" > 1", end='')
    logger.write(" > 2", end='')
    logger.write(" > 3", end='')
    logger.write(" > 4", end='')
    logger.write(" > 5")
    logger.write("END")

