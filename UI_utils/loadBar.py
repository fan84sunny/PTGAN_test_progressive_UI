#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on 2018年1月30日
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file: SimpleStyle
@description:
"""

import argparse
import os
import sys

from PyQt5.QtCore import QTimer, QThread
from PyQt5.QtWidgets import QWidget, QApplication, QVBoxLayout, QProgressBar

from config import cfg

# except ImportError:
#     from PySide2.QtCore import QTimer
#     from PySide2.QtWidgets import QWidget, QApplication, QVBoxLayout, QProgressBar
from gan.model import Model
from process_for_test_CCK import do_inference, get_pose, do_inference_reid
from utils.logger import setup_logger

StyleSheet = """
/*设置红色进度条*/
#RedProgressBar {
    text-align: center; /*进度值居中*/
}
#RedProgressBar::chunk {
    background-color: #F44336;
}
#GreenProgressBar {
    min-height: 12px;
    max-height: 12px;
    border-radius: 6px;
}
#GreenProgressBar::chunk {
    border-radius: 6px;
    background-color: #009688;
}
#BlueProgressBar {
    border: 2px solid #2196F3;/*边框以及边框颜色*/
    border-radius: 5px;
    background-color: #E0E0E0;
}
#BlueProgressBar::chunk {
    background-color: #2196F3;
    width: 10px; /*区块宽度*/
    margin: 0.5px;
}
"""


class ProgressBar(QProgressBar):

    def __init__(self, *args, **kwargs):
        super(ProgressBar, self).__init__(*args, **kwargs)
        self.setValue(0)
        if self.minimum() != self.maximum():
            self.timer = QTimer(self, timeout=self.onTimeout)
            self.timer.start(80 * 10)

    def add_value(self, amount):
        self.setValue(self.value() + amount)

    def onTimeout(self):
        if self.value() >= 100:
            self.timer.stop()
            self.timer.deleteLater()
            del self.timer
            return
        if self.value() == 20:
            self.setValue(20)
        else:
            self.setValue(self.value() + 1)


class Window(QWidget):

    def __init__(self, *args, **kwargs):
        super(Window, self).__init__(*args, **kwargs)
        self.resize(800, 600)
        layout = QVBoxLayout(self)
        # layout.addWidget(
        #     ProgressBar(self, minimum=0, maximum=100, objectName="RedProgressBar"))
        # layout.addWidget(  # 繁忙状态
        #     ProgressBar(self, minimum=0, maximum=0, objectName="RedProgressBar"))
        #
        # layout.addWidget(
        #     ProgressBar(self, minimum=0, maximum=100, textVisible=False,
        #                 objectName="GreenProgressBar"))
        # layout.addWidget(
        #     ProgressBar(self, minimum=0, maximum=0, textVisible=False,
        #                 objectName="GreenProgressBar"))
        #
        layout.addWidget(
            ProgressBar(self, minimum=0, maximum=100, textVisible=False,
                        objectName="BlueProgressBar"))
        layout.addWidget(
            ProgressBar(self, minimum=0, maximum=0, textVisible=False,
                        objectName="BlueProgressBar"))


class Runthread(QThread):
    def __init__(self, mainWin):
        super(Runthread, self).__init__()
        self.mainWin = mainWin

    def run(self):
        self.startInference()

    def startInference(self):
        parser = argparse.ArgumentParser(description="ReID Baseline Training")
        parser.add_argument(
            "--config_file", default="", help="path to config file", type=str
        )
        parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                            nargs=argparse.REMAINDER)

        args = parser.parse_args()

        if args.config_file != "":
            cfg.merge_from_file(args.config_file)
        cfg.merge_from_list(args.opts)
        cfg.freeze()
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
        output_dir = cfg.OUTPUT_DIR
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        logger = setup_logger("reid_baseline", output_dir, if_train=False)
        logger.info(args)
        if args.config_file != "":
            logger.info("Loaded configuration file {}".format(args.config_file))
            with open(args.config_file, 'r') as cf:
                config_str = "\n" + cf.read()
                logger.info(config_str)
        logger.info("Running with config:\n{}".format(cfg))
        query_data = self.mainWin.get_one_img(self.mainWin.query_path)
        # model = Model("cuda")
        # model.reset_model_status()
        # model.eval()
        query_poseid = get_pose(query_data)
        query_feats = do_inference_reid(cfg, query_data)
        do_inference(cfg, query_data, query_feats=query_feats, query_poseid=query_poseid, resultWindow=self.mainWin)
        self.mainWin.ProgressBar.add_value(100)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(StyleSheet)
    w = Window()
    w.show()
    sys.exit(app.exec_())
