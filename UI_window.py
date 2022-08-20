import os
import sys
import warnings

import cv2
import torch
from PIL import Image, ImageQt
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QWidget, QLabel, QHBoxLayout, QVBoxLayout, \
    QApplication, QLineEdit, QScrollArea, QFileDialog, QPushButton, QTextEdit

from UI_utils import loadBar
from UI_utils.loadBar import Runthread
from config import cfg
from utils import transforms


torch.multiprocessing.set_sharing_strategy('file_system')
warnings.filterwarnings("ignore")
# from UI_utils import

# Please have to change env path, this path is for Linux/conda.
# https://stackoverflow.com/questions/63829991/qt-qpa-plugin-could-not-load-the-qt-platform-plugin-xcb-in-even-though-it
os.environ[
    'QT_QPA_PLATFORM_PLUGIN_PATH'] = '/home/ANYCOLOR2434/.conda/envs/DMT/lib/python3.7/site-packages/PyQt5/Qt5/plugins/xcbglintegrations/libqxcb-glx-integration.so'


class interface(QWidget):
    printSignal = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.k = None
        self.query_path = None
        self.printSignal.connect(self.ui_result)
        self.ProgressBar = loadBar.ProgressBar()
        # 滾動條
        self.label = QLabel("Result image")
        # self.topFiller = QWidget()
        self.label.resize(1000, 500)
        self.scroll = QScrollArea()
        self.scroll.setWidget(self.label)

        self.vbox = QVBoxLayout()
        self.vbox.setContentsMargins(20, 20, 20, 20)
        # file_button
        self.file_button = QPushButton()
        self.file_button.setObjectName("file_button")
        self.file_button.setText("Open query image")
        self.show_file_path = QTextEdit()
        self.show_file_path.setFixedHeight(70)
        # self.show_file_path.setGeometry(QtCore.QRect(280, 130, 451, 81))
        self.show_file_path.setObjectName("show_file_path")

        self.label_q = QLabel("Query image")
        self.topK_button = QPushButton()
        self.topK_button.setObjectName("topK_button")
        self.topK_button.setText("set and run")

        self.h = QHBoxLayout()
        self.v2 = QVBoxLayout()
        self.v2.addWidget(self.file_button)
        self.v2.addWidget(self.show_file_path)
        self.v2.addWidget(self.label_q)
        self.input_TopK()
        self.setup_control()

        # self.vbox.addWidget(self.label)
        self.vbox.addWidget(self.scroll)

        self.h.addLayout(self.v2)
        self.h.addLayout(self.vbox)
        self.h.addStretch(1)

        self.setLayout(self.h)

        self.thread = Runthread(self)

    def ui_result(self, cur_result):
        topK_result = cur_result[:self.k]
        bg = Image.new('RGBA', (150 * 5, 150 * ((self.k // 5)+1)), '#FFFFFF')
        for i, result in enumerate(topK_result):
            img_path = result[0]
            img_dist = result[1]
            im = cv2.imread(img_path)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = cv2.resize(im, (150, 150))
            cv2.putText(im, img_path.split('/')[-1], (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 128, 0), 1)
            cv2.putText(im, round(img_dist, 4).astype('str'), (80, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255),
                        1)
            img = Image.fromarray(im).convert("RGBA")
            x = (i + 1 - 1) % 5
            y = (i + 1 - 1) // 5
            bg.paste(img, (x * 150, y * 150))
        self.ProgressBar.add_value(1)
        qtImage = ImageQt.ImageQt(bg)
        self.label.setPixmap(QPixmap.fromImage(qtImage))
        rect = qtImage.rect()
        self.label.resize(rect.width(), rect.height())
        self.scroll.resize(rect.width(), self.scroll.rect().height())
        self.setFixedSize(self.label_q.rect().width() + 40 + rect.width(), self.rect().height())
        self.resize(self.label_q.rect().width() + 40 + rect.width(), self.rect().height())

        # .emit(image)

    def input_TopK(self):
        # pure label
        self.top_k = QLabel("rank-k", self)
        # top_k.setGeometry(20, 20, 50, 20)
        # text box
        self.edit_topk = QLineEdit(self)
        self.edit_topk.setPlaceholderText("integer number (<11578)")
        # edit_topk.setGeometry(90, 20, 200, 20)
        self.v2.addWidget(self.top_k)
        self.v2.addWidget(self.edit_topk)
        self.v2.addWidget(self.topK_button)

        self.v2.addStretch(3)

    def show_image(self, filename):
        self.query_path = filename
        img = Image.open(filename).convert("RGBA")
        qtImage = ImageQt.ImageQt(img)
        self.label_q.setPixmap(QPixmap.fromImage(qtImage))

    def set_topK_run(self):
        self.k = int(self.edit_topk.text())
        self.ProgressBar = loadBar.ProgressBar(self, minimum=0, maximum=100, textVisible=False,
                                               objectName="BlueProgressBar")
        self.v2.addWidget(self.ProgressBar)

        self.thread.start()

    def setup_control(self):
        self.file_button.clicked.connect(self.open_file)
        self.topK_button.clicked.connect(self.set_topK_run)
        # self.ui.folder_button.clicked.connect(self.open_folder)

    def open_file(self):
        # folder = QFileDialog.getExistingDirectory(UI, '選擇資料夾', dest)
        filename, filetype = QFileDialog.getOpenFileName(self,
                                                         "Open file",
                                                         "/home/ANYCOLOR2434/AICITY2021_Track2_DMT/AIC21/veri_pose/query/",
                                                         options=QFileDialog.DontUseNativeDialog)  # start path
        self.show_image(filename)
        print(filename, filetype)
        self.show_file_path.setText(filename)

    def get_one_img(self, query_img_path):
        img = Image.open(query_img_path).convert("RGB")
        normalizer = transforms.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
        transform = transforms.Compose([transforms.RectScale(256, 256),
                                        transforms.ToTensor(),
                                        normalizer])
        img = transform(img)
        pid = int(query_img_path[-24:-20])
        camid = int(query_img_path[-18:-15])

        return {'origin': img,
                'pid': pid,
                'camid': camid,
                'trackid': -1,
                'file_name': query_img_path
                }


if __name__ == '__main__':
    root = QApplication(sys.argv)

    win = interface()
    # title
    win.setWindowTitle("ReID result")

    # show window
    win.show()
    # root cycle waiting state
    sys.exit(root.exec_())
