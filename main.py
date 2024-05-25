import os, sys
import traceback

import pandas as pd
from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow

root = os.path.dirname(os.path.abspath(__file__))
MainUI = uic.loadUiType(os.path.join(root, 'ui/main.ui'))[0]


class Main(QMainWindow, MainUI):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        try:
            super().__init()
            self.setupUi(self)

        except Exception as e:
            print(e)
            print(traceback.format_exc())

