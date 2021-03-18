"""
    Module dùng để hỗ trợ việc đọc, ghi các dạng file khác nhau.
"""

import pandas as pd
import xlrd

def readExcel(path, encoding_override='utf-8'):
    wb = xlrd.open_workbook(path, encoding_override=encoding_override)
    return pd.read_excel(wb)

def writeDfToCsv(data_df, path):
    data_df.to_csv(path)