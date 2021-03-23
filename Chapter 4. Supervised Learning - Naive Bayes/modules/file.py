import pandas as pd

class CPandasFile:
    """ Class dùng để đọc ghi file bằng package pandas
    """
    def readCsv(self, path, usecols=None):
        """ Đọc file csv

        Args:
            path ([str]): đường dẫn đến file cần đọc
            usecols ([str | list of str]): tên các column mà cần sử dụng

        Returns:
            [type]: [description]
        """
        if type(usecols) == str:
            usecols = [usecols]
        
        return pd.read_csv(path, usecols=usecols)
    
    def readExcel(self, path):
        return pd.read_excel(path)