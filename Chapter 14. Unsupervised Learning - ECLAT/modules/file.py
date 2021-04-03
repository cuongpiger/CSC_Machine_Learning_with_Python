import pandas as pd
import pickle
import matplotlib.pyplot as plt
import imageio


class CPandasFile:
    """ Class dùng để đọc ghi file bằng package pandas
    """

    def readCsv(self, path, usecols=None, index_col=None):
        """ Đọc file csv

        Args:
            path ([str]): đường dẫn đến file cần đọc
            usecols ([str | list of str]): tên các column mà cần sử dụng

        Returns:
            [type]: [description]
        """
        if type(usecols) == str:
            usecols = [usecols]

        return pd.read_csv(path, usecols=usecols, index_col=index_col)

    def writeCsv(self, df, path):
        try:
            df.to_csv(path)
        except:
            print('Error!')
            return

        print('Success!')

    def readExcel(self, path):
        return pd.read_excel(path)


class CPickleFile:
    def write(self, path, model, option='overwrite'):
        if option == 'overwrite':  # ghi đè
            option = 'wb'

        try:
            with open(path, option) as file:
                pickle.dump(model, file)

            print('Success!')
        except:
            print('Error!')

    def read(self, path):
        content = None

        try:
            with open(path, 'rb') as file:
                content = pickle.load(file)

        except:
            print('Error!')

            return None

        print('Success!')
        return content


class CImage:
    def readImage(self, path):
        photo_data = imageio.imread(path)
        plt.figure(figsize=(20, 20))
        plt.imshow(photo_data)
