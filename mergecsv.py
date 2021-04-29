"""
    複数ページに分かれている(ドロ数20+とその次のドロ数) fgosccnt.py の出力を、
    周回数毎のデータに統合して書き出す
"""

import argparse
from pathlib import Path
from csv2graph import make_df
import pandas


VERSION = '20200906'
PROGNAME = 'merge csv'
BASE_DIR = Path(__file__).resolve().parent

def write_csv(csv_path, total_row=True):
    """
        複数ページに分かれている(ドロ数20+とその次のドロ数) fgosccnt.py の出力を、
        周回数毎のデータに統合して書き出す

        csv_path (str or pathlib.xxPath):
            fgosccntで出力したcsvファイルのパスを指定する

        total_row (bool):
            totalをFalseにすると csv2counter.py との互換性がなくなる
            合計の行が必要ない場合はFalseにする
    """
    if args.remove_total_row:
        total_row = False
    df = make_df(csv_path, total_row=total_row)
    if type(df) == pandas.core.frame.DataFrame:
        csv_path = Path(csv_path)
        if args.output_path:
            output_path = Path(args.output_path)
        elif args.output_folder:
            output_path = Path(args.output_folder) / csv_path.name
        else:
            output_path = BASE_DIR / csv_path.name

        if not output_path.parent.is_dir():
            output_path.parent.mkdir(parents=True)

        df.to_csv(
            output_path,
            encoding=args.character_code,
            index=False
        )
        print('\rcsv書き出し完了  ')

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='20+と2ページ目以降に分かれているCSVのデータを1周毎のデータに変換する')
    parser.add_argument('filenames', help='入力ファイル', nargs='*')
    parser.add_argument('--version', action='version', version=PROGNAME + " " + VERSION)
    parser.add_argument('-c', '--character_code', help='出力時の文字コード', default='shift_jis')
    parser.add_argument('-o', '--output_path', help='ファイル名までの出力先パスを指定')
    parser.add_argument('-f', '--output_folder', help='出力先フォルダ')
    parser.add_argument('-r', '--remove_total_row', help='合計行を削除する', action='store_true')
    
    args = parser.parse_args()

    if args.filenames:
        csv_paths = args.filenames

    # 指定がなければテストファイルを実行
    else:
        csv_paths = [
            BASE_DIR / 'test_csv_files' / 'Heaven\'s_hotel.csv',
            BASE_DIR / 'test_csv_files' / 'Silent_garden_B.csv'
        ]
    
    if type(csv_paths) == list:
        for csv_path in csv_paths:
            write_csv(csv_path)
    else:
        write_csv(csv_paths)
    

