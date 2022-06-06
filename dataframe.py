import io
import re
from pathlib import Path
import requests
import traceback

import pandas as pd
import numpy as np


class Data:

    def __init__(self, csv_path=None, total_row=False, qp_sum=False):
        self.df             = self.read_csv(csv_path)
        self.isNewSpec      = self.isNewSpecifications(self.df)
        self.quest_name     = self.load_quest_name(csv_path)
        self.reward_QP_name = self.load_reward_QP()
        self.QP_col_loc     = self.get_qp_col_loc(self.df)
        self.dro_col_loc    = self.df.columns.get_loc('ドロ数')
        self.df             = self.remove_total_row(self.df, total_row)
        self.df             = self.cast(self.df)
        self.raw_df         = self.df
        self.df             = self.mergeLines(self.df)
        self.df             = self.calc_qp_sum_columns(self.df, qp_sum=qp_sum)
        self.df             = self.remove_qp_sum_columns(self.df, qp_sum=qp_sum)
        self.run            = self.get_number_of_run(self.df)

    def read_csv(self, csv_path):
        try:
            df = pd.read_csv(csv_path, encoding='shift-jis')
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding='UTF-8')
        except pd.errors.EmptyDataError:
            
            # 空のファイルを指定した可能性がある
            print(f'\rファイルが空の可能性があります:\n{csv_path}')
            return None
        return df

    # アイテム数を含む仕様か確認
    #  True : ドロ数 63, 42, 21
    #  False: ドロ数 20++, 21+, 20+
    def isNewSpecifications(self, df):
        return 'アイテム数' in df.columns

    # 新仕様のデータのみ処理する
    def process_only_new_specification_data(f):
        def wrapper(self, df, **kwargs):
            if self.isNewSpec:
                return f(self, df, **kwargs)
            else:
                return df
        return wrapper

    # データ型を調節する
    def cast(self, df):
        df = df.fillna(0)

        # ドロ数の列 19+ 20+ 21+ 20++ etc が出現した行以降は全てstr型になるため、
        # str型になっている数値を数値型に変換
        if not self.isNewSpec:
            for i, row in enumerate(df[df.columns[1]]):
                if not ('+' in str(row)):
                    df.iloc[i, 1] = np.uint16(row)

        # ドロ数より後の列は numpy.float64 として読み込まれるので numpy.uint16 にキャスト
        for col in df.columns:
            if type(df[col].values[0]) == np.float64:
                df[col] = df[col].astype(np.uint16)

        return df

    # 報酬QPのカラム名を取得
    # ex. '報酬QP(+9400)'
    def load_reward_QP(self):
        try:
            reward_QP_name = self.df.filter(like='報酬QP', axis=1).columns[0]
            return reward_QP_name
        except IndexError:
            print('csvの1行目に 報酬QP が含まれていません')

    # 報酬QP(+xxxx) の列の位置
    # この後の列はアイテムドロップ数
    def get_qp_col_loc(self, df):
        return df.columns.get_loc(self.reward_QP_name) + 1

    # ドロップ数の列の位置を取得
    def get_drop_col_loc(self):
        try:
            drop_num = self.df.filter(like='ドロ数', axis=1).columns[0]
        except IndexError:
            print('csvの1行目に ドロ数 が含まれていません')
        return df.columns.get_loc('ドロ数') + 1

    # クエスト名を取得する
    def load_quest_name(self, csv_path):

        # csvデータからクエスト名を取得する
        try:
            quest_name = self.df.values[0][0]
        except IndexError:
            print("IndexError :1周分のデータのみ、重複、missingの可能性")
        except UnboundLocalError:
            print('UnboundLocalError')

        # csvにクエスト名が記述されていない場合は、csvのファイル名をクエスト名と
        # して取得する
        if quest_name == '合計':

            # google drive から直にcsvを読み込んだ場合など
            if type(csv_path) == io.BytesIO:
                quest_name = '[blank]'

            else:
                # 日時を取り除く (バッチファイル利用時に発生する日時)
                match = re.search('(?<=_\d\d_\d\d_\d\d_\d\d_\d\d).*', Path(csv_path).stem) 
                if match != None:

                    # クエスト場所名の文字列が空文字列になる場合は'[blank]'に置換
                    if match.group(0) == '':   
                        quest_name = '[blank]'

                    else:
                        quest_name = match.group(0)

                # csvファイル名がクエスト名と仮定してそのまま利用
                else:
                    print(quest_name, Path(csv_path).stem)
                    quest_name = Path(csv_path).stem

        return quest_name

    # 合計の行を除去
    # 報酬QP(+xxxx)の1行目の要素が1より大きければ合計行
    def remove_total_row(self, df, total_row):
        if not total_row and (df[self.reward_QP_name][0] > 1):
            try:
                df = df.drop(index=0)
                df = df.reset_index(drop=True)
            except IndexError:
                print(traceback.format_exc())
        return df

    # def shorten_the_quest_name(self):
    #     is_short = False

    #     # try:
    #     m = re.search('[a-zA-Z]+', self.quest_name)

    #     if self.quest_name == '[blank]':
    #         pass

    #     elif m != None:
    #         prev_quest_name = self.quest_name

            # アルファベットから始まっている場合は、アルファベットと文末のみにする
            # m = re.search('^[a-zA-Z]+', self.quest_name)
            # if m != None:
            #     # m = re.search('([a-zA-Z]+)(.+)(\s.+$)', self.quest_name)
                # m = re.search('([a-zA-Z]+)([^a-zA-Z]+)(\s)(.+$)', self.quest_name)
                # if m != None:
                #     self.quest_name = m.group(1) + m.group(3) + m.group(4)
            #         if prev_quest_name != self.quest_name:
            #             is_short = True

            # アルファベット以外から始まっている場合は、アルファベットを取り除く
            # else:

                # アルファベットで終わる場合、アルファベットを取り除く
                # m = re.search('[a-zA-Z]+$', self.quest_name)
                # if m != None:
                #     m = re.search('[^a-zA-Z]+', self.quest_name)
                #     if m != None:
                #         self.quest_name = m.group(0)
                #     if prev_quest_name != self.quest_name:
                #         is_short = True

                # アルファベットが挟まっている場合
                # VIP級など、クエストのランクがアルファベットになっている場合があったため、
                # そのままにする
                # else:
                    # m = re.search('([^a-zA-Z]+)([a-zA-Z]+)(.+$)', self.quest_name)
                    # self.quest_name = m.group(1) + m.group(3)
                    # is_short = True

        # except Exception:
        #     print('Exception occured. self.quest_name value is:', self.quest_name)
        #     print(traceback.format_exc())

        # self.quest_name = re.sub('\s+', ' ', self.quest_name)
        # if is_short:
        #     print('アルファベットが含まれているため、クエスト名を短縮します')
        #     print(prev_quest_name, ' ->', self.quest_name)

    # 複数行に分かれたデータを1行に統合する
    def mergeLines(self, df):

        def get_end_indexes(df):
            """各周回の最後の画像から得たデータ行のindex値を取得する"""
            df_end = df.index.stop - 1
            sum = 0
            end_indexes = []
            if self.isNewSpec:
                for i in range(df.index.start, df.index.stop):
                    sum += df['アイテム数'][i]

                    # 末尾の要素であればその周で最後の画像からのデータ
                    if i == df_end:

                        # missing の場合をチェックする
                        # missing でなければ、アイテム数の総和とドロ数[+1]が等しくなる
                        if (sum == df['ドロ数'][i]) | (sum == df['ドロ数'][i] + 1):
                            end_indexes.append(i)
                            sum = 0
                        else:
                            print(f"WARNING: 末尾の行でmissingの可能性があります Line:{i+1}")

                    else:

                        # アイテム数の総和がドロ数と等しければその周の最後の画像から得たデータ
                        # ただし、その次の行の総和がドロ数+1と等しい場合は特殊ケース(まんわかイベ等)
                        # のためスキップする
                        if (sum == df['ドロ数'][i]) & (sum + df['アイテム数'][i+1] != df['ドロ数'][i+1] + 1):
                            end_indexes.append(i)
                            sum = 0

                        # 特殊ケース(まんわかイベ等)
                        # アイテム数の総和がドロ数+1になる
                        if sum == df['ドロ数'][i]+1:
                            end_indexes.append(i)
                            sum = 0
            else:
                try:
                    end_indexes = df[~df['ドロ数'].astype(str).str.contains('\+', na=False)].index
                except AttributeError as e:
                    print(e)

            return end_indexes
            
        def get_start_indexes(df):
            """各周回の最初の画像から得たデータ行のindex値を取得する"""

            # startの位置は、最初は 0 で2ヵ所目からは end_indexes + 1 から求められる
            start_indexes = [0] + list(map(lambda x: x+1, get_end_indexes(df)))

            # ラストは存在しないので取り除く
            start_indexes.pop()

            return start_indexes

        def sum_rows(df, start_index, end_index):
            """指定された行範囲の各アイテムドロ数の総和を取得する"""

            # row行目の各アイテムドロ数
            def row(df, row):
                return df.iloc[row: row+1, self.QP_col_loc: ].values

            # start_index から end_index までの summation
            def recursive(f, i):
                if i == start_index:
                    return row(df, start_index)
                return recursive(f, i-1) + row(df, i)

            return recursive( row(df, start_index) , end_index )

        # アイテム数を削除する
        def remove_item_number(df):
            return df.drop(columns='アイテム数')

        # missing, duplicate, invaild などの無効な行を削除する
        def remove_invailed_row(df):
            try:

                # 報酬QPが 0 になっている行を削る
                df = df.drop(df[df[self.reward_QP_name] == 0].index)
                df = df.reset_index(drop=True)

            # QP0が存在しない場合はスキップする
            except IndexError:
                pass

            return df

        # 変数
        sIs = get_start_indexes(df)    # 開始位置
        eIs = get_end_indexes(df)      # 終了位置
        runs = len(sIs)                # 周回数

        # 各周回の最初の画像から得たデータ行にその周のデータの総和を書込む
        for sI, eI in [(sIs[i], eIs[i]) for i in range(runs)]:

            # 1周分のデータを計算して、報酬QP以前のカラムと結合
            df.iloc[ sI : sI + 1 ] = df.iloc[ sI : sI + 1, : self.QP_col_loc ].join(pd.DataFrame(
                    sum_rows(df, sI, eI),
                    columns = df.iloc[ : , self.QP_col_loc : ].columns,
                    index   = [sI]
            ))

            #ドロ数を更新する
            # ex. 20++ → 54,  (manwaka) 46 → 47
            df.iloc[sI : sI + 1, self.dro_col_loc : self.dro_col_loc + 1 ] = df.iloc[sI : sI + 1, self.QP_col_loc : ].sum(axis=1)


        # 既に足した行を削除する
        for sI, eI in [(sIs[i], eIs[i]) for i in range(runs)]:
            for i in range(sI + 1, eI + 1):
                df = df.drop(i)

        # index を更新する
        df = df.reset_index(drop=True)

        if self.isNewSpec:
            df = remove_item_number(df)
        df = remove_invailed_row(df)

        return df
    
    # TODO: ボーナスの影響を排除した獲得QP合計を計算して、獲得QP合計カラムに上書きする
    def calc_qp_sum_columns(self, df, qp_sum=False):

        if qp_sum == True:
            rew_qp = int(re.search(r'\d+', self.reward_QP_name).group())  # 報酬QPの値
            qp_cols = df.filter(regex='^QP')          # ドロップQPのカラム
            qp_col_names = qp_cols.columns            # ドロップQPのカラム名リスト
            n = len(qp_col_names)                     # ドロップQPのカラム数

            # 万, k などを数値に変換
            def change_value(line):
                line = re.sub("百万", '000000',  str(line))
                line = re.sub("万",   '0000',  str(line))
                line = re.sub("千",   '000',  str(line))
                line = re.sub("M",    '000000',  str(line))
                line = re.sub("K",    '000',  str(line))
                return line

            # i番目のQPカラムにおける QPドロップ値 の series を取得
            def get_drop_qp_values(i):
                qp_value = int( re.search(r'\d+', change_value( qp_col_names[i]) ).group() )
                qp_drops = qp_cols[ qp_col_names[i] ]
                return qp_value * qp_drops

            # 獲得QP合計を計算する
            res = rew_qp
            for i in range(n):
                res += get_drop_qp_values(i)

            # 獲得QP合計を書込む
            if '獲得QP合計' in df.columns:
                df['獲得QP合計'] = res
            else:
                df.insert(loc=self.dro_col_loc + 1, column='獲得QP合計', value=res)

        return df

    # 獲得QP合計 を削除する
    #   qp_sum True : 残す
    #   qp_sum False: 削除する
    @process_only_new_specification_data
    def remove_qp_sum_columns(self, df, qp_sum=False):
        if (qp_sum == False) & ('獲得QP合計' in df.columns):
            try:
                df = df.drop(columns='獲得QP合計')
            except KeyError as e:
                print(e)
        return df

    def get_number_of_run(self, df):

        # 報酬QP(+xxxx) カラムから周回数を計算
        try:
            QpColName = df.filter(like='報酬QP', axis=1).columns[0]
        except IndexError:
            print('csvの1行目に 報酬QP が含まれていません')
        return df[QpColName].sum()
