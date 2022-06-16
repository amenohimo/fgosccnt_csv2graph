import io
import re
from pathlib import Path
import logging
import requests

import pandas as pd
import numpy as np


class Data:

    def __init__(self, csv_path=None, total_row=False, qp_sum=False):
        self.df = self.read_csv(csv_path)
        self.isNewSpec = self.isNewSpecifications(self.df)
        self.quest_name = self.load_quest_name(csv_path)
        self.reward_QP_name = self.load_reward_QP()
        self.QP_col_loc = self.get_qp_col_loc(self.df)
        self.drop_col_loc = self.df.columns.get_loc('ドロ数')
        self.df = self.remove_total_row(self.df, total_row)
        self.df = self.cast(self.df)
        self.raw_df = self.df
        self.df = self.merge_data_lines(self.df)
        self.df = self.calc_qp_sum_columns(self.df, qp_sum=qp_sum)
        self.df = self.remove_qp_sum_columns(self.df, qp_sum=qp_sum)
        self.run = self.get_number_of_run(self.df)

    def read_csv(self, csv_path: str) -> pd.core.frame.DataFrame:

        def get_csv_path_on_google_drive(shared_scv_url: str) -> io.BytesIO:
            id = shared_scv_url.split('/')[5]
            url = 'https://drive.google.com/uc?id=' + id
            return io.BytesIO(requests.get(url).content)

        if 'drive.google' in csv_path:
            csv_path = get_csv_path_on_google_drive(csv_path)

        try:
            return pd.read_csv(csv_path, encoding='shift-jis')
        except UnicodeDecodeError:
            return pd.read_csv(csv_path, encoding='UTF-8')
        except pd.errors.EmptyDataError:
            logging.error(f'The contents of csv are empty. csv_path = {csv_path}')
            raise pd.errors.EmptyDataError(
                f'The contents of csv are empty. csv_path = {csv_path}')

    # アイテム数を含む仕様か確認
    #    True : ドロ数 63, 42, 21
    #    False: ドロ数 20++, 21+, 20+
    def isNewSpecifications(self, df) -> bool:
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
            for i, row in enumerate(df['ドロ数']):
                if '+' not in str(row):
                    df.iloc[i, self.drop_col_loc] = np.uint16(row)

        # ドロ数より後の列は numpy.float64 として読み込まれるので numpy.uint16 にキャスト
        for col in df.columns:
            if type(df[col].values[0]) == np.float64:
                df[col] = df[col].astype(np.uint16)

        return df

    # 報酬QPのカラム名を取得
    # ex. '報酬QP(+9400)'
    def load_reward_QP(self) -> str:
        try:
            reward_QP_name = self.df.filter(like='報酬QP', axis=1).columns[0]
        except IndexError:
            logging.error('csvの1行目に 報酬QP が含まれていません')
            raise IndexError('csvの1行目に 報酬QP が含まれていません')
        else:
            return reward_QP_name

    # 報酬QP(+xxxx) の列の位置
    # この後の列はアイテムドロップ数
    def get_qp_col_loc(self, df):
        return df.columns.get_loc(self.reward_QP_name) + 1

    # クエスト名を取得する
    def load_quest_name(self, csv_path):

        # csvデータからクエスト名を取得する
        try:
            quest_name = self.df.values[0][0]
        except IndexError:
            logging.warning("IndexError :1周分のデータのみ、重複、missingの可能性")
        except UnboundLocalError as e:
            logging.error('UnboundLocalError')
            raise UnboundLocalError(f'UnboundLocalError: {e}')

        # csvにクエスト名が記述されていない場合は、csvのファイル名をクエスト名と
        # して取得する
        if quest_name == '合計':

            # google drive から直にcsvを読み込んだ場合など
            if type(csv_path) == io.BytesIO:
                quest_name = '[blank]'

            else:
                # 日時を取り除く (バッチファイル利用時に発生する日時)
                match = re.search('(?<=_\d\d_\d\d_\d\d_\d\d_\d\d).*', Path(csv_path).stem)
                if match is not None:

                    # クエスト場所名の文字列が空文字列になる場合は'[blank]'に置換
                    if match.group(0) == '':
                        quest_name = '[blank]'

                    else:
                        quest_name = match.group(0)

                # csvファイル名がクエスト名と仮定してそのまま利用
                else:
                    quest_name = Path(csv_path).stem

        return quest_name

    # 合計の行を除去
    # 報酬QP(+xxxx)の1行目の要素が1より大きければ合計行
    def remove_total_row(self, df, total_row):
        if not total_row and (df[self.reward_QP_name][0] > 1):
            try:
                df = df.drop(index=0)
            except IndexError:
                raise
            else:
                df = df.reset_index(drop=True)
        return df

    # 複数行に分かれたデータを1行に統合する
    def merge_data_lines(self, df):

        def get_rap_indexes(df):
            """
            各周の最初と最後のデータ行を取得する

            - 周の最初のデータ行は、周の最後のデータ行+1で得られる

            - 各周回の最後の画像から得たデータ行のindex値は以下の処理で得られる
            全てのデータフレームの行の要素に対して以下の処理を繰り返す

            1. 最初の行から最後の1つ手前の行まで
                1-1. その周(行)のドロ数dropが
                            drop <= 13
                を満たす場合は、以下の処理を行う

                    1-1-1. その周(行)のドロ数dropとその行(画像)のアイテム数itemに対し
                    て次のいずれかを満たす場合、
                            1-1-1. drop = item
                            1-1-2. drop = item - 1
                    結果のindex値にその行のindexを追加し、アイテム数の総和をリセット
                    する

                    1-1-2. 1-1-1を満たさない場合 (例えばドロ数10に対してアイテム
                    数8など) はデータに異常がある

                1-1-2. 1-1を満たさず、
                    その周(行)のドロ数dとその周のn枚の画像(n行)のアイテム数の総和
                    Sn = Σ_j=i-n^i(item_j)が次のいずれかを満たす場合、
                        1-2-1. drop = Sn
                        1-2-1. drop = Sn - 1
                    結果のindex値にその行のindexを追加し、アイテム数の総和をリセット
                    する

            2. 最後の行
                2-1. その周(行)のドロ数dropとその行(画像)のアイテム数itemに対し
                て次のいずれかを満たす場合、
                        2-1-1. drop = item
                        2-1-2. drop = item - 1
                結果のindex値にその行のindexを追加する

                2-2. その周(行)のドロ数dとその周のn枚の画像(n行)のアイテム数の総
                和 Sn = Σ_j=i-n^i(item_j) が次のいずれかを満たす場合、
                        2-2-1. drop = Sn
                        2-2-1. drop = Sn - 1
                結果のindex値にその行のindexを追加する

            - missing への対処
            ドロ数が変化した時に Sn が0になっていなければmissing等が発生している
                前回の周の不完全なデータを削除する
                不完全な周の最初と最後のデータ行indexを削除する
                indexの値を修正する
                Snをリセットする
            invaildはこの後の処理で報酬QP=0の行として削除される
            """
            file_col = df.columns.get_loc('filename')
            if df.iat[0, file_col] == 'missing':
                raise Exception('There is missing at the beginning of the data')
            isNewSpecifications = 'アイテム数' in df.columns
            if isNewSpecifications:
                drop: pd.core.series.Series = df['ドロ数']
                item: pd.core.series.Series = df['アイテム数']
                is_missing: pd.core.series.Series = df['filename'] == 'missing'
                missing_line = [idx for idx, b in enumerate(is_missing) if b is True]
                if missing_line:
                    raise Exception(
                        f'Contains "missing" at line {missing_line}. Check the csv data.')
                df_end = df.index.stop - 1
                sum = 0
                start_indexes = []
                end_indexes = []
                prv_drop = drop[0]
                skip = []
                for i in range(df.index.start, df.index.stop):
                    if i in skip:
                        continue
                    if i == 0 and not is_missing[i]:
                        start_indexes.append(0)
                    sum = sum + item[i]
                    value_of_one_image_is_equal = drop[i] == item[i]
                    value_of_one_image_is_equal_manwaka = drop[i] == item[i] - 1
                    values_of_sum_are_equal = drop[i] == sum
                    values_of_sum_are_equal_manwaka = drop[i] == sum - 1

                    logging.debug(
                        f'i={i}, value_of_one_image_is_equal={value_of_one_image_is_equal}, '
                        + f'value_of_one_image_is_equal_manwaka={value_of_one_image_is_equal_manwaka}, '
                        + f'values_of_sum_are_equal={values_of_sum_are_equal}, '
                        + f'values_of_sum_are_equal_manwaka={values_of_sum_are_equal_manwaka}, '
                        + f'is_missing={is_missing[i]}, '
                        + f'drop={drop[i]}, item={item[i]}, sum={sum}, '
                        + f'start_index={start_indexes}, end_index={end_indexes}'
                    )

                    # When 'missing' is detected
                    if((prv_drop != drop[i]) & (sum != item[i])):

                        # When missing occurs at the middle or end position.
                        if is_missing[i]:

                            # search start_index of missing
                            m = 1
                            while drop[i - m] == drop[i - m - 1]:
                                m += 1
                            if start_indexes[-1] != i - m:
                                start_indexes.append(i - m)

                            # search end_index of missing
                            if drop[i - 1] != drop[i + 1]:
                                end_indexes.append(i)
                            else:
                                n = 1
                                while drop[i + n] == drop[i + n + 1]:
                                    n += 1
                                end_indexes.append(i + n)

                            # Delete bad data
                            for k in range(start_indexes[-1], end_indexes[-1] + 1):
                                df = df.drop(k)

                            logging.warning(
                                'Detected the kind of missing at '
                                + f'line {start_indexes[-1]}-{end_indexes[-1]}, '
                                + f'detected_point={i}, '
                                + 'When missing occurs at the middle or end position. '
                                + f'prv_drop={prv_drop}, drop={drop[i]}, '
                                + f'sum={sum}, item={item[i]}, '
                                + f'start_index={start_indexes}, end_index={end_indexes}')
                            skip = range(i + 1, end_indexes[-1] + 1)
                            sum = 0

                            start_indexes.pop()
                            start_indexes.append(end_indexes[-1] + 1)
                            end_indexes.pop()

                            logging.debug(
                                f'start_index={start_indexes}, end_index={end_indexes}')

                        # When missing occurs at the beginning position.
                        else:
                            end_indexes.append(i - 1)
                            logging.debug(
                                'Detected the kind of missing at '
                                + f'line {start_indexes[-1]}-{end_indexes[-1]}, '
                                + f'detected_point={i}, '
                                + 'When missing occurs at the beginning position.'
                                + f'prv_drop={prv_drop}, drop={drop[i]}, '
                                + f'sum={sum}, item={item[i]}, '
                                + f'start_index={start_indexes}, end_index={end_indexes}')

                            # Delete bad data
                            for k in range(start_indexes[-1], end_indexes[-1] + 1):
                                df = df.drop(k)

                            start_indexes.pop()
                            end_indexes.pop()
                            start_indexes.append(i)
                            sum = 0
                            logging.debug(
                                'remove these data and continue processing. '
                                + f' doro={drop[i]}, item={item[i]}, sum={sum}, '
                                + f'is_missing={is_missing[i]}, '
                                + f'start_index={start_indexes}, end_index={end_indexes}'
                            )

                    # 最初の行から最後の1つ手前の行まで
                    if i != df_end:
                        if (drop[i] <= 13):
                            if (value_of_one_image_is_equal
                                    or value_of_one_image_is_equal_manwaka):
                                if not is_missing[i]:
                                    start_indexes.append(i + 1)
                                    end_indexes.append(i)

                            else:
                                logging.warning(
                                    'Even though the number of dorosu is 13 or less, '
                                    + 'it does not match the number of items!'
                                    + f'i={i}, drop={drop[i]}, item={item[i]}, sum={sum}, '
                                    + f'ei={end_indexes[-1]}, si={start_indexes[-1]}')
                            sum = 0
                        else:
                            next_sum = sum + item[i + 1]
                            not_manwaka = drop[i] != next_sum - 1
                            if ((values_of_sum_are_equal and not_manwaka)
                                    or values_of_sum_are_equal_manwaka):
                                start_indexes.append(i + 1)
                                end_indexes.append(i)
                                sum = 0

                    # 最後の行
                    else:
                        if (value_of_one_image_is_equal
                                or value_of_one_image_is_equal_manwaka):
                            end_indexes.append(i)
                        elif (values_of_sum_are_equal
                                or values_of_sum_are_equal_manwaka):
                            end_indexes.append(i)
                        else:
                            logging.warning(
                                'Last data of the last row does not match the number of items! '
                                + f'line={i}, drop={drop[i]}, item={item[i]}, sum={sum}'
                                + f'ei={end_indexes[-10:]}, si={start_indexes[-10:]}'
                            )

                    prv_drop = drop[i]

            else:
                try:
                    end_indexes = df[~df['ドロ数'].astype(str).str.contains('\+', na=False)].index
                except AttributeError as e:
                    print(e)
                start_indexes = [0] + list(map(lambda x: x + 1, end_indexes))
                start_indexes.pop()

            return start_indexes, end_indexes

        def sum_rows(df, start_index, end_index):
            """指定された行範囲の各アイテムドロ数の総和を取得する"""

            # row行目の各アイテムドロ数
            def row(df, row):
                return df.iloc[row: row + 1, self.QP_col_loc:].values

            # start_index から end_index までの summation
            def recursive(f, i):
                if i == start_index:
                    return row(df, start_index)
                return recursive(f, i - 1) + row(df, i)

            return recursive(row(df, start_index), end_index)

        # アイテム数を削除する
        def remove_item_number(df):
            return df.drop(columns='アイテム数')

        # missing, duplicate, invaild などの無効な行を削除する
        def _remove_invailed_row(df):

            # 報酬QPが 0 になっている行を削る
            try:
                df = df.drop(df[df[self.reward_QP_name] == 0].index)

            # QP0が存在しない場合はスキップする
            except IndexError:
                pass

            else:
                df = df.reset_index(drop=True)
            finally:
                return df

        sIs, eIs = get_rap_indexes(df)
        runs = len(sIs)                # 周回数
        if len(sIs) != len(eIs):
            raise Exception(
                'The number of indexes at the start position and the number of indexes '
                + 'at the end position of each lap do not match.')
            logging.warning(f'sIs={len(sIs)}, eIs={len(eIs)}')
            logging.warning(f'sIs={sIs}\n' + f'eIs={eIs}')

        # 各周回の最初の画像から得たデータ行にその周のデータの総和を書込む
        for sI, eI in [(sIs[i], eIs[i]) for i in range(runs)]:

            # 1周分のデータを計算して、報酬QP以前のカラムと結合
            df.iloc[sI: sI + 1] = df.iloc[sI:sI + 1, :self.QP_col_loc].join(pd.DataFrame(
                sum_rows(df, sI, eI),
                columns=df.iloc[:, self.QP_col_loc:].columns,
                index=[sI]
            ))

            # ドロ数を更新する
            # 20++ → 54,    (manwaka) 46 → 47
            df.iloc[sI:sI + 1, self.drop_col_loc: self.drop_col_loc + 1] = df.iloc[sI: sI + 1, self.QP_col_loc:].sum(axis=1)

        # 既に足した行を削除する
        for sI, eI in [(sIs[i], eIs[i]) for i in range(runs)]:
            for i in range(sI + 1, eI + 1):
                df = df.drop(i)

        # index を更新する
        df = df.reset_index(drop=True)

        if self.isNewSpec:
            df = remove_item_number(df)

        return df

    # TODO: ボーナスの影響を排除した獲得QP合計を計算して、獲得QP合計カラムに上書きする
    def calc_qp_sum_columns(self, df, qp_sum=False):

        if qp_sum is True:
            rew_qp = int(re.search(r'\d+', self.reward_QP_name).group())   # 報酬QPの値
            qp_cols = df.filter(regex='^QP')
            qp_col_names = qp_cols.columns
            n = len(qp_col_names)              # ドロップQPのカラム数

            # 万, k などを数値に変換
            def change_value(line):
                line = re.sub("百万", '000000', str(line))
                line = re.sub("万", '0000', str(line))
                line = re.sub("千", '000', str(line))
                line = re.sub("M", '000000', str(line))
                line = re.sub("K", '000', str(line))
                return line

            # i番目のQPカラムにおける QPドロップ値 の series を取得
            def get_drop_qp_values(i):
                qp_value = int(re.search(r'\d+', change_value(qp_col_names[i])).group())
                qp_drops = qp_cols[qp_col_names[i]]
                return qp_value * qp_drops

            # 獲得QP合計を計算する
            res = rew_qp
            for i in range(n):
                res += get_drop_qp_values(i)

            # 獲得QP合計を書込む
            if '獲得QP合計' in df.columns:
                df['獲得QP合計'] = res
            else:
                df.insert(loc=self.drop_col_loc + 1, column='獲得QP合計', value=res)

        return df

    # 獲得QP合計 を削除する
    #     qp_sum True : 残す
    #     qp_sum False: 削除する
    @process_only_new_specification_data
    def remove_qp_sum_columns(self, df, qp_sum=False):
        if (qp_sum is False) & ('獲得QP合計' in df.columns):
            try:
                df = df.drop(columns='獲得QP合計')
            except KeyError:
                raise
        return df

    def get_number_of_run(self, df):
        num_of_run = df[self.reward_QP_name].sum()
        return num_of_run
