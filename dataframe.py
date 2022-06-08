import io
import re
from pathlib import Path
import traceback
import logging

import pandas as pd
import numpy as np


class Data:

  def __init__(self, csv_path=None, total_row=False, qp_sum=False):
    self.df           = self.read_csv(csv_path)
    self.isNewSpec    = self.isNewSpecifications(self.df)
    self.quest_name   = self.load_quest_name(csv_path)
    self.reward_QP_name = self.load_reward_QP()
    self.QP_col_loc   = self.get_qp_col_loc(self.df)
    self.dro_col_loc  = self.df.columns.get_loc('ドロ数')
    self.df           = self.remove_total_row(self.df, total_row)
    self.df           = self.cast(self.df)
    self.raw_df       = self.df
    self.df           = self.merge_data_lines(self.df)
    self.df           = self.calc_qp_sum_columns(self.df, qp_sum=qp_sum)
    self.df           = self.remove_qp_sum_columns(self.df, qp_sum=qp_sum)
    self.run          = self.get_number_of_run(self.df)

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
      for i, row in enumerate(df['ドロ数']):
        if not ('+' in str(row)):
          df.iloc[i, self.dro_col_loc] = np.uint16(row)

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
        df = df.reset_index(drop=True)
      except IndexError:
        print(traceback.format_exc())
    return df
    
  # 複数行に分かれたデータを1行に統合する
  def merge_data_lines(self, df):

    def get_end_indexes(df):
      """各周回の最後の画像から得たデータ行のindex値を取得する"""
      drop: pd.core.series.Series = df['ドロ数']
      item: pd.core.series.Series = df['アイテム数']
      df_end = df.index.stop - 1
      sum = 0
      end_indexes = []
      isNewSpecifications = 'アイテム数' in df.columns
      if isNewSpecifications:
        for i in range(df.index.start, df.index.stop):
          sum = sum + item[i]
          value_of_one_image_is_equal          = drop[i] == item[i]
          value_of_one_image_is_equal_manwaka  = drop[i] == item[i] - 1
          values_of_sum_are_equal              = drop[i] == sum
          values_of_sum_are_equal_manwaka      = drop[i] == sum - 1

          # 最初の行から最後の1つ手前の行まで
          if i != df_end:
            if (drop[i] <= 13):
              if value_of_one_image_is_equal or value_of_one_image_is_equal_manwaka:
                end_indexes.append(i)
              else:
                logging.warning(
                  'Even though the number of dorosu is 13 or less,'
                  + 'it does not match the number of items!'
                  + f'i={i}, drop={drop[i]}, item={item[i]}, sum={sum}')
              sum = 0
            else:
              next_sum = sum + item[i + 1]
              not_manwaka = drop[i] != next_sum - 1
              if ((values_of_sum_are_equal and not_manwaka)
                  or values_of_sum_are_equal_manwaka):
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
                'Last data of the last row does not match the number of items!'
                + f'drop={drop[i]}, item={item[i]}, sum={sum}'
              )
          
          logging.debug(
            f'i={i}, value_of_one_image_is_equal={value_of_one_image_is_equal}, '
            + f'value_of_one_image_is_equal_manwaka={value_of_one_image_is_equal_manwaka}, '
            + f'values_of_sum_are_equal={values_of_sum_are_equal}, '
            + f'values_of_sum_are_equal_manwaka={values_of_sum_are_equal_manwaka}, '
            + f'drop={drop[i]}, item={item[i]}, sum={sum}'
          )

      else:
        try:
          end_indexes = d[~d['ドロ数'].astype(str).str.contains('\+', na=False)].index
        except AttributeError as e:
          print(e)

      return end_indexes

    def get_start_indexes(end_indexes):
      """各周回の最初の画像から得たデータ行のindex値を取得する"""

      # startの位置は、最初は 0 で2ヵ所目からは end_indexes + 1 から求められる
      start_indexes = [0] + list(map(lambda x: x + 1, end_indexes))

      # ラストは存在しないので取り除く
      start_indexes.pop()

      return start_indexes

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

      return recursive(row(df, start_index) , end_index)

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
    eIs = get_end_indexes(df)     # 終了位置
    sIs = get_start_indexes(eIs)  # 開始位置
    runs = len(sIs)        # 周回数

    # 各周回の最初の画像から得たデータ行にその周のデータの総和を書込む
    for sI, eI in [(sIs[i], eIs[i]) for i in range(runs)]:

      # 1周分のデータを計算して、報酬QP以前のカラムと結合
      df.iloc[ sI : sI + 1 ] = df.iloc[ sI : sI + 1, : self.QP_col_loc ].join(pd.DataFrame(
        sum_rows(df, sI, eI),
        columns=df.iloc[:, self.QP_col_loc:].columns,
        index=[sI]
      ))

      #ドロ数を更新する
      # 20++ → 54,  (manwaka) 46 → 47
      df.iloc[sI : sI + 1, self.dro_col_loc: self.dro_col_loc + 1] = df.iloc[sI: sI + 1, self.QP_col_loc : ].sum(axis=1)

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
      qp_cols = df.filter(regex='^QP')      # ドロップQPのカラム
      qp_col_names = qp_cols.columns      # ドロップQPのカラム名リスト
      n = len(qp_col_names)           # ドロップQPのカラム数

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
        df.insert(loc=self.dro_col_loc + 1, column='獲得QP合計', value=res)

    return df

  # 獲得QP合計 を削除する
  #   qp_sum True : 残す
  #   qp_sum False: 削除する
  @process_only_new_specification_data
  def remove_qp_sum_columns(self, df, qp_sum=False):
    if (qp_sum is False) & ('獲得QP合計' in df.columns):
      try:
        df = df.drop(columns='獲得QP合計')
      except KeyError as e:
        print(e)
    return df

  def get_number_of_run(self, df):

    # 報酬QP(+xxxx) カラムから周回数を計算
    # try:
    #   QpColName = df.filter(like='報酬QP', axis=1).columns[0]
    # except IndexError:
    #   print('csvの1行目に 報酬QP が含まれていません')
    # return df[QpColName].sum()
    num_of_run = df[self.reward_QP_name].sum()
    return num_of_run
