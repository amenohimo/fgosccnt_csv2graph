"""
csvからグラフを作る

fgosccnt.py から作成した csv ファイルを元に、グラフを作成するツール
fgosccnt.py : https://github.com/fgosc/fgosccnt

機能：
・箱ひげ図  
・ヴァイオリンプロット
・周回数毎のアイテムのドロップ数
・周回数毎のアイテムのドロップ率
・平行座標
・複数のファイルに対応
・ヒストグラム (予定)
・イベントアイテムのイベントボーナス+毎の表を作成 (予定)
・データテーブルの表示
・統計データの出力 (予定)
・画像に出力
・HTMLに出力
"""
import argparse
import warnings
import traceback
import re
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.offline as offline
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import unicodedata
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import japanize_matplotlib

from kaleido.scopes.plotly import PlotlyScope
scope = PlotlyScope()

import cufflinks as cf
cf.go_offline()
# cf.set_config_file(offline=True, theme="white", offline_show_link=False)

progname = "csv2graph"
version = "0.0.0.20200907"
warnings.simplefilter('ignore', FutureWarning)
pd.options.plotting.backend = "plotly"
quest_name = ''

def make_df(csv_path, total_row=False):
    """
    DataFrameを作成する
    fgosccntで作成したcsvから、プロットや統計処理に使用するDataFrameを作成する

    Args:
        csv_path (str): fgosccntで作成したcsvファイルのパス
        total_row (bool): 合計の行を残すか否か
                      グラフの処理では邪魔になるので残さない

    Returns:
        DataFrame: プロットや統計処理に使用するDataFrame
    """
    # print('\rDataFrame作成開始', end='')
    print('\r処理開始', end='')
    try:
        df = pd.read_csv(csv_path, encoding='shift-jis')
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding='UTF-8')
    except pd.errors.EmptyDataError:
        # 空のファイルを指定した可能性がある
        print(f'\rファイルが空の可能性があります:\n{csv_path}')
        return None

    # print('\rcsv読み込み完了  ', end='')

    # 新しい仕様かどうか
    # ドロ数 63, 42, 21, etc: True
    # ドロ数 20++, 21+, 20+ : False
    isNewSpecifications = 'アイテム数' in df.columns

    # csvにクエスト名があれば利用
    global quest_name
    try:
        # quest_name = df[df['ドロ数'].isnull()].values[0][0]
        quest_name = df.values[0][0]
    except IndexError:
        # IndexError が出るのは、1周分のデータしかないか、重複やmissingの可能性がある
        pass
    except UnboundLocalError:
        print('UnboundLocalError')
        return None

    # 合計の行を除去
    if not total_row:
        try:
            # df = df.drop(df[df['filename'].str.contains('合計', na=False)].index[0])
            # df = df.drop(df[df['ドロ数'].isnull()].index[0]) # fgoscdataに対応
            df = df.drop(index=0)        # ドロ数の合計を出しているファイルがあることを想定
            df = df.reset_index(drop=True)
        except IndexError:
            print('合計の行を取り除く：既に合計の行は取り除かれているか、始めから存在しません')

    df = df.fillna(0)

    if not isNewSpecifications:
        # ドロ数の列 20+が出現した行以降は全てstr型になるため、str型になっている数値を数値型に変換
        for i, row in enumerate(df[df.columns[1]]):
            if not ((row == '20+') or (row == '20++') or (row == '21+')):
                df.iloc[i, 1] = np.uint16(row)

    # ドロ数より後の列は numpy.float64 として読み込まれるので numpy.uint16 にキャスト
    for col in df.columns:
        if type(df[col].values[0]) == np.float64:
            df[col] = df[col].astype(np.uint16)

    ###
    ### over 20 collections start
    ###
    ### idxs_two  : 2枚の画像を使ったカウントにおける 最初の1枚目のindex値 の配列 21-41ドロ
    ### idxs_three: 3枚の画像を使ったカウントにおける 最初の1枚目のindex値 の配列 42-62ドロ
    ###

    # 文字列 報酬QP(+xxxx) を取得
    QpColName = df.filter(like='報酬QP', axis=1).columns[0]
    
    # 報酬QP(+xxxx) の列の位置 この後の列はアイテムドロップ数であり、ドロ数によっては2-3枚の和をとる
    QpColLoc = df.columns.get_loc(QpColName) + 1

    #
    # 42-62ドロ 
    #

    # 1枚目の行番号のリストを取得
    if isNewSpecifications:
        idxs_three = df.query('(42 <= ドロ数 <= 62) and (アイテム数 == 20)').index.tolist()
    else:

        # Int64Index([  1,   7,  ..., 121, 125], dtype='int64')
        idxs_three = df[df['ドロ数'] == '20++'].index.tolist()
    
    # [125, 121, ..., 7, 1]
    idxs_three.reverse()

    # 42-62ドロ の行を1行にまとめる
    for i, idx_three in enumerate(idxs_three):

        # filename, ドロ数, (アイテム数), 報酬QP
        df.iloc[idx_three: idx_three+1] = df.iloc[idx_three: idx_three+1, : QpColLoc].join(

            # 礼装～
            pd.DataFrame(
                (

                    # 0-20 の行
                    df.iloc[idx_three: idx_three+1, QpColLoc: ].values +

                    # 21-41 の行
                    df.iloc[idx_three+1: idx_three+2, QpColLoc: ].values +

                    # 42-62 の行
                    df.iloc[idx_three+2: idx_three+3, QpColLoc: ].values
                ),
                columns=df.iloc[idx_three: idx_three+1, QpColLoc: ].columns,
                index=[idx_three]
            )
        )

        # 20++ を正しい周回数に修正する
        if not isNewSpecifications:
            df.iloc[idx_three:idx_three+1, 1:2] = (
                20 + 21 + int(df.iloc[idx_three+2:idx_three+3, 1:2].values[0][0])
            )

        # 既に足した行　(次の2行)　を削除する
        df = df.drop(idx_three+1)
        df = df.drop(idx_three+2)

    df = df.reset_index(drop=True)

    #
    # 21-41ドロ 
    #

    # 1枚目の行番号のリストを取得
    if isNewSpecifications:
        idxs_two = df.query('(21 <= ドロ数 <= 41) and (アイテム数 == 20)').index.tolist()
    else:
        idxs_two = df[df['ドロ数'] == '20+'].index.tolist()
    idxs_two.reverse()

    # 21-41ドロ の行を1行にまとめる
    for i, idx_two in enumerate(idxs_two):

        # filename, ドロ数, 報酬QP
        df.iloc[idx_two: idx_two+1] = df.iloc[idx_two: idx_two+1, : QpColLoc].join(

            # 礼装～
            pd.DataFrame(
                (

                    # 0-20 の行
                    df.iloc[idx_two: idx_two+1, QpColLoc: ].values +

                    # 21-41 の行
                    df.iloc[idx_two+1: idx_two+1+1, QpColLoc: ].values

                ),
                columns=df.iloc[idx_two: idx_two+1, QpColLoc: ].columns,
                index=[idx_two]
            )
        )

        # 20+ を正しい周回数に修正する
        if not isNewSpecifications:
            df.iloc[idx_two:idx_two+1, 1:2] = (
                20 + int(df.iloc[idx_two+1:idx_two+2, 1:2].values[0][0])
            )

        # 既に足した行　(次の行)　を削除する
        df = df.drop(idx_two+1)

    df = df.reset_index(drop=True)
    ###
    ### over 20 collections end
    ###

    # アイテム数を削除
    if isNewSpecifications:
        df = df.drop(columns='アイテム数')

    # QPが0の行を取り除く (エラー発生行)
    try:
        df = df.drop(df[df[df.columns[2]] == 0].index[0])
        df = df.reset_index(drop=True)
    except IndexError: # QP0が存在しない場合しなければ次の処理へ
        pass

    # 合計行のドロ数を0から空欄に変更
    if total_row:
        df.iloc[0:1, 1:2] = ''

    # print('\rDataFrame作成完了', end='')

    return df

def get_east_asian_width_count(text):
    """
    与えられた文字列の幅 (pixel) を計算する

    前提は以下の通り
    font size : 14
    font: meiryo ?
    全角: 14 pixel
    半角:  4 pixel -> 8 pixel

    `unicodedata.east_asian_width` については以下を参照
    Unicode® Standard Annex #11 EAST ASIAN WIDTH:
    http://www.unicode.org/reports/tr11/

    全角: F, W, A
    半角: H, Na, N

    Wikipedia:
    https://ja.wikipedia.org/wiki/%E6%9D%B1%E3%82%A2%E3%82%B8%E3%82%A2%E3%81%AE%E6%96%87%E5%AD%97%E5%B9%85
    """
    pel = 0
    for c in text:
        if unicodedata.east_asian_width(c) in 'FWA':
            pel += 14
        else:
            pel += 8 # 4
    return pel

def get_east_asian_width_count_at(text, half, full):
    """
    指定された半角と全角の文字幅から、与えられた文字列の幅 (pixel) を計算する

    前提は以下の通り
    font size : ?
    font: meiryo ?
    全角: full pixel
    半角: half pixel

    `unicodedata.east_asian_width` については以下を参照
    Unicode® Standard Annex #11 EAST ASIAN WIDTH:
    http://www.unicode.org/reports/tr11/

    全角: F, W, A
    半角: H, Na, N

    Wikipedia:
    https://ja.wikipedia.org/wiki/%E6%9D%B1%E3%82%A2%E3%82%B8%E3%82%A2%E3%81%AE%E6%96%87%E5%AD%97%E5%B9%85
    """
    pel = 0
    for c in text:
        if unicodedata.east_asian_width(c) in 'FWA':
            pel += full
        else:
            pel += half
    return pel

def is_evernt(df):
    """イベントアイテムを含むか"""
    return len(df.columns[df.columns.str.contains('\(x')]) != 0

def plt_Ridgeline(df):
    df = drop_filename(df)
    fig = go.Figure()
    data = [go.Violin(x=df[col], name=col, showlegend=False, box_visible=True, meanline_visible=True) 
            for col in df.columns]
    layout = dict(title='')
    fig = go.Figure(data=data, layout=layout)
    fig.update_traces(orientation='h', side='positive', width=2, points=False)
    fig.update_xaxes(title_text="ドロップ数", dtick=1)
    # fig.update_layout(xaxis_showgrid=False, xaxis_zeroline=False)
    if args.web:
        offline.iplot(
            fig,
            config={
                "displaylogo":False,
                "modeBarButtonsToRemove":["sendDataToCloud"]
            }
        )
    if args.imgdir != None:
        export_img(fig, '稜線図')

def plt_violine(df):
    df = drop_filename(df)
    template = "seaborn"
    fig = go.Figure()
    data = [
        go.Violin(
            y=df[col], name=col, showlegend=False, box_visible=True, meanline_visible=True
        ) for col in df.columns
    ]
    TEXT_SIZE = 15
    SIZE_TO_TEXT_HEIGHT = {15: 13, 16: 15, 17: 15, 18: 15} # 実測値 TEXT_SIZE: px 
    if 15 <= TEXT_SIZE <= 18:
        TEXT_HEIGHT = SIZE_TO_TEXT_HEIGHT[TEXT_SIZE]
    elif TEXT_SIZE < 15:
        TEXT_HEIGHT = 13 # 実測値未測定
    elif 18 < TEXT_SIZE:
        TEXT_HEIGHT = 15 # 実測値未測定
    TOP = 50
    BOTTOM = 64
    AX_HEIGHT = 600
    FIG_HEIGHT = TOP + AX_HEIGHT + BOTTOM
    FIG_WIDTH = 70 * len(df.columns)
    OFFSET = 1
    # タイトルのy座標は、ボトム + 図の高さ + トップの半分の高さ + 文字の半分の高さ と 補正値
    # 割合で指定するため、FIG_HEIGHTで割る必要がある
    y = ( BOTTOM + AX_HEIGHT + ( TOP + TEXT_HEIGHT) / 2 - OFFSET ) / FIG_HEIGHT
    layout = dict(
        # updatemenus=updatemenus,
        title={
            'text':get_quest_name(),
            'x':0.5,
            'y':y, #0.955,
            'xanchor': 'center',
            'font':dict(size=TEXT_SIZE)
        },
        # yaxis=dict(range=[-0.5, df.max().max()]),
        height=FIG_HEIGHT, width=100*len(df.columns),
        margin=dict(l=40, t=TOP, b=BOTTOM, r=40, pad=0, autoexpand=False),
        template=template,
        yaxis=dict(
            # range=[-1.5, 30],
            dtick=1
        ),
    )
    fig = go.Figure(data=data, layout=layout)
    ymax = df.max().values.max()
    dtick = round(ymax/11/10)*10 if 200 < ymax else 10 if 100 < ymax else 5 if 30 < ymax else 1
    fig.update_yaxes(title_text="", dtick=dtick)
    if args.web:
        offline.iplot(
            fig,
            config={
                "displaylogo":False,
                "modeBarButtonsToRemove":["sendDataToCloud"]
            }
        )
    if args.imgdir != None:
        export_img(fig, 'ヴァイオリンプロット')

def plt_box(df):
    df = drop_filename(df)
    template = "seaborn"
    fig = go.Figure()
    data = [go.Box(y=df[col], name=col, showlegend=False)
            for col in df.columns]
    TEXT_SIZE = 15
    SIZE_TO_TEXT_HEIGHT = {15: 13, 16: 15, 17: 15, 18: 15} # 実測値 TEXT_SIZE: px 
    if 15 <= TEXT_SIZE <= 18:
        TEXT_HEIGHT = SIZE_TO_TEXT_HEIGHT[TEXT_SIZE]
    elif TEXT_SIZE < 15:
        TEXT_HEIGHT = 13 # 実測値未測定
    elif 18 < TEXT_SIZE:
        TEXT_HEIGHT = 15 # 実測値未測定
    TOP = 50
    BOTTOM = 64
    AX_HEIGHT = 600
    FIG_HEIGHT = TOP + AX_HEIGHT + BOTTOM
    FIG_WIDTH = 70 * len(df.columns)
    OFFSET = 1
    # タイトルのy座標は、ボトム + 図の高さ + トップの半分の高さ + 文字の半分の高さ と 補正値
    # 割合で指定するため、FIG_HEIGHTで割る必要がある
    y = ( BOTTOM + AX_HEIGHT + ( TOP + TEXT_HEIGHT) / 2 - OFFSET ) / FIG_HEIGHT
    layout = dict(
        # updatemenus=updatemenus,
        title={
            'text':get_quest_name(),
            'x':0.5,
            'y':y, #0.955,
            'xanchor': 'center',
            'font':dict(size=TEXT_SIZE)
        },
        height=FIG_HEIGHT, width=FIG_WIDTH, 
        margin=dict(
            l=40,
            t=TOP,
            b=BOTTOM,
            r=40,
            pad=0,
            autoexpand=False
        ),
        paper_bgcolor='#FFFFFF',# "#aaf",EAEAF2,DBE3E6,#FFFFFF (白)
        template=template
    )
    fig = go.Figure(data=data, layout=layout)
    ymax = df.max().values.max()
    dtick = round(ymax/11/10)*10 if 200 < ymax else 10 if 100 < ymax else 5 if 30 < ymax else 1
    fig.update_yaxes(title_text="", dtick=dtick)
    if args.web:
        offline.iplot(
            fig,
            config={
                "displaylogo":False,
                "modeBarButtonsToRemove":["sendDataToCloud"]
            }
        )
    if args.imgdir != None:
        export_img(fig, '箱ひげ図')

def plt_all(df, title='各周回数における素材ドロップ数', rate=False, range_expans=False):
    
    TOP = 65
    BOTTOM = 55
    left = 70
    right = 38
    axs = 100   # 図の1つあたりの縦幅 [pixel]
    vs_px = 72  # サブプロット間の間隔 [pixel]
    template = "seaborn"
    if rate:
        fill = 'tozerox' #  ['none', 'tozeroy', 'tozerox', 'tonexty', 'tonextx','toself', 'tonext']
        ticksuffix = '%'
        mode = 'lines+markers'
    else:
        fill = 'none'
        ticksuffix = ''
        mode = "lines+markers"
        ytext = 'ドロ数'
    df = drop_filename(df)
    
    total_runs = df.index.max() + 1
    cols = 2

    # グラフの行数
    rows = int(len(df.columns)/cols) if len(df.columns) %2 == 0 else int(len(df.columns)/cols + 1)

    # figure全体の高さは、図の高さ*個数 + 余白の高さの和
    fig_height = axs * rows + vs_px * (rows - 1) + TOP + BOTTOM

    # 図の間の間隔
    # plotlyではvertical spacingは割合でしか指定できない
    vs = vs_px / (fig_height - TOP - BOTTOM)

    fig = make_subplots(
        rows=rows,
        cols=cols,
        vertical_spacing=vs,
        subplot_titles=df.columns
    )

    for i, col in enumerate(df.columns):

        y = df[col]
        ymean = y[len(y)-1]

        # 300%以上の素材、100%以上の種火は%表記にしない
        is_exp = re.search('種火|灯火|猛火|業火', col) != None
        if 300 <= ymean:
            df[col] /= 100
            y = df[col]
            ymean = y[len(y)-1]
            if rate:
                ytext = '平均ドロ数'
            tix = ''
        elif (100 <= ymean) & is_exp:
            df[col] /= 100
            y = df[col]
            ymean = y[len(y)-1]
            if rate:
                ytext = '平均ドロ数'
            tix = ''
        else:
            if rate:
                ytext = 'ドロ率'
            tix = ticksuffix

        _ymin, _ymax = y.min(), y.max()
        yrange = _ymax - _ymin

        # 末尾付近をy軸方向に拡大
        if range_expans:
            # ymin = ymean - 1 * y[5:].std()
            ymin = ymean - yrange * 0.1
            if ymin <= 0:
                ymin = 0 - yrange * 0.04
            # ymax = ymean + 1 * y[5:].std()
            ymax = ymean + yrange * 0.1

        # 線が見えなくならないようにy軸の範囲を微調整する
        else:
            ymin = _ymin - yrange * 0.05  # 0%の時線が見えなくなるので 範囲 * 5% 下げる
            ymax = _ymax + yrange * 0.05  # minだけ調整すると上にずれるので 範囲 * +5% 上げる

        # 線と点の大きさを、周回数によって変化させる
        # 周回数が多い場合、サイズが大きいとそれぞれの点や線が重なり合って潰れてしまう
        if total_runs < 70:
            marker_size = 6
            line_width = 2
        elif total_runs <= 140:
            marker_size = 2
            line_width = 1

        # 200周以上はマーカーが完全に潰れるので、線を無しにする (100~200は未確認)
        else: 

            # ドロ率はfillと点で頑張る
            if rate: 
                marker_size = 1
                line_width = 0
            
            # ドロ数は線がないと意味不明瞭になるので、線を残す
            else: 
                marker_size = 2
                line_width = 1

        fig.add_trace(
            go.Scatter(
                x=df.index+1, y=y,
                mode=mode,
                name=col,
                fill=fill,

                # 凡例は、現在のplotlyの仕様だと一カ所にまとめて表示しかできない
                # 離れると分かりにくいため、図の上にそれぞれ素材名を表示して代用する
                # 手動で線や文字をお絵かきすれば、凡例擬きを自作することはおそらく可能
                showlegend=False,

                opacity=0.9,
                marker_size=marker_size,
                line_width=line_width
            ),
            row=int(i/2)+1, col=i%2+1 # グラフの位置
        )

        fig.update_yaxes(
            title_text=ytext,
            title_standoff=5,
            title_font={"size":11},
            range=[ymin, ymax],
            ticksuffix=tix,
            type="linear", # log
            # rangemode="tozero",
            row=int(i/2)+1, col=i%2+1
        )

        """
            Formatting Ticks in Python
            https://plotly.com/python/tick-formatting/

            https://plotly.com/python/axes/

            tickmode
            Parent: layout.coloraxis.colorbar
            Type: enumerated , one of ( 'auto' | 'linear' | 'array' )
            Sets the tick mode for this axis. If 'auto', the number of
            ticks is set via `nticks`. If 'linear', the placement of the
            ticks is determined by a starting position `tick0` and a tick
            step `dtick` ('linear' is the default value if `tick0` and `dtick`
            are provided). If 'array', the placement of the ticks is set via
            `tickvals` and the tick text is `ticktext`. ('array' is the default
            value if `tickvals` is provided).
            https://plotly.com/matlab/reference/#layout-margin
        """
        if total_runs < 30:
            dtick = 5
        elif total_runs < 101:
            dtick = 10
        elif total_runs < 300:
            dtick = 25
        elif total_runs < 1000:
            dtick = 100
        elif total_runs < 5000:
            dtick = 500
        elif total_runs < 10000:
            dtick = 1000
        else:
            dtick = 5000

        fig.update_xaxes(
            # range=[0, df.index.max()+1],
            # fixedrange=True, #  固定範囲 trueの場合、ズームは無効
            # tickmode='array', # Type: enumerated , one of ( 'auto' | 'linear' | 'array' )

            # tickmode='auto' の場合は、nticks でメモリ数を指定する
            # 楽だがどの刻み幅が使われるかわからない
            # nticks=20,

            # tickmode = 'linear' の場合は、tick0 と dick によってメモリを設定する
            # 0開始の0,5,10,...か、1開始の1,6,11,...の選択を迫られる
            tick0=0,
            dtick=dtick,

            # tickmode='array' の場合は、ticktext でメモリテキストを tickvals でメモリの配置を設定する
            # 周回数が少ない場合は、1,5,10,... でいいが、多い場合が課題になる
            # ticktext=[1 if i ==0 else 5 * i for i in range(int((df.index.max()+1)/5)+1)],
            # tickvals=[1 if i == 0 else 5 * i for i in range(int((total_runs)/5)+1)],

            title_text='周回数',
            title_standoff=0,
            title_font={"size":11},

            # x軸のラベルの位置の調整は、ドキュメントを探した限りだとやり方がなかった
            # 表示をOFFにして、位置を指定してテキストを直打することで代用はおそらく可能
            # title_xanchor='right',　

            row=int(i/2)+1, col=i%2+1
        )

    fig.update_layout(
        height=fig_height, width=1000, 

        # 背景色を変えてfigの範囲を確認する場合や、単に背景色を変えたい時に変更
        paper_bgcolor='#FFFFFF',# "#aaf",EAEAF2,DBE3E6

        title={'text':title,'x':0.5,'y':0.985,'xanchor': 'center', 'font':dict(size=15)},
        font=dict(size=12), template=template, legend = dict(x=1.005, y=1),
        margin=dict(l=left, t=TOP, b=BOTTOM, r=right, pad=0, autoexpand=False)
    )
    if args.web:
        offline.iplot(
            fig,
            config={
                "displaylogo":False,
                "modeBarButtonsToRemove":["sendDataToCloud"]
            }
        )
    if args.imgdir != None:
        export_img(fig, title)

def plt_rate(df):
    """
    ドロップ率の収束過程を表示する
    """
    df = drop_filename(df)

    # 一周ずつ素材数を加算
    tmp = df.values
    n = len(df)
    for i in range(n):

        # 最初は既に値が入っているので、スキップ
        if i == 0:
            continue

        # 現在の値と前の値を加算
        tmp[i] = tmp[i] + tmp[i-1]

    # それぞれの周回数で割り、%表記に合わせるために *100
    droprate_df = pd.DataFrame(columns=df.columns, data=[tmp[i]  / (i + 1) * 100 for i in range(n)])

    # ドロ数は%だと見にくいのでそのままで
    # droprate_df['ドロ数'] /= 100

    # %表記を指定してプロット
    plt_all(droprate_df.copy(), title='各周回数における素材ドロップ率', rate=True)
    plt_all(droprate_df.copy(), title='各周回数における素材ドロップ率 (平均値近傍の拡大)', rate=True, range_expans=True)

def export_img(fig, title, format='png'):
    """
        plotlyの出力結果を画像として保存する
        
        保存先：　<ディレクトリのパス>/<クエスト名> - <グラフのタイトル>.png
    """
    Img_dir = Path(args.imgdir)
    if not Img_dir.parent.is_dir():
        Img_dir.parent.mkdir(parents=True)
    img_path = Img_dir / Path(get_quest_name() + '-' + title + ".png")
    with open(img_path, "wb") as f:
        f.write(scope.transform(fig, format=format))

def drop_filename(df):
    """DataFrameからファイル名の列を削除する"""
    try:
        df = df.drop('filename', axis=1)
    except KeyError:
        pass
    return df

def get_quest_name():
    """
        短縮規則
        アルファベットから始まっている場合: アルファベットと文末の単語(～級など)のみに短縮
        (アルファベットが挟まっている場合:   アルファベットを取り除く) -> 問題があったため廃止
        アルファベットで終わる場合:        アルファベットを取り除く
    """
    if quest_name != '合計':
        return quest_name
    else:
        match = re.search('(?<=_\d\d_\d\d_\d\d_\d\d_\d\d).*', Path(csv_path).stem) # バッチファイル利用時
        if match != None:
            if match.group(0) == '':   # クエスト場所名の文字列が空文字列になる場合は置換
                place = '[blank]'
            else:
                place = match.group(0)
        else:
            place = Path(csv_path).stem   # csvファイル名がクエスト名と仮定してそのまま利用
        is_short = False
        try:

            # 文字列にアルファベットが含まれると、文字幅が変わるため、短縮する
            m = re.search('[a-zA-Z]+', place)
            if place == '[blank]':
                pass
            elif m != None:
                print(place, end='')

                # アルファベットから始まっている場合は、アルファベットと語末の単語のみにする
                m = re.search('^[a-zA-Z]+', place)
                if m != None:
                    m = re.search('([a-zA-Z]+)(.+)(\s.+$)', place)
                    place = m.group(1) + m.group(3)
                    is_short = True
                    print('アルファベットが含まれているため、クエスト名を短縮します')

                # アルファベット以外から始まっている場合は、アルファベットを取り除く
                else:

                    # (アルファベットから始まる場合と)アルファベットで終わる場合
                    m = re.search('[a-zA-Z]+$', place)
                    if m != None:
                        place = re.search('[^a-zA-Z]+', place).group(0)
                        is_short = True
                        print('アルファベットが含まれているため、クエスト名を短縮します')


                    # アルファベットが挟まっている場合
                    else:
                        # m = re.search('([^a-zA-Z]+)([a-zA-Z]+)(.+$)', place)
                        # place = m.group(1) + m.group(3)
                        pass # VIP級など、クエストのランクがアルファベットになっている場合があったため、処理しない

        except Exception as e:
            print(traceback.format_exc())
            print('Exception occured. place value is:', place)
        place = re.sub('\s+', ' ', place)
        if is_short:
            print(' ->', place)
        return place

def plt_table(df):
    """
        ドロップ数とドロップ率のテーブルを表示する
        ブラウザ上で枠線が表示されない場合は、ブラウザの拡大率を100％に戻すことで表示される

        クエストの名前は、以下から取得する
          - csvのファイル名
          - file nameの2行目 (fgoscdataに対応)
          - TODO　指定できるようにする

        表の幅は自動で調整する
          - プロポーショナルフォントには対応していない > TODO
          - Ｍなどの横幅の広いアルファベットが多いと改行が発生する
          - とりあえず短くしてみる

                `HIMEJIサバイバルカジノ ビギナー級` →　`HIMEJI ビギナー級`

        表示上のアイテム名列の幅 (実測値): 左右8pxずつ + 文字列の幅
        実際にぴったりの幅を指定すると、改行されレイアウトが崩れるため、更に7px余裕を持たせる

        デフォルトの列幅の比率は、15:6:9
            
    """
    TOP = 30
    BOTTOM = 30
    LEFT = 40
    RIGHT = 40
    CELL_HEIGHT = 26
    LINE_WIDTH = 1
    HIGHT_OFFSET = 1 # 上下の枠線が消える問題
    df = drop_filename(df)
    quest_name = get_quest_name()
    qn_width = get_east_asian_width_count(quest_name)
    place_width = (
        qn_width + 8 * 2 + 7 if 150 < qn_width + 8 * 2 + 14 else 150
    )
    DROPS_WIDTH, RATES_WIDTH = 6, 9
    items_width = np.ceil(place_width / 150 * (DROPS_WIDTH + RATES_WIDTH))
    width = place_width + 150 + LEFT + RIGHT

    runs = df.sum().values[1:2][0]
    items = df.sum().index[2:]
    drops = df.sum().values[2:]

    # ドロップ率

    #   小数点1位で統一する場合
    # rates = [f'{i/runs:>.2%}' for i in drops]

    #   有効桁数3桁以上にする場合
    rates = []
    for i in drops:
        dr = i / runs * 100
        # 4桁の時は有効桁数5 1234.5678... -> 1234.5%
        # n桁の時は有効桁数n+1
        # ただし、2桁以下の場合は有効桁数3    1.2345... -> 1.23%
        # 有効数字 significant figures (s.f.)
        n = len(str(int(dr//1)))
        if 3 <= n:
            sf = n + 1
        else:
            sf = 3
        drstr = f'{dr:>.{sf}g}'
        if is_integer(drstr):
            # 小数点第1位 (the tenths place) が0の場合、.0が省略される
            # 明示的に.0の表示を指定
            rates.append(f'{dr:>.1f} %')
        else:
            rates.append(drstr + ' %')

    fig = go.Figure(
        data=[
            go.Table(
                columnorder=[0, 1, 2],
                columnwidth=[items_width, DROPS_WIDTH, RATES_WIDTH],
                header=dict(
                    values=[quest_name, runs, ''],
                    line_color='black',
                    line_width=LINE_WIDTH,
                    fill_color='white',
                    align=['left', 'right', 'right'],
                    font_color='black',
                    font_size=14,
                    # font=dict(
                    #     color='black',
                    #     size=14
                    # ),
                    height=CELL_HEIGHT
                ),
                cells=dict(
                    values=[items, drops, rates],
                    # suffix=['', '', '%'],
                    line_color='black',
                    line_width=LINE_WIDTH,
                    fill_color='white',
                    align=['left', 'right'],
                    font=dict(
                        color='black',
                        size=14
                    ),
                    height=CELL_HEIGHT
                )
            )
        ]
    )
    fig.update_layout(
        # LINE_WIDTH は現時点で1or2以外の場合を考慮していない
        # 線幅を考える時に考える
        # HIGHT_OFFSETを設定しないと下の枠線が消える
        height=CELL_HEIGHT * len(df.columns[1:]) + TOP + BOTTOM + LINE_WIDTH + HIGHT_OFFSET,
        width=width,
        font=dict(size=14),
        # 背景色を変えてfigの範囲を確認する場合や、単に背景色を変えたい時に変更
        paper_bgcolor='white', # white', '#FFFFFF', "#aaf", '#EAEAF2', '#DBE3E6'
        margin=dict(
            # l=1,
            # r=1,
            # b=0,
            # t=1,
            # pad=0,
            l=LEFT,
            r=RIGHT,
            b=BOTTOM,
            t=TOP,
            pad=0,
            autoexpand=False
        )
    )
    if args.web:
        offline.iplot(
            fig,
            config={
                "displaylogo":False,
                "modeBarButtonsToRemove":["sendDataToCloud"]
            }
        )
    if args.imgdir != None:
        export_img(fig, 'table')
    if args.htmldir != None:
        exporpt_html(fig, 'table')       

def exporpt_html(fig, title):
    html_dir = Path(args.htmldir)
    if not html_dir.parent.is_dir():
        html_dir.parent.mkdir(parents=True)
    html_path = html_dir / Path(get_quest_name() + '-' + title + ".html")
    fig.write_html(fig, html_path)


def is_integer(n):
    """
        数値が整数かどうか判定する

        整数ならTrue, 整数以外ならFalseを返す
    """
    try:
        float(n)
    except ValueError:
        return False
    else:
        return float(n).is_integer()

def plt_event_line(df):
    """
        ボーナス毎のイベントアイテムのドロップ数を線形グラフを表示する
    """
   
    ##  イベントアイテムに使用するDF1
    #       TODO 変数名を考える
    #
    #       データ
    #         - (x3)などを取り除いたアイテム名
    #         - (x3)がついているアイテム名
    #         - (x3)がついているものの数
    #         - (x3)などを取り除いた場合の数

    # |    | アイテム名   | 枠名             |   ドロップ枠数 |   枠数 |   アイテム数 |
    # |----|--------------|------------------|----------------|--------|--------------|
    # |  0 | チェーンソー | チェーンソー(x3) |           1002 |      3 |         3006 |
    # |  1 | 薪           | 薪(x3)           |            153 |      3 |          459 |
    E_df = pd.DataFrame({
        'アイテム名':[re.search('.+(?=\(x\d)', i).group(0) for i in df.columns[df.columns.str.contains('\(x')]],
        '枠名': df.columns[df.columns.str.contains('\(x')],
        'ドロップ枠数':[df[i].sum() for i in df.columns[df.columns.str.contains('\(x')]],
        '枠数':[np.uint8(re.search('(?<=\(x)\d+', i).group(0)) for i in df.columns[df.columns.str.contains('\(x')]],
        'アイテム数':[df[i].sum() * np.uint8(re.search('(?<=\(x)\d+', i).group(0)) for i in df.columns[df.columns.str.contains('\(x')]]
    })

    ##  イベントアイテムに使用するDF2
    #       TODO 変数名を考える

    #       データ
    #         - アイテム毎、礼装ボーナス毎のドロップ数

    # |     |   チェーンソー(x3) |   薪(x3) |
    # |-----|--------------------|----------|
    # |  +0 |            45.5455 |  6.95455 |
    # |  +1 |            60.7273 |  9.27273 |
    # |  +2 |            75.9091 | 11.5909  |
    # |  +3 |            91.0909 | 13.9091  |
    # |  +4 |           106.273  | 16.2273  |
    # |  +5 |           121.455  | 18.5455  |
    # |  +6 |           136.636  | 20.8636  |
    # |  +7 |           151.818  | 23.1818  |
    # |  +8 |           167      | 25.5     |
    # |  +9 |           182.182  | 27.8182  |
    # | +10 |           197.364  | 30.1364  |
    # | +11 |           212.545  | 32.4545  |
    # | +12 |           227.727  | 34.7727  |
    E_df2 = pd.DataFrame(
        np.array(
            [
                # ボーナス礼装装備時の1周あたり平均アイテムドロップ数 [アイテムドロップ数/周]
                # (基本束数 + ボーナス増加数) [アイテムドロップ数/枠] * 平均枠数 [枠/周]
                # ex. (3 + 0~12) * 1002 / 66
                # ex. (3 + 0~12) * 153 / 66
                (E_df['枠数'].values[j] + i) * E_df['ドロップ枠数'].values[j] / len(df)
                for i in range(13) for j in range(len(E_df))
            ]
        ).reshape(13, len(E_df)),
        columns=[i for i in E_df['枠名']],
        index=['+' + str(i) for i in range(13)]
    )

    ##  イベントアイテムに使用するDF3
    #       TODO 変数名を考える

    #       データ
    #         - 束数毎に分かれていたアイテム毎の合計

    # |     |   チェーンソー |       薪 |
    # |-----|----------------|----------|
    # |  +0 |        45.5455 |  6.95455 |
    # |  +1 |        60.7273 |  9.27273 |
    # |  +2 |        75.9091 | 11.5909  |
    # |  +3 |        91.0909 | 13.9091  |
    # |  +4 |       106.273  | 16.2273  |
    # |  +5 |       121.455  | 18.5455  |
    # |  +6 |       136.636  | 20.8636  |
    # |  +7 |       151.818  | 23.1818  |
    # |  +8 |       167      | 25.5     |
    # |  +9 |       182.182  | 27.8182  |
    # | +10 |       197.364  | 30.1364  |
    # | +11 |       212.545  | 32.4545  |
    # | +12 |       227.727  | 34.7727  |
    E_df3 = pd.DataFrame()
    for i in E_df2.columns:
        if not re.search('.+(?=\(x\d)', i).group(0) in E_df3.columns: # アイテムの列がまだなければ作成
            E_df3[re.search('.+(?=\(x\d)', i).group(0)] = E_df2[i]
        else:
            E_df3[re.search('.+(?=\(x\d)', i).group(0)] += E_df2[i] # 既にあれば加算

    # 確認用コード
    # from tabulate import tabulate # コード確認用
    # print()
    # print(tabulate(E_df3, E_df3.columns, tablefmt='github', showindex=True))

    max2 = E_df2.max().max()
    max3 = E_df3.max().max()
    dtick2 = round(max2/11/10)*10 if 130 < max2 else 10 if 70 < max2 else 5 if 13 < max2 else 1
    dtick3 = round(max3/11/10)*10 if 130 < max3 else 10 if 70 < max3 else 5 if 13 < max3 else 1

    # イベントアイテム毎にドロ枠が何種類あるか
    keys = E_df3.columns
    values = np.zeros(len(E_df3.columns), dtype=np.uint8)
    d = dict(zip(keys, values))
    for j in E_df3.columns:
        for i in range(len(E_df2.columns)):
            m = re.search(j, E_df2.columns[i])
            if m != None:
                d[j] += 1

    # 2種類以上ある場合は、2つグラフを表示
    if 1 < max(d.values()):

        # イベントアイテムの平均ドロップ数
        from plotly.subplots import make_subplots
        template = "seaborn"
        fig = make_subplots(rows=1, cols=2, subplot_titles=('枠毎の平均ドロップ数', 'アイテム毎の平均ドロップ数'))
        
        # 左
        for i in range(len(E_df2.columns)):
            fig.add_trace(
                go.Scatter(
                    x=E_df2.index, # 礼装ボーナス増加数
                    y=E_df2[E_df2.columns[i]], # 平均アイテムドロップ数
                    name=E_df2.columns[i]
                ),
                row=1, col=1
            )
        fig.update_xaxes(
            title_text="礼装ボーナス",
            dtick=1,
            range=[0, 12],
            domain=[0, 0.45],
            row=1, col=1
        )
        fig.update_yaxes(
            title_text="ドロップ数",
            dtick=dtick2,
            row=1, col=1
        )

        # 右
        for i in range(len(E_df3.columns)):
            fig.add_trace(
                go.Scatter(
                    x=E_df3.index,
                    y=E_df3[E_df3.columns[i]],
                    name=E_df3.columns[i]
                ),
                row=1, col=2
            )
        fig.update_xaxes(
            title_text="礼装ボーナス",
            dtick=1,
            range=[0, 12],
            domain=[0.55, 1],
            row=1, col=2
        )
        fig.update_yaxes(
            title_text="",
            dtick=dtick3,
            row=1, col=2
        )

        fig.update_layout(
            height=570,
            width=1000,
            title={
                'text':"イベントアイテムの平均ドロップ数",
                'x':0.45,
                'y':0.98,
                'xanchor': 'center',
                'font':dict(size=15)
            },
            font=dict(size=12),
            annotations=[dict(font=dict(size=14))],
            template=template,
            legend=dict(x=1.005, y=1),
            margin=dict(l=70, t=65, b=55, r=120, pad=0, autoexpand=False),
            paper_bgcolor='white' # 'white' "LightSteelBlue"
        )
        if args.web:
            offline.iplot(
                fig,
                config={
                    "displaylogo":False,
                    "modeBarButtonsToRemove":["sendDataToCloud"]
                }
            )
        if args.imgdir != None:
            export_img(fig, 'ボーナス毎のイベントアイテムのドロップ数')

    # 1種類の場合は、1つグラフを表示
    else:
        template = "seaborn"
        # fig.titles('アイテム毎の平均ドロップ数')
        # fig = px.scatter(E_df3, x=E_df3.index, y=E_df3.columns,
        #                  #color=E_df2.columns,
        #                  labels=dict(index="ボーナス+", value="ドロップ数", variable=""),
        fig = go.Figure()
        for i in range(len(E_df3.columns)):
            fig.add_trace(
                go.Scatter(
                    x=E_df3.index, y=E_df3[E_df3.columns[i]],
                    name=E_df3.columns[i],
                    mode='lines+markers'
                )
            )
        fig.update_xaxes(
            title_text="概念礼装ボーナス",
            dtick=1,
            range=[0, 12],
            domain=[0, 1]
        )
        fig.update_yaxes(
            title_text="ドロップ数",
            title_standoff=5,
            dtick=dtick3
        )
        fig.update_layout(
            height=550, width=480,
            title={
                'text':"イベントアイテムの平均ドロップ数",
                # 'x':0.45, 'y':0.98,
                'x':0.5, 'y':0.96,
                'xanchor': 'center',
                'font':dict(size=14)
            },
            font=dict(size=12),
            template=template,
            legend=dict(x=0.03, y=.97), # 左上
            # legend=dict(x=1.05, y=1), # 右上外
            margin=dict(l=70, t=50, b=55, r=38, pad=0, autoexpand=False),
            paper_bgcolor='white'
        )
        if args.web:
            offline.iplot(
                fig,
                config={
                    "displaylogo":False,
                    "modeBarButtonsToRemove":["sendDataToCloud"]
                }
            )
        if args.imgdir != None:
            export_img(fig, 'ボーナス毎のイベントアイテムのドロップ数')

def plt_line_matplot(df):
    """
        概念礼装ボーナス毎のイベントアイテム獲得量をラインプロットで描く

    """
    E_df = pd.DataFrame({
        'アイテム名':[re.search('.+(?=\(x\d)', i).group(0) for i in df.columns[df.columns.str.contains('\(x')]],
        '枠名': df.columns[df.columns.str.contains('\(x')],
        'ドロップ枠数':[df[i].sum() for i in df.columns[df.columns.str.contains('\(x')]],
        '枠数':[np.uint8(re.search('(?<=\(x)\d+', i).group(0)) for i in df.columns[df.columns.str.contains('\(x')]],
        'アイテム数':[df[i].sum() * np.uint8(re.search('(?<=\(x)\d+', i).group(0)) for i in df.columns[df.columns.str.contains('\(x')]]
    })

    E_df2 = pd.DataFrame(
        np.array([(E_df['枠数'].values[j]+i)*E_df['ドロップ枠数'].values[j]/len(df.index.values)
                  for i in range(13) for j in range(len(E_df))]).reshape(13, len(E_df)),
        columns=[i for i in E_df['枠名']],
        index=['+' + str(i) for i in range(13)]
    )

    E_df3 = pd.DataFrame()
    for i in E_df2.columns:
        if not re.search('.+(?=\(x\d)', i).group(0) in E_df3.columns: # アイテムの列がまだなければ作成
            E_df3[re.search('.+(?=\(x\d)', i).group(0)] = E_df2[i]
        else:
            E_df3[re.search('.+(?=\(x\d)', i).group(0)] += E_df2[i] # 既にあれば加算

    #prepare data
    x = range(10)
    y = [i * 0.5 for i in range(10)]

    #2行1列のグラフの描画
    #subplot で 2*1 の領域を確保し、それぞれにグラフ・表を描画
    nrow = 2
    ncol = 2
    # plt.figure(figsize=(6*ncol,6*nrow))
    plt.figure(figsize=(12, 7))

    #１つ目のsubplot領域にグラフ
    plt.subplot(nrow, ncol, 1)
    for i in range(len(E_df2.columns)):
        plt.plot(
            E_df2.index,
            E_df2[E_df2.columns[i]],
            label=E_df2.columns[i],
            marker="o",
            markersize=5
        )
    plt.xticks(np.arange(13))
    plt.legend()
    plt.xlabel('礼装ボーナス')
    plt.ylabel('ドロップ数')

    plt.subplot(nrow, ncol, 2)
    for i in range(len(E_df3.columns)):
        plt.plot(E_df3.index, E_df3[E_df3.columns[i]], label=E_df3.columns[i], marker="p")
    plt.xticks(np.arange(13))
    plt.legend()
    plt.xlabel('礼装ボーナス')
    plt.ylabel('ドロップ数')

    #2つ目のsubplot領域に表
    plt.subplot(nrow, ncol, 3)
    column_names = ['col_{}'.format(i) for i in range(10)]
    row_names = ['x', 'y']
    plt.axis('off')  #デフォルトでgraphが表示されるので、非表示設定
    values = [x, y] # [[1,2,3],[4,5,6]]
    plt.table(cellText=values, colLabels=column_names, rowLabels=row_names, loc='upper center')

    plt.subplot(nrow, ncol, 4)
    column_names = ['col_{}'.format(i) for i in range(10)]
    row_names = ['x', 'y']
    plt.axis('off')  #デフォルトでgraphが表示されるので、非表示設定
    values = [x, y] # [[1,2,3],[4,5,6]]
    plt.table(cellText=values, colLabels=column_names, rowLabels=row_names, loc='upper center')

    #表示
    plt.subplots_adjust(left=0.05, right=0.91, bottom=0.1, top=0.95)
    plt.show()

def plt_sunburst(df):
    """
        イベントアイテムの円グラフを描く
        一目でドロップ割合の傾向を掴むことが目的

        ドロップ数で表示するとボーナスによって比率が変化する

        TODO ドロップ数を表示するか、枠数の比率を表示するか要検討
    """
    fig = px.sunburst(
        pd.DataFrame({'アイテム名':[re.search('.+(?=\(x\d)', i).group(0) for i in df.columns[df.columns.str.contains('\(x')]],
                    '枠名': df.columns[df.columns.str.contains('\(x')],
                    'アイテム数':[df[i].sum() * np.uint8(re.search('(?<=\(x)\d+', i).group(0)) for i in df.columns[df.columns.str.contains('\(x')]]}),
        path=['アイテム名', '枠名'], values='アイテム数',
    #     color='アイテム数',
    #     color_continuous_scale='RdBu'
    )
    template="seaborn" # ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "none"]
    fig.update_layout(
        height=600, width=1000, title={'text':"イベントアイテムの割合",'x':0.5,'xanchor': 'center'},
        font=dict(size=12), template=template, legend = dict(x = 1.005, y = 1))
    if args.web:
        offline.iplot(
            fig,
            config={
                "displaylogo":False,
                "modeBarButtonsToRemove":["sendDataToCloud"]
            }
        )
    if args.imgdir != None:
        export_img(fig, 'イベントアイテムの割合')

def plt_simple_parallel_coordinates(df):
    """
        平行座標を描く

        シンプルすぎて細かい表示や拘束範囲の調整ができないため、`plotly.express.parallel_coordinates`を使うのは
        表示のテストをしたいときに使う程度 (実はできるのかもしれないが方法がわからない)
            
            できないこと
              - データ軸の最大最小値の設定
              - 拘束範囲の設定
    """
    fig = px.parallel_coordinates(
        df.drop('filename', axis=1),
        color=df.columns[1],
        dimensions=[dim for dim in df.drop('filename', axis=1).columns],
        color_continuous_scale=px.colors.diverging.Portland,
        height=800, width=1300
    )
    # offline.plot(fig, filename = 'parallel_coordinates.html', config={"displaylogo":False, "modeBarButtonsToRemove":["sendDataToCloud"]}, auto_open=True)
    if args.web:
        offline.plot(
            fig,
            filename = 'simple_parallel_coordinates.html',
            config={
                "displaylogo":False,
                "modeBarButtonsToRemove":["sendDataToCloud"]
            },
            auto_open=True
        )
    if args.imgdir != None:
        export_img(fig, 'simple_parallel_coordinates')

def plt_parallel_coordinates(df):
    """
        平行座標を描く

        データフレームを受取り、平行座標を描く

        描いた平行座標はWebブラウザを開いて表示される
        開いたグラフはhtmlファイルとしてローカルに保存することができる

        color は何を着目しているかをよく考えて、セットを選ぶ
        必要であれば自分で定義する

            `colors.diverging` : 平均や中央値が重要な意味をもつデータの場合に優先
            `plotly.colors.cyclical`: 曜日・日・年など周期的構造がある場合に優先
            `colors.sequential`: 汎用性が高いので連続データなら何でも

        TODO html で保存するオプションを追加する
    """
    df = drop_filename(df)
    
    # 重複データを削除する
    df = df.drop_duplicates()

    dims = []
    width = 970
    # margin_left = margin_right = max([len(col) for col in df.columns])*10/2
    # label_len = int(np.floor((width - margin_left - margin_right) / (len(df.columns) - 1) / 10)) - 1 # 軸の間隔より
    margin_left = margin_right = max([get_east_asian_width_count_at(col, 5, 10) for col in df.columns])/2 + 4 # プロポーショナルの場合もあるので、念のため +4
    label_width = int(np.floor((width - margin_left - margin_right) / (len(df.columns))))
    for i in range(len(df.columns)):
        rmax = df[df.columns[i]].max()
        rmin = df[df.columns[i]].min() # if df.columns[i] == 'ドロ数' else 0 # 選択できるようにするべき?
        # cmin = 
        # cmax =
        l=''
        for s in df.columns[i]:
            if label_width < get_east_asian_width_count_at(l+s, 5, 10):
                break
            l += s
        label_len = len(l)
        dims.append(
            dict(range=[rmin, rmax],
        #          constraintrange=[cmin, cmax],  # TODO　引数で渡す？
                tickvals=list(set(df[df.columns[i]].tolist())), # ユニークな値をメモリ表示用に使用
                label=df.columns[i].replace('コイン', '').replace('報酬', '')[:label_len], # 長すぎると重なって読めなくなるので、適度にカットする
                values=df[df.columns[i]]
            )
        )
    lin = dict(
        color=df[df.columns[0]],
        colorscale='jet', # px.colors.diverging.Portland # 視認性はデータの把握に重要なのでいい設定を探す
        showscale=False,  # TODO　どちらも役に立つので、引数で指定できるようにするべきか？
        cmin=df[df.columns[0]].min(),
        cmax=df[df.columns[0]].max()
    )    
    fig = go.Figure(data=go.Parcoords(line = lin, dimensions = dims))
    fig.update_layout(
        width=width, height=400,
        margin = dict(l=margin_left, r=margin_right, b=20, t=50, pad=4),
        paper_bgcolor='white'#, #'black' # "LightSteelBlue" # 視認性はデータの把握に重要なのでいい設定を探す
        # plot_bgcolor='gold' # 'rgba(0,0,0,0)'
    )
    # offline.plot(fig, config={"displaylogo":False, "modeBarButtonsToRemove":["sendDataToCloud"]})
    if args.web:
        offline.plot(
            fig,
            filename = 'parallel_coordinates.html',
            config={
                "displaylogo":False,
                "modeBarButtonsToRemove":["sendDataToCloud"]
            },
            auto_open=True
        )
    if args.imgdir != None:
        export_img(fig, 'parallel_coordinates')

def plts(df):
    # print('\rグラフの描画開始', end='')
    if args.table or args.all:
        plt_table(df)
    if args.pc or args.all:
        plt_parallel_coordinates(df)
    # # plt_line_matplot(df)
    if args.event or args.all:
        if is_evernt(df): # イベントアイテム
            plt_event_line(df) 
            plt_sunburst(df)
        else:
            print('イベントアイテムを確認できませんでした')
    if args.box or args.all:
        plt_box(df)
    if args.violine or args.all:
        plt_violine(df)
    if args.drop or args.all:
        plt_all(df)
    if args.rate or args.all:
        plt_rate(df)
    # plt_Ridgeline(df)
    # print('\rグラフの描画完了', end='')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CSVからグラフを作成する')
    parser.add_argument('filenames', help='入力ファイル', nargs='*')
    parser.add_argument('--version', action='version', version=progname + " " + version)
    parser.add_argument('-w', '--web', action='store_true', help='webブラウザに出力する')
    parser.add_argument('-i', '--imgdir', help='画像ファイルの出力フォルダ')
    parser.add_argument('-html', '--htmldir', help='画像ファイルの出力フォルダ')
    parser.add_argument('-a', '--all', action='store_true', help='全てのプロットを作成')
    parser.add_argument('-v', '--violine', action='store_true', help='ヴァイオリンプロットを作成')
    parser.add_argument('-b', '--box', action='store_true', help='ボックスプロット(箱ひげ図)を作成')
    parser.add_argument('-p', '--pc', action='store_true', help='平行座標を作成')
    parser.add_argument('-t', '--table', action='store_true', help='表を作成')
    parser.add_argument('-d', '--drop', action='store_true', help='周回数ごとのドロップ数を作成')
    parser.add_argument('-r', '--rate', action='store_true', help='周回数ごとの素材ドロップ率を作成')
    parser.add_argument('-e', '--event', action='store_true', help='イベントアイテムのプロットを作成')

    args = parser.parse_args()

    # 出力先が指定されていない場合はwebブラウザに出力
    if (args.imgdir == None) & (args.web == False):
        args.web = True

    if args.all:
        args.web = False

    # ファイルが指定された場合
    # csvファイルを引数として受け取る
    # 指定するcsvファイルの枚数は複数にも対応
    if args.filenames:

        # 複数のファイルの場合
        if type(args.filenames) == list:
            for csv_path in args.filenames:
                plts(make_df(csv_path))
        
        # 1つのファイルの場合
        else:
            csv_path = args.filenames
            plts(make_df(csv_path))

    # ファイル名の指定がない場合は、テスト用のデータを実行
    else:
        # print('csvファイルの指定がないため、テストファイルによるプロットを実行します', end='')
        BASE_DIR = Path(__file__).resolve().parent
        args.all = True

        # テスト用のグラフ画像のファイル出力先を適宜指定
        args.imgdir = BASE_DIR / 'images'

        if not args.imgdir.parent.is_dir():
            args.imgdir.parent.mkdir(parents=True)
        csv_path = BASE_DIR / 'test_csv_files\Silent_garden_B.csv'
        plts(make_df(str(csv_path)))

    print('\r処理完了        ')
