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
version = "0.0.0.20200806"
warnings.simplefilter('ignore', FutureWarning)
pd.options.plotting.backend = "plotly"
quest_name = ''

def get_correct_20plus_df(df):
    """
    20+ になっているドロ数を正しい数に直して返す
    """

    # 20+ の行番号を取得
    row_20p = df[df['ドロ数'] == '20+'].index.tolist()    # Int64Index([  1,   7,  23,  31,  ..., 113, 115, 121, 125], dtype='int64')
    row_20p.reverse()    # [125, 121, 115, 113, ..., 31, 23, 1]

    for i in range(0, len(row_20p), 1):

        # ドロ数が20を超える場合のアイテムのドロップ数を計算して上書きする
        df.iloc[row_20p[i]:row_20p[i]+1] = df.iloc[row_20p[i]:row_20p[i]+1, :3].join(
            pd.DataFrame(
                (df.iloc[row_20p[i]:row_20p[i]+1, 3:len(df.columns)].values +
                 df.iloc[row_20p[i]+1:row_20p[i]+1+1, 3:len(df.columns)].values),
                columns=df.iloc[row_20p[i]:row_20p[i]+1, 3:len(df.columns)].columns,
                index=df.iloc[row_20p[i]:row_20p[i]+1].index
            )
        )

        # 20+ を正しい周回数に修正する
        df.iloc[row_20p[i]:row_20p[i]+1, 1:2] = (
            20 + int(df.iloc[row_20p[i]+1:row_20p[i]+2, 1:2].values[0][0])
        )

        # 既に足した行　(次の行)　を削除する
        df = df.drop(row_20p[i]+1)

    df = df.reset_index(drop=True)
    return df

def make_df(csv_path):
    """
    DataFrameを作成する
    fgosccntで作成したcsvから、プロットや統計処理に使用するDataFrameを作成する

    Args:
        csv_path (str): fgosccntで作成したcsvファイルのパス

    Returns:
        DataFrame: プロットや統計処理に使用するDataFrame
    """
    print('\rDataFrame作成開始', end='')
    try:
        df = pd.read_csv(csv_path, encoding='shift-jis')
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding='UTF-8')

    print('\rcsv読み込み完了', end='')

    # csvにクエスト名があれば利用
    global quest_name
    quest_name = df[df['ドロ数'].isnull()].values[0][0]

    # 合計の行を除去
    try:
        # df = df.drop(df[df['filename'].str.contains('合計', na=False)].index[0])
        df = df.drop(df[df['ドロ数'].isnull()].index[0]) # fgoscdataに対応
        df = df.reset_index(drop=True)
    except IndexError:
        print('合計の行を取り除く：既に合計の行は取り除かれているか、始めから存在しません')

    df = df.fillna(0)

    # ドロ数の列 20+が出現した行以降は全てstr型になるため、数値を数値型に変換
    for i, row in enumerate(df[df.columns[1]]):
        if not row == '20+':
            df.iloc[i, 1] = np.uint16(row)

    # 3列目以降は numpy.float64 として読み込まれる
    for col in df.columns:
        if type(df[col].values[0]) == np.float64:
            df[col] = df[col].astype(np.uint16)

    df = get_correct_20plus_df(df)

    # QPが0の行を取り除く (エラー発生行)
    try:
        df = df.drop(df[df[df.columns[2]] == 0].index[0])
        df = df.reset_index(drop=True)
    except IndexError: # QP0が存在しない場合しなければ次の処理へ
        pass

    print('\rDataFrame作成完了', end='')
    
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
    offline.iplot(fig, config={"displaylogo":False, "modeBarButtonsToRemove":["sendDataToCloud"]})
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
    # updatemenus = list([
    #     dict(active=1,
    #          buttons=list([
    #             dict(label='Log Scale',
    #                  method='update',
    #                  args=[{'visible': [True, True]},
    #                        {#'title': 'Log scale',
    #                            'yaxis': {'type': 'log'}}]),
    #             dict(label='Linear Scale',
    #                  method='update',
    #                  args=[{'visible': [True, False]},
    #                        {#'title': 'Linear scale',
    #                         'yaxis': {'type': 'linear'}
    #                        }
    #                  ]
    #                 )
    #          ]),
    #     )
    # ])
    layout = dict(
        # updatemenus=updatemenus,
        title='',
        # yaxis=dict(range=[-0.5, df.max().max()]),
        height=600, width=100*len(df.columns),
        margin=dict(l=70, t=60, b=55, r=40, pad=0, autoexpand=False),
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
    # fig.update_layout(
    #     height=700, width=1000, 
    #     title={'text':title,'x':0.45,'y':0.985,'xanchor': 'center', 'font':dict(size=15)}, 
    #     font=dict(size=12), template=template, legend = dict(x=1.005, y=1), 
    #     margin=dict(l=20, t=60, b=0, r=0, pad=0))
    offline.iplot(fig, config={"displaylogo":False, "modeBarButtonsToRemove":["sendDataToCloud"]})
    export_img(fig, 'ヴァイオリンプロット')

def plt_box(df):
    df = drop_filename(df)
    template = "seaborn"
    fig = go.Figure()
    data = [go.Box(y=df[col], name=col, showlegend=False)
            for col in df.columns]
    # updatemenus = list([
    #     dict(active=1,
    #          buttons=list([
    #             dict(label='Log Scale',
    #                  method='update',
    #                  args=[{'visible': [True, True]},
    #                        {#'title': 'Log scale',
    #                            'yaxis': {'type': 'log'}}]),
    #             dict(label='Linear Scale',
    #                  method='update',
    #                  args=[{'visible': [True, False]},
    #                        {#'title': 'Linear scale',
    #                         'yaxis': {'type': 'linear'}
    #                        }
    #                  ]
    #                 )
    #          ]),
    #     )
    # ])
    layout = dict(
        # updatemenus=updatemenus,
        title='',
        height=600, width=60*len(df.columns), 
        margin=dict(l=70, t=60, b=55, r=40, pad=0, autoexpand=False),
        template=template
    )
    fig = go.Figure(data=data, layout=layout)
    ymax = df.max().values.max()
    dtick = round(ymax/11/10)*10 if 200 < ymax else 10 if 100 < ymax else 5 if 30 < ymax else 1
    fig.update_yaxes(title_text="", dtick=dtick)
    # fig.update_layout(
    #     height=700, width=1000, 
    #     title={'text':title,'x':0.45,'y':0.985,'xanchor': 'center', 'font':dict(size=15)}, 
    #     font=dict(size=12), template=template, legend = dict(x=1.005, y=1), 
    #     margin=dict(l=20, t=60, b=0, r=0, pad=0))
    offline.iplot(fig, config={"displaylogo":False, "modeBarButtonsToRemove":["sendDataToCloud"]})
    export_img(fig, '箱ひげ図')

def plt_all(df, title='各周回数における素材ドロップ数', rate=False, range_expans=False):
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
    template = "seaborn"
    total_runs = df.index.max() + 1
    cols = 2

    # グラフの行数
    rows = int(len(df.columns)/cols) if len(df.columns) %2 == 0 else int(len(df.columns)/cols + 1)

    # サブプロットの設定
    top = 65
    bottom = 55
    left = 70
    right = 38
    axs = 100
    # slopes = 160
    # intercepts = 200
    # height = rows * slopes + intercepts
    vs_px = 72 # サブプロット間の間隔のピクセル値
    height = axs * rows + vs_px * (rows - 1) + top + bottom
    vs = vs_px / (height - top - bottom)
    fig = make_subplots(
        rows=rows, cols=cols,
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

        # 末尾付近のy軸を拡大
        if range_expans:
            # ymin = ymean - 1 * y[5:].std()
            ymin = ymean - yrange * 0.1
            if ymin <= 0:
                ymin = 0 - yrange * 0.04
            # ymax = ymean + 1 * y[5:].std()
            ymax = ymean + yrange * 0.1

        # y軸の全体を表示
        else:
            ymin = _ymin - yrange * 0.05  # 0%の時線が見えなくなるので 範囲 * -5% 下げる
            ymax = _ymax + yrange * 0.05  # minだけ調整すると上にずれるので 範囲 * +5% 上げる

        # 線と点の大きさを、周回数によって変化させる
        if total_runs < 70:
            marker_size = 6
            line_width = 2
        elif total_runs <= 140:
            marker_size = 2
            line_width = 1
        else: # 200周囲上はマーカーが完全に潰れるので、線を無しにする (100~200は未確認)
            if rate: # ドロ率はfillと点で頑張る
                marker_size = 1
                line_width = 0
            else: # ドロ数は線がないと意味不明になるので、線を残す
                marker_size = 2
                line_width = 1

        # Add traces
        fig.add_trace(
            go.Scatter(
                x=df.index+1, y=y, mode=mode, name=col, fill=fill, showlegend=False,
                opacity=0.9,
                # text='text',
                # textposition='top right',
                marker_size=marker_size,
                line_width=line_width
            ),
            row=int(i/2)+1, col=i%2+1 # グラフの位置
        )

        # Update yaxis properties
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

            # tickmode='array'の場合は、ticktext でメモリテキストを tickvals でメモリの配置を設定する
            # 周回数が少ない場合は、1,5,10,... でいいが、多い場合が課題になる
            # ticktext=[1 if i ==0 else 5 * i for i in range(int((df.index.max()+1)/5)+1)],
            # tickvals=[1 if i == 0 else 5 * i for i in range(int((total_runs)/5)+1)],

            title_text='周回数 [周]',
            title_standoff=0,
            title_font={"size":11},
            # title_xanchor='right',　# HELP どうやって右に表示するか分からない
            row=int(i/2)+1, col=i%2+1
        )

    # ymax = df.max().values.max()
    # dtick = round(ymax/11/10)*10 if 130 < ymax else 10 if 70 < ymax else 5 if 13 < ymax else 1
    # fig.update_xaxes(dtick=5) # メモリ幅
    # fig.update_yaxes(title_text="", ticksuffix = ticksuffix)
    fig.update_layout(
        height=height, width=1000, 
        paper_bgcolor='#FFFFFF',# "#aaf",EAEAF2,DBE3E6
        title={'text':title,'x':0.5,'y':0.985,'xanchor': 'center', 'font':dict(size=15)},
        font=dict(size=12), template=template, legend = dict(x=1.005, y=1),
        # xaxis=dict(
        #     dtick=5,
        # ),                             #title='', range = [0,100],
        # yaxis=dict(
        #     # range=[]
        #     # dtick=5
        # ),
        # yaxis=dict(title_text="", ticksuffix=ticksuffix),#title='', range = [0,100], dtick=5),
        margin=dict(l=left, t=top, b=bottom, r=right, pad=0, autoexpand=False))
    offline.iplot(fig, config={"displaylogo":False, "modeBarButtonsToRemove":["sendDataToCloud"]})
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
    plt_all(droprate_df.copy(), title='各周回数における累積素材ドロップ率', rate=True)
    plt_all(droprate_df.copy(), title='各周回数における累積素材ドロップ率 (平均値近傍の拡大)', rate=True, range_expans=True)

def export_img(fig, title):
    """"""
    Img_dir = Path(args.imgdir)
    if not Img_dir.parent.is_dir():
        Img_dir.parent.mkdir(parents=True)
    img_path = Img_dir / Path(get_quest_name() + '-' + title + ".png")
    with open(img_path, "wb") as f:
        f.write(scope.transform(fig, format="png"))

def drop_filename(df):
    """DataFrameからファイル名の列を削除する"""
    try:
        df = df.drop('filename', axis=1)
    except KeyError:
        pass
    return df

def get_quest_name():
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
        """
        短縮規則
        アルファベットから始まっている場合: アルファベットと文末の単語(～級など)のみに短縮
        アルファベットが挟まっている場合:   アルファベットを取り除く
        アルファベットで終わる場合:        アルファベットを取り除く
        """
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
    plotlyを使用
    枠線が表示されない場合は、ブラウザの拡大率を大きくして100％に戻すことで表示される

    クエストの名前は、パスから取得する
    表の幅は自動で調整する
        プロポーショナルフォントには対応していない > TODO
    Ｍなどの横幅の広いアルファベットが多いと改行が発生する
    `HIMEJIサバイバルカジノ ビギナー級` →　`HIMEJI ビギナー級` # とりあえず短くしてみる

    アイテム名の列は、左右8 pixel ずつに 文字列の幅を 加えて横幅が決まる
    デフォルトの幅の比率は、15:6:9
    ぴったりの幅にすると改行され、レイアウトが崩れるため、余裕を持たせている (+7 pxcel)
    """
    df = drop_filename(df)
    place = get_quest_name()
    place_width = (get_east_asian_width_count(place) + 8 * 2 + 7
                   if 150 < get_east_asian_width_count(place) + 8 * 2 + 14 else 150)
    drops_width, rates_width = 6, 9
    items_width = np.ceil(place_width / 150 * (drops_width + rates_width))
    width = place_width + 150 + 2 # 左右の線幅 1+1=2
    runs = df.sum().values[1:2][0]
    items = df.sum().index[2:]
    drops = df.sum().values[2:]
    rates = ['{:>.1%}'.format(i/runs) for i in drops]
    # rates = ['{:>.2g}'.format(i/runs) for i in drops]
    height = 26

    fig = go.Figure(data=[go.Table(
        columnorder = [0,1,2],
        columnwidth = [items_width, drops_width, rates_width],
        header=dict(values=[place, runs, ''],
                    line_color='black',
                    fill_color='white',
                    align=['left', 'right', 'right'],
                    font_size=14,
                    height=height),
        cells=dict(values=[items, drops, rates],
                   # suffix=['', '', '%'],
                   line_color='black',
                   fill_color='white',
                   align=['left', 'right'],
                   height=height
        ))
    ])
    fig.update_layout(
        height=height*len(df.columns[1:])+2, width=width,
        font=dict(size=14), paper_bgcolor='white',
        margin=dict(l=1, r=1, b=0, t=1, pad=0, autoexpand=False) # bは1でないと線が下の枠線が消える
    )
    offline.iplot(fig, config={"displaylogo":False, "modeBarButtonsToRemove":["sendDataToCloud"]})
    export_img(fig, '0_table')

def plt_event_line(df):
    """
    ボーナス毎のイベントアイテムのドロップ数を線形グラフを表示する
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

# #     E_df2.plot(labels=dict(index="ボーナス+", value="ドロップ数", variable=""))
# #     E_df3.plot(title='イベントアイテムドロップ数',
# # #            template="simple_white",
# #            labels=dict(index="ボーナス+", value="ドロップ数", variable=""))
#     fig = px.scatter(E_df2, x=E_df2.index, y=E_df2.columns,
#                     #color=E_df2.columns,
#                     labels=dict(index="ボーナス+", value="ドロップ数", variable=""),
#                     trendline="lowess")
#     fig.update_layout(
#         width=700, height=500, paper_bgcolor='white'#, #'black' # "LightSteelBlue" # 視認性はデータの把握に重要なのでいい設定を探す
#     )
#     offline.iplot(fig, filename = 'イベントアイテム',  config={"displaylogo":False, "modeBarButtonsToRemove":["sendDataToCloud"]})


#     fig = px.scatter(E_df3, x=E_df3.index, y=E_df3.columns,
#                  #color=E_df2.columns,
#                  labels=dict(index="ボーナス+", value="ドロップ数", variable=""),
#                  trendline="lowess")
#     fig.update_layout(
#         width=700, height=500, paper_bgcolor='white'#, #'black' # "LightSteelBlue" # 視認性はデータの把握に重要なのでいい設定を探す
#     )
#     offline.iplot(fig, filename = 'イベントアイテム',  config={"displaylogo":False, "modeBarButtonsToRemove":["sendDataToCloud"]})

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
        for i in range(len(E_df2.columns)):
            fig.add_trace(
                go.Scatter(x=E_df2.index, y=E_df2[E_df2.columns[i]], name=E_df2.columns[i]),
                row=1, col=1
            )
        for i in range(len(E_df3.columns)):
            fig.add_trace(
                go.Scatter(x=E_df3.index, y=E_df3[E_df3.columns[i]], name=E_df3.columns[i]),
                row=1, col=2
            )
        fig.update_xaxes(title_text="礼装ボーナス", dtick=1, range=[0, 12], domain=[0, 0.45],
                         row=1, col=1)
        fig.update_xaxes(title_text="礼装ボーナス", dtick=1, range=[0, 12], domain=[0.55, 1],
                         row=1, col=2)
        fig.update_yaxes(title_text="ドロップ数", dtick=dtick2, row=1, col=1)
        fig.update_yaxes(title_text="", dtick=dtick3, row=1, col=2)
        fig.update_layout(
            height=570, width=1000,
            title={
                'text':"イベントアイテムの平均ドロップ数",
                'x':0.45, 'y':0.985,
                'xanchor': 'center',
                'font':dict(size=15)
            },
            font=dict(size=12),
            annotations=[dict(font=dict(size=14))],
            template=template,
            legend=dict(x=1.005, y=1),
            margin=dict(l=70, t=65, b=55, r=90, pad=0, autoexpand=False),
            paper_bgcolor='white'
        )
        offline.iplot(fig, config={"displaylogo":False, "modeBarButtonsToRemove":["sendDataToCloud"]})
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
        fig.update_xaxes(title_text="概念礼装ボーナス", dtick=1, range=[0, 12], domain=[0, 1])
        fig.update_yaxes(title_text="ドロップ数", title_standoff=5, dtick=dtick3)
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
            margin=dict(l=70, t=65, b=55, r=38, pad=0, autoexpand=False),
            paper_bgcolor='white'
        )
        offline.iplot(fig, config={"displaylogo":False, "modeBarButtonsToRemove":["sendDataToCloud"]})
        export_img(fig, 'ボーナス毎のイベントアイテムのドロップ数')

def plt_line_matplot(df):
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
    y = [i*0.5 for i in range(10)]

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
        height=600, width=1000, title={'text':"",'x':0.5,'xanchor': 'center'},
        font=dict(size=12), template=template, legend = dict(x = 1.005, y = 1))
    offline.iplot(fig, filename = 'イベントアイテム',  config={"displaylogo":False, "modeBarButtonsToRemove":["sendDataToCloud"]})
    export_img(fig, 'イベントアイテムの割合')

def plt_simple_parallel_coordinates(df):
    """
    平行座標を描く
    シンプルすぎて細かい表示や拘束範囲の調整ができないため、`plotly.express.parallel_coordinates`を使うのは
    表示のテストをしたいときに使う程度 (実はできるのかもしれないが方法がわからない)
    できないこと：
    ・データ軸の最大最小値の設定
    ・拘束範囲の設定
    """
    fig = px.parallel_coordinates(
        df.drop('filename', axis=1),
        color=df.columns[1],
        dimensions=[dim for dim in df.drop('filename', axis=1).columns],
        color_continuous_scale=px.colors.diverging.Portland,
        height=800, width=1300
    )
    offline.plot(fig, filename = 'parallel_coordinates.html', config={"displaylogo":False, "modeBarButtonsToRemove":["sendDataToCloud"]}, auto_open=True)

def plt_parallel_coordinates(df):
    """
    平行座標を描く
    データフレームを受取り、平行座標を描く
    描いた平行座標はWebブラウザを開いて表示される
    開いたグラフはhtmlファイルとしてローカルに保存することができる
    color は何を着目しているかをよく考えて、セットを選ぶ
    必要であれば自分で定義する
    `colors.diverging` : 平均や中央値が重要な意味をもつデータの場合には優先
    `plotly.colors.cyclical`: 曜日・日・年など周期的構造がある場合には優先
    `colors.sequential`: 汎用性が高いので連続データなら何でも
    TODO png で保存するオプションを追加する
    TODO html で保存するオプションを追加する
    """
    df = drop_filename(df)
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
                label=df.columns[i][:label_len], # 長すぎると重なって読めなくなる
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
    offline.plot(fig, filename = 'parallel_coordinates.html', config={"displaylogo":False, "modeBarButtonsToRemove":["sendDataToCloud"]}, auto_open=True)
    export_img(fig, 'parallel_coordinates')

def plts(df):
    print('\rグラフの描画開始', end='')
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
    print('\rグラフの描画完了', end='')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CSVからグラフを作成する')
    parser.add_argument('filenames', help='入力ファイル', nargs='*')
    parser.add_argument('--version', action='version', version=progname + " " + version)
    parser.add_argument('-i', '--imgdir', help='画像ファイルの出力フォルダ')
    parser.add_argument('-a', '--all', action='store_true', help='全てのプロットを作成')
    parser.add_argument('-v', '--violine', action='store_true', help='ヴァイオリンプロットを作成')
    parser.add_argument('-b', '--box', action='store_true', help='ボックスプロット(箱ひげ図)を作成')
    parser.add_argument('-p', '--pc', action='store_true', help='平行座標を作成')
    parser.add_argument('-t', '--table', action='store_true', help='表を作成')
    parser.add_argument('-d', '--drop', action='store_true', help='周回数ごとのドロップ数を作成')
    parser.add_argument('-r', '--rate', action='store_true', help='周回数ごとの累積素材ドロップ率を作成')
    parser.add_argument('-e', '--event', action='store_true', help='イベントアイテムのプロットを作成')

    args = parser.parse_args()

    # ファイルが指定された場合
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
        csv_path = 'O:\_workspace\FGO_Count\csv\output2020_06_02_19_34_18フリーゲームバトル 上級.csv'
        plts(make_df(csv_path))

    print('\r処理完了        ')