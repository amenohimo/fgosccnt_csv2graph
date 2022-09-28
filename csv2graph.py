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
from asyncio.log import logger
import warnings
import re
from pathlib import Path
import unicodedata
from typing import NoReturn
from typing import Union

import numpy as np
import pandas as pd
import plotly
import plotly.offline as offline
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns
sns.set()
from kaleido.scopes.plotly import PlotlyScope
scope = PlotlyScope()
import cufflinks as cf
cf.go_offline()

from dataframe import Data

progname = "csv2graph"
version = "0.0.1.20220606.2"

warnings.simplefilter('ignore', FutureWarning)

pd.options.plotting.backend = "plotly"

quest_name = ''
report_data = None


def make_df(csv_path: str, total_row=False, qp_sum=False) -> pd.core.frame.DataFrame:
    """
    fgosccntで作成したcsvからDataFrameを作成する

    Args:
        csv_path (str): fgosccntで作成したcsvファイルのパス
        total_row (bool): 合計の行を残すか否か
                      グラフの処理では残さない

    Returns:
        DataFrame
    """
    global report_data
    global quest_name

    report_data = Data(csv_path, total_row=total_row, qp_sum=qp_sum)
    quest_name = report_data.quest_name
    return report_data.df


def output_graphs(
    fig: plotly.graph_objs._figure.Figure,
    graph_type: str
) -> NoReturn:

    def plot_on_web_browser(fig):
        offline.iplot(
            fig,
            config={
                "displaylogo": False,
                "modeBarButtonsToRemove": ["sendDataToCloud"]}
        )

    def export_img_file(fig, title, img_format='png'):
        output_path = _create_output_path(title, args.imgdir, img_format)
        with open(output_path, "wb") as f:
            f.write(scope.transform(fig, format=img_format))

    def exporpt_html(fig, graph_type):
        output_path = _create_output_path(graph_type, args.htmldir, '.html')
        fig.write_html(fig, output_path)

    def _create_output_path(graph_type, args_dir, file_suffix):
        dir = Path(args_dir)
        if not dir.parent.is_dir():
            dir.parent.mkdir(parents=True)
        return dir / Path(quest_name + '-' + graph_type + '.' + file_suffix)

    if args.web:
        plot_on_web_browser(fig)
    if args.imgdir is not None:
        export_img_file(fig, graph_type)
    if args.htmldir is not None:
        exporpt_html(fig, graph_type)


def get_pixel_of_width_of_string(text: str, half: int = 8, full: int = 14) -> int:
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
    width = 0
    for c in text:
        if unicodedata.east_asian_width(c) in 'FWA':
            width += full
        else:
            width += half
    return width


def plt_ridgeline(df: pd.core.frame.DataFrame) -> NoReturn:
    df: pd.core.frame.DataFrame = drop_filename(df)
    fig: plotly.graph_objs._figure.Figure = go.Figure()
    data = [go.Violin(x=df[col], name=col, showlegend=False, box_visible=True, meanline_visible=True)
            for col in df.columns]
    layout = dict(title='')
    fig = go.Figure(data=data, layout=layout)
    fig.update_traces(orientation='h', side='positive', width=2, points=False)
    fig.update_xaxes(title_text="ドロップ数", dtick=1)
    # fig.update_layout(xaxis_showgrid=False, xaxis_zeroline=False)
    output_graphs(fig, '稜線図')


def plt_not_ordered_graphs(df: pd.core.frame.DataFrame) -> NoReturn:
    """plot iolin plot or box plot"""

    layout = _get_not_ordered_graphs_layout(df)

    if args.violin:
        plot_data = _get_violine_data(df)
        fig = go.Figure(data=plot_data, layout=layout)
        output_graphs(fig, 'violin_plot')

    if args.box:
        plot_data = _get_box_data(df)
        fig = go.Figure(data=plot_data, layout=layout)
        output_graphs(fig, 'box_plot')


def _get_not_ordered_graphs_layout(df: pd.core.frame.DataFrame) -> dict:
    FONT_SIZE = 15
    AXIS_FONT_SIZE = 13
    TEXT_Y_OFFSET = 1
    TITLE_X = 0.5
    MARGIN_TOP = 50
    MARGIN_BOTTOM = 64
    MARGIN_RIGHT = 40
    MARGIN_LEFT = 76
    MARGIN_PAD = 0
    AXES_HEIGHT = 600

    fig_height = MARGIN_TOP + AXES_HEIGHT + MARGIN_BOTTOM
    fig_width = 100 * len(df.columns)  # 70 * len(df.columns)
    Text_size_2_text_heiht = {15: 13, 16: 15, 17: 15, 18: 15}
    if FONT_SIZE < 15:
        text_height = 13  # 未検証
    elif 15 <= FONT_SIZE <= 18:
        text_height = Text_size_2_text_heiht[FONT_SIZE]
    elif 18 < FONT_SIZE:
        text_height = 15  # 未検証
    title_y = (MARGIN_BOTTOM + AXES_HEIGHT + (MARGIN_TOP + text_height) / 2 - TEXT_Y_OFFSET) / fig_height
    df = drop_filename(df)
    ymax = df.max().values.max()
    dtick = round(ymax / 11 / 10) * 10 if 200 < ymax else 10 if 100 < ymax else 5 if 30 < ymax else 1

    layout = dict(
        title={
            'text': quest_name,
            'x': TITLE_X,
            'y': title_y,
            'xanchor': 'center',
            'font': dict(size=FONT_SIZE)},
        height=fig_height,
        width=fig_width,
        margin=dict(l=MARGIN_LEFT, t=MARGIN_TOP, b=MARGIN_BOTTOM, r=MARGIN_RIGHT, pad=MARGIN_PAD, autoexpand=False),
        template="seaborn",
        paper_bgcolor='#FFFFFF',  # "#aaf",EAEAF2,DBE3E6,#FFFFFF (白)
        xaxis=dict(
            title_text='obtained items',
            title_font=dict(size=AXIS_FONT_SIZE)
        ),
        yaxis=dict(
            title_text='number of items obtained',
            title_font=dict(size=AXIS_FONT_SIZE),
            # range=[-0.5, df.max().max()]
            # range=[-1.5, 30],
            # dtick=1,
            dtick=dtick
        )
    )
    return layout


def _get_violine_data(df: pd.core.frame.DataFrame):
    df = drop_filename(df)
    data = [
        go.Violin(
            y=df[col],
            name=col,
            showlegend=False,
            box_visible=True,
            meanline_visible=True
        ) for col in df.columns]
    return data


def _get_box_data(df: pd.core.frame.DataFrame):
    df = drop_filename(df)
    data = [
        go.Box(
            y=df[col],
            name=col,
            showlegend=False
        ) for col in df.columns]
    return data


def plot_line(
    df: pd.core.frame.DataFrame,
    title: str = '各周回数における素材ドロップ数',
    rate: bool = False,
    range_expans: bool = False
) -> NoReturn:

    MARGIN_TOP = 65
    MARGIN_BOTTOM = 55
    MARGIN_LEFT = 70
    MARGIN_RIGHT = 38
    height_of_sub_figure = 100
    vertical_spacing = 72  # サブプロット間の間隔 [pixel]
    template = "seaborn"
    if rate:
        fill = 'tozerox'  # ['none', 'tozeroy', 'tozerox', 'tonexty', 'tonextx','toself', 'tonext']
        ticksuffix = '%'
        mode = 'lines+markers'
    else:
        fill = 'none'
        ticksuffix = ''
        mode = "lines+markers"
        ytext = 'ドロ数'
    df = drop_filename(df)
    total_runs = df.index.max() + 1
    number_of_cols = 2
    r = int(len(df.columns) / number_of_cols)
    number_of_rows = r if len(df.columns) % 2 == 0 else r + 1
    fig_height = height_of_sub_figure * number_of_rows +\
        MARGIN_TOP + vertical_spacing * (number_of_rows - 1) + MARGIN_BOTTOM

    # In plotly, vertical spacing can only be specified as a percentage,
    # so calculate the percentage.
    vertical_spacing = vertical_spacing / (fig_height - MARGIN_TOP - MARGIN_BOTTOM)

    fig = make_subplots(
        rows=number_of_rows,
        cols=number_of_cols,
        vertical_spacing=vertical_spacing,
        subplot_titles=df.columns
    )

    for i, col in enumerate(df.columns):
        y = df[col]
        ymean = y[len(y) - 1]
        is_exp = re.search('種火|灯火|猛火|業火', col) is not None

        # %表記にしないアイテム
        # Over 300%
        if 300 <= ymean:
            df[col] /= 100
            y = df[col]
            ymean = y[len(y) - 1]
            if rate:
                ytext = '平均ドロ数'
            tix = ''

        # 100% over Ember, Light, Fire, Blaze
        elif (100 <= ymean) & is_exp:
            df[col] /= 100
            y = df[col]
            ymean = y[len(y) - 1]
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

        marker_size, line_width = _get_line_width_and_marker_size(total_runs, rate)

        fig.add_trace(
            go.Scatter(
                x=df.index + 1,
                y=y,
                mode=mode,
                name=col,
                fill=fill,

                # 凡例は、作成時現在のplotlyの仕様だと一カ所にまとめて表示しかできない
                # 離れると分かりにくいため、図の上にそれぞれ素材名を表示して代用する
                # 手動で線や文字を描画することで、凡例を自作することも可能と思われる
                showlegend=False,

                opacity=0.9,
                marker_size=marker_size,
                line_width=line_width
            ),

            # グラフの位置
            row=int(i / 2) + 1,
            col=i % 2 + 1
        )

        fig.update_yaxes(
            title_text=ytext,
            title_standoff=5,
            title_font={"size": 11},
            range=[ymin, ymax],
            ticksuffix=tix,
            type="linear",  # log
            # rangemode="tozero",
            row=int(i / 2) + 1,
            col=i % 2 + 1
        )

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
            dtick=_get_dtick(total_runs),

            # tickmode='array' の場合は、ticktext でメモリテキストを tickvals でメモリの配置を設定する
            # 周回数が少ない場合は、1,5,10,... でいいが、多い場合が課題になる
            # ticktext=[1 if i ==0 else 5 * i for i in range(int((df.index.max()+1)/5)+1)],
            # tickvals=[1 if i == 0 else 5 * i for i in range(int((total_runs)/5)+1)],

            title_text='周回数',
            title_standoff=0,
            title_font={"size": 11},

            # x軸のラベルの位置の調整は、ドキュメントを探した限りだとやり方がなかった
            # 表示をOFFにして、位置を指定してテキストを直打することで代用はおそらく可能
            # title_xanchor='right',　

            row=int(i / 2) + 1,
            col=i % 2 + 1
        )

    fig.update_layout(
        height=fig_height, width=1000,

        # 背景色を変えてfigの範囲を確認する場合や、単に背景色を変えたい時に変更
        paper_bgcolor='#FFFFFF',  # "#aaf",EAEAF2,DBE3E6

        title={'text': title, 'x': 0.5, 'y': 0.985, 'xanchor': 'center', 'font': dict(size=15)},
        font=dict(size=12), template=template, legend=dict(x=1.005, y=1),
        margin=dict(l=MARGIN_LEFT, t=MARGIN_TOP, b=MARGIN_BOTTOM, r=MARGIN_RIGHT, pad=0, autoexpand=False)
    )
    output_graphs(fig, title)


def _get_line_width_and_marker_size(total_runs, rate):

    # 周回数が多い場合、サイズが大きいとそれぞれの点や線が重なり合って潰れてしまうため、
    # 線と点の大きさを、周回数によって変化させる
    if total_runs < 70:
        marker_size = 6
        line_width = 2
    elif total_runs <= 140:
        marker_size = 2
        line_width = 1

    # 200周以上はマーカーが完全に潰れるので、線を無しにする (100~200は未確認)
    else:

        # ドロ率はfillと点にする
        if rate:
            marker_size = 1
            line_width = 0

        # ドロ数は線がないと意味不明瞭になるので、線を残す
        else:
            marker_size = 2
            line_width = 1

    return marker_size, line_width


def _get_dtick(total_runs):
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

    return dtick


def plt_rate(df: pd.core.frame.DataFrame):
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
        tmp[i] = tmp[i] + tmp[i - 1]

    # それぞれの周回数で割り、%表記に合わせるために *100
    droprate_df = pd.DataFrame(columns=df.columns, data=[tmp[i] / (i + 1) * 100 for i in range(n)])

    # ドロ数は%だと見にくいのでそのままで
    # droprate_df['ドロ数'] /= 100

    # %表記を指定してプロット
    plot_line(droprate_df.copy(), title='各周回数における素材ドロップ率', rate=True)
    plot_line(droprate_df.copy(), title='各周回数における素材ドロップ率 (平均値近傍の拡大)', rate=True, range_expans=True)


def drop_filename(df: pd.core.frame.DataFrame):
    """DataFrameからファイル名の列を削除する"""
    try:
        df = df.drop('filename', axis=1)
    except KeyError as e:
        logger.info(f'filenameのカラムが存在しません。drop_filename(df), {e}')
    return df


def plt_table(df: pd.core.frame.DataFrame) -> NoReturn:
    """
        ドロップ数とドロップ率のテーブルを表示する
        ブラウザ上で枠線が表示されない場合は、ブラウザの拡大率を100％に戻すことで表示される

        クエストの名前は、以下から取得する
          - csvのファイル名
          - file nameの2行目 (fgoscdataに対応)
          - TODO 指定できるようにする

        表の幅は自動で調整する
            既知の問題
            - プロポーショナルフォントには対応していない
            - Ｍなどの横幅の広いアルファベットが多いと改行が発生する
                -> 短くして対処
                    `HIMEJIサバイバルカジノ ビギナー級` →　`HIMEJI ビギナー級`

        表示上のアイテム名列の幅 (実測値): 左右8pxずつ + 文字列の幅
        実際にぴったりの幅を指定すると、改行されレイアウトが崩れるため、更に7px余裕を持たせる

        デフォルトの列幅の比率は、15:6:9
    """
    def is_integer(n: Union[int, float]) -> bool:
        """
            Receives a number and determines if it is an integer
            Returns True if it is an integer, False if it is not an integer
        """
        try:
            float(n)
        except ValueError:
            return False
        else:
            return float(n).is_integer()

    MARGIN_TOP = 30
    MARGIN_BOTTOM = 30
    MARGIN_LEFT = 40
    MARGIN_RIGHT = 40
    MARGIN_PAD = 0
    CELL_HEIGHT = 26
    LINE_WIDTH = 1
    HIGHT_OFFSET = 1  # 上下の枠線が消える問題のため調整を行う
    quest_name_width = get_pixel_of_width_of_string(quest_name)
    if 150 < quest_name_width + 8 * 2 + 14:
        place_width = quest_name_width + 8 * 2 + 7
    else:
        place_width = 150
    DROPS_WIDTH, RATES_WIDTH = 6, 9
    items_width = np.ceil(place_width / 150 * (DROPS_WIDTH + RATES_WIDTH))
    width = place_width + 150 + MARGIN_LEFT + MARGIN_RIGHT

    df = drop_filename(df)
    runs = report_data.run

    # アイテムカラムは、報酬QP(+xxxx) 次のカラム以降と仮定
    # ドロップしたアイテム名を取得
    QpColIndex = df.columns.get_loc(report_data.reward_QP_name)
    items = df.columns[QpColIndex + 1:]

    # ドロップしたアイテム数を取得
    drops = df.sum().values[QpColIndex + 1:]

    # ドロップ率

    # 小数点1位で統一する場合
    # rates = [f'{i/runs:>.2%}' for i in drops]

    # 有効桁数を3桁以上にする
    #    4桁の時は有効桁数       5       1234.5678... -> 1234.5%
    #    n桁の時は有効桁数       n + 1
    #    2桁以下の場合は有効桁数 3        1.2345... -> 1.23%
    rates = []
    for drop in drops:

        # drop rate
        drop_rate = drop / runs * 100

        # 整数部の桁数
        n = len(str(int(drop_rate // 1)))

        # 1000% 以上の場合に、改行されないよう調整
        if 4 <= n:
            DROPS_WIDTH = DROPS_WIDTH - 0.5
            RATES_WIDTH = RATES_WIDTH + 0.5

        # 3桁以上の場合は有効数字 n+1 桁
        elif 3 <= n:
            significant_figures = n + 1

        else:
            significant_figures = 3

        drop_rate_str = f'{drop_rate:>.{significant_figures}g}'

        # % を付与
        # 小数点第1位 (the tenths place) が0の場合、.0が省略されるため、.0を加える
        if is_integer(drop_rate_str):
            rates.append(f'{drop_rate:>.1f} %')

        else:
            rates.append(drop_rate_str + ' %')

    fig: plotly.graph_objs._figure.Figure = go.Figure(
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
        height=CELL_HEIGHT * len(df.columns[1:]) + MARGIN_TOP + MARGIN_BOTTOM + LINE_WIDTH + HIGHT_OFFSET,
        width=width,
        font=dict(size=14),
        # 背景色を変えてfigの範囲を確認する場合や、単に背景色を変えたい時に変更
        paper_bgcolor='white',   # white', '#FFFFFF', "#aaf", '#EAEAF2', '#DBE3E6'
        margin=dict(
            l=MARGIN_LEFT,
            r=MARGIN_RIGHT,
            b=MARGIN_BOTTOM,
            t=MARGIN_TOP,
            pad=MARGIN_PAD,
            autoexpand=False
        )
    )
    output_graphs(fig, 'table')


def plt_event_line(df: pd.core.frame.DataFrame):
    """
        ボーナス毎のイベントアイテムのドロップ数を線形グラフを表示する
    """

    #  イベントアイテムに使用するDF1
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
        'アイテム名': [re.search('.+(?=\(x\d)', i).group(0) for i in df.columns[df.columns.str.contains('\(x')]],
        '枠名': df.columns[df.columns.str.contains('\(x')],
        'ドロップ枠数': [df[i].sum() for i in df.columns[df.columns.str.contains('\(x')]],
        '枠数': [np.uint8(re.search('(?<=\(x)\d+', i).group(0)) for i in df.columns[df.columns.str.contains('\(x')]],
        'アイテム数': [df[i].sum() * np.uint8(re.search('(?<=\(x)\d+', i).group(0)) for i in df.columns[df.columns.str.contains('\(x')]]
    })

    #  イベントアイテムに使用するDF2
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

    #  イベントアイテムに使用するDF3
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
        if not re.search('.+(?=\(x\d)', i).group(0) in E_df3.columns:  # アイテムの列がまだなければ作成
            E_df3[re.search('.+(?=\(x\d)', i).group(0)] = E_df2[i]
        else:
            E_df3[re.search('.+(?=\(x\d)', i).group(0)] += E_df2[i]  # 既にあれば加算

    # 確認用コード
    # from tabulate import tabulate # コード確認用
    # print()
    # print(tabulate(E_df3, E_df3.columns, tablefmt='github', showindex=True))

    max2 = E_df2.max().max()
    max3 = E_df3.max().max()
    dtick2 = round(max2 / 11 / 10) * 10 if 130 < max2 else 10 if 70 < max2 else 5 if 13 < max2 else 1
    dtick3 = round(max3 / 11 / 10) * 10 if 130 < max3 else 10 if 70 < max3 else 5 if 13 < max3 else 1

    # イベントアイテム毎にドロ枠が何種類あるか
    keys = E_df3.columns
    values = np.zeros(len(E_df3.columns), dtype=np.uint8)
    d = dict(zip(keys, values))
    for j in E_df3.columns:
        for i in range(len(E_df2.columns)):
            m = re.search(j, E_df2.columns[i])
            if m is not None:
                d[j] += 1

    # 2種類以上ある場合は、2つグラフを表示
    if 1 < max(d.values()):

        # イベントアイテムの平均ドロップ数
        from plotly.subplots import make_subplots
        template = "seaborn"
        fig: plotly.graph_objs._figure.Figure = make_subplots(
            rows=1, cols=2, subplot_titles=('枠毎の平均ドロップ数', 'アイテム毎の平均ドロップ数'))

        # left plot
        for i in range(len(E_df2.columns)):
            fig.add_trace(
                go.Scatter(
                    x=E_df2.index,              # 礼装ボーナス増加数
                    y=E_df2[E_df2.columns[i]],  # 平均アイテムドロップ数
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

        # right plot
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
                'text': "イベントアイテムの平均ドロップ数",
                'x': 0.45,
                'y': 0.98,
                'xanchor': 'center',
                'font': dict(size=15)
            },
            font=dict(size=12),
            annotations=[dict(font=dict(size=14))],
            template=template,
            legend=dict(x=1.005, y=1),
            margin=dict(l=70, t=65, b=55, r=120, pad=0, autoexpand=False),
            paper_bgcolor='white'  # 'white' "LightSteelBlue"
        )
        output_graphs(fig, 'ボーナス毎のイベントアイテムのドロップ数')

    # 1種類の場合は、1つグラフを表示
    else:
        template = "seaborn"
        # fig.titles('アイテム毎の平均ドロップ数')
        # fig = px.scatter(E_df3, x=E_df3.index, y=E_df3.columns,
        #                  #color=E_df2.columns,
        #                  labels=dict(index="ボーナス+", value="ドロップ数", variable=""),
        fig: plotly.graph_objs._figure.Figure = go.Figure()
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
                'text': "イベントアイテムの平均ドロップ数",
                # 'x': 0.45, 'y':0.98,
                'x': 0.5, 'y': 0.96,
                'xanchor': 'center',
                'font': dict(size=14)
            },
            font=dict(size=12),
            template=template,
            legend=dict(x=0.03, y=.97),  # 左上
            # legend=dict(x=1.05, y=1),  # 右上外
            margin=dict(l=70, t=50, b=55, r=38, pad=0, autoexpand=False),
            paper_bgcolor='white'
        )
        output_graphs(fig, 'ボーナス毎のイベントアイテムのドロップ数')


def plt_line_matplot(df: pd.core.frame.DataFrame) -> NoReturn:
    """
        概念礼装ボーナス毎のイベントアイテム獲得量をラインプロットで描く

    """
    E_df = pd.DataFrame({
        'アイテム名': [re.search('.+(?=\(x\d)', i).group(0) for i in df.columns[df.columns.str.contains('\(x')]],
        '枠名': df.columns[df.columns.str.contains('\(x')],
        'ドロップ枠数': [df[i].sum() for i in df.columns[df.columns.str.contains('\(x')]],
        '枠数': [np.uint8(re.search('(?<=\(x)\d+', i).group(0)) for i in df.columns[df.columns.str.contains('\(x')]],
        'アイテム数': [df[i].sum() * np.uint8(re.search('(?<=\(x)\d+', i).group(0)) for i in df.columns[df.columns.str.contains('\(x')]]
    })

    E_df2 = pd.DataFrame(
        np.array([(E_df['枠数'].values[j] + i) * E_df['ドロップ枠数'].values[j] / len(df.index.values)
                  for i in range(13) for j in range(len(E_df))]).reshape(13, len(E_df)),
        columns=[i for i in E_df['枠名']],
        index=['+' + str(i) for i in range(13)]
    )

    E_df3 = pd.DataFrame()
    for i in E_df2.columns:
        if not re.search('.+(?=\(x\d)', i).group(0) in E_df3.columns:  # アイテムの列がまだなければ作成
            E_df3[re.search('.+(?=\(x\d)', i).group(0)] = E_df2[i]
        else:
            E_df3[re.search('.+(?=\(x\d)', i).group(0)] += E_df2[i]    # 既にあれば加算

    # prepare data
    x = range(10)
    y = [i * 0.5 for i in range(10)]

    # 2行1列のグラフの描画
    # subplot で 2*1 の領域を確保し、それぞれにグラフ・表を描画
    nrow = 2
    ncol = 2
    # plt.figure(figsize=(6*ncol,6*nrow))
    plt.figure(figsize=(12, 7))

    # 1つ目のsubplot領域にグラフ
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

    # 2つ目のsubplot領域に表
    plt.subplot(nrow, ncol, 3)
    column_names = ['col_{}'.format(i) for i in range(10)]
    row_names = ['x', 'y']
    plt.axis('off')  # デフォルトでgraphが表示されるので、非表示設定
    values = [x, y]  # [[1,2,3],[4,5,6]]
    plt.table(cellText=values, colLabels=column_names, rowLabels=row_names, loc='upper center')

    plt.subplot(nrow, ncol, 4)
    column_names = ['col_{}'.format(i) for i in range(10)]
    row_names = ['x', 'y']
    plt.axis('off')  # デフォルトでgraphが表示されるので、非表示設定
    values = [x, y]  # [[1,2,3],[4,5,6]]
    plt.table(cellText=values, colLabels=column_names, rowLabels=row_names, loc='upper center')

    # 表示
    plt.subplots_adjust(left=0.05, right=0.91, bottom=0.1, top=0.95)
    plt.show()


def plt_sunburst(df: pd.core.frame.DataFrame) -> NoReturn:
    """
        イベントアイテムの円グラフを描く
        一目でドロップ割合の傾向を掴むことが目的

        ドロップ数で表示するとボーナスによって比率が変化する

        TODO ドロップ数を表示するか、枠数の比率を表示するか要検討
    """
    fig: plotly.graph_objs._figure.Figure = px.sunburst(
        pd.DataFrame({
            'アイテム名': [re.search('.+(?=\(x\d)', i).group(0) for i in df.columns[df.columns.str.contains('\(x')]],
            '枠名': df.columns[df.columns.str.contains('\(x')],
            'アイテム数': [df[i].sum() * np.uint8(re.search('(?<=\(x)\d+', i).group(0)) for i in df.columns[df.columns.str.contains('\(x')]]
        }),
        path=['アイテム名', '枠名'], values='アイテム数',
        # color='アイテム数',
        # color_continuous_scale='RdBu'
    )
    template = "seaborn"  # ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "none"]
    fig.update_layout(
        height=600, width=1000, title={'text': "イベントアイテムの割合", 'x': 0.5, 'xanchor': 'center'},
        font=dict(size=12), template=template, legend=dict(x=1.005, y=1))
    output_graphs(fig, 'イベントアイテムの割合')


def plt_simple_parallel_coordinates(df: pd.core.frame.DataFrame) -> NoReturn:
    """
        平行座標を描く

        シンプルすぎて細かい表示や拘束範囲の調整ができないため、`plotly.express.parallel_coordinates`を使うのは
        表示のテストをしたいときに使う程度 (実はできるのかもしれないが方法がわからない)

            できないこと
              - データ軸の最大最小値の設定
              - 拘束範囲の設定
    """
    fig: plotly.graph_objs._figure.Figure = px.parallel_coordinates(
        df.drop('filename', axis=1),
        color=df.columns[1],
        dimensions=[dim for dim in df.drop('filename', axis=1).columns],
        color_continuous_scale=px.colors.diverging.Portland,
        height=800, width=1300
    )
    output_graphs(fig, 'simple_parallel_coordinates')


def plt_parallel_coordinates(df: pd.core.frame.DataFrame) -> NoReturn:
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

    # 報酬QPを削除
    df = df.drop(columns=df.filter(like='報酬', axis=1))

    # 重複データを削除する
    df = df.drop_duplicates()

    dims = []
    width = 970
    # margin_left = margin_right = max([len(col) for col in df.columns])*10/2
    # label_len = int(np.floor((width - margin_left - margin_right) / (len(df.columns) - 1) / 10)) - 1  # 軸の間隔より

    # プロポーショナルフォントの場合もあるので、念のため +4
    margin_left = margin_right = max([get_pixel_of_width_of_string(col, 5, 10) for col in df.columns]) / 2 + 4
    label_width = int(np.floor((width - margin_left - margin_right) / (len(df.columns))))
    for i in range(len(df.columns)):
        rmax = df[df.columns[i]].max()
        rmin = df[df.columns[i]].min()  # if df.columns[i] == 'ドロ数' else 0 # 選択できるようにするべき?
        # cmin =
        # cmax =
        lab = ''
        for s in df.columns[i]:
            if label_width < get_pixel_of_width_of_string(lab + s, 5, 10):
                break
            lab += s
        label_len = len(lab)
        dims.append(
            dict(
                range=[rmin, rmax],
                #  constraintrange=[cmin, cmax],  # TODO　引数で渡す？
                tickvals=list(set(df[df.columns[i]].tolist())),  # ユニークな値をメモリ表示用に使用
                label=df.columns[i].replace('コイン', '').replace('報酬', '')[:label_len],  # 長すぎると重なって読めなくなるので、適度にカットする
                values=df[df.columns[i]]
            )
        )
    lin = dict(
        color=df[df.columns[0]],
        colorscale='jet',  # px.colors.diverging.Portland # 視認性はデータの把握に重要なのでいい設定を探す
        showscale=False,   # TODO　どちらも役に立つので、引数で指定できるようにするべきか？
        cmin=df[df.columns[0]].min(),
        cmax=df[df.columns[0]].max()
    )
    fig = go.Figure(data=go.Parcoords(line=lin, dimensions=dims))
    fig.update_layout(
        width=width, height=400,
        margin=dict(l=margin_left, r=margin_right, b=20, t=50, pad=4),
        paper_bgcolor='white'  # 'black' 'LightSteelBlue' # 視認性はデータの把握に重要なのでいい設定を探す
        # plot_bgcolor='gold' # 'rgba(0,0,0,0)'
    )
    output_graphs(fig, 'parallel_coordinates')


def plot_graphs(df: pd.core.frame.DataFrame) -> NoReturn:

    def _is_evernt(df):
        """イベントアイテムを含むか"""
        return len(df.columns[df.columns.str.contains('\(x')]) != 0

    if args.table:
        plt_table(df)
    if args.pc:
        plt_parallel_coordinates(df)
    if args.event:
        if _is_evernt(df):
            plt_event_line(df)
            plt_sunburst(df)
            # plt_line_matplot(df)
    if args.box or args.violin:
        plt_not_ordered_graphs(df)
    if args.ridgeline:
        plt_ridgeline(df)
    if args.drops:
        plot_line(df)
    if args.rates:
        plt_rate(df)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='CSVからグラフを作成する')
    parser.add_argument('filenames', help='入力ファイル', nargs='*')
    parser.add_argument('--version', action='version', version=progname + " " + version)
    parser.add_argument('-w', '--web', action='store_true', help='output web browser')
    parser.add_argument('-i', '--imgdir', help='画像ファイルの出力フォルダ')
    parser.add_argument('-html', '--htmldir', help='htmlファイルの出力フォルダ')
    parser.add_argument('-a', '--plot_all_graphs', action='store_true', help='plot all graphs')
    parser.add_argument('-t', '--table', action='store_true', help='plot table')
    parser.add_argument('-d', '--drops', action='store_true', help='周回数ごとのドロップ数を作成')
    parser.add_argument('-r', '--rates', action='store_true', help='周回数ごとの素材ドロップ率を作成')
    parser.add_argument('-v', '--violin', action='store_true', help='plot violin plot')
    parser.add_argument('-b', '--box', action='store_true', help='plot box plot')
    parser.add_argument('--ridgeline', action='store_true', help='plot ridgeline plot')
    parser.add_argument('-p', '--pc', action='store_true', help='plot parallel coordinates')
    parser.add_argument('-e', '--event', action='store_true', help='plot event items')
    args = parser.parse_args()

    # If the output destination is not specified,
    # the graph drawing result is output to the web browser.
    if (args.imgdir is None) & (args.htmldir is None) & (args.web is False):
        args.web = True

    if args.plot_all_graphs:
        args.table = True
        args.drops = True
        args.rates = True
        args.violin = True
        args.box = True
        args.ridgeline = True
        args.pc = True
        args.event = True

    if args.filenames:
        if type(args.filenames) == list:
            for csv_path in args.filenames:
                plot_graphs(make_df(csv_path))
        else:
            csv_path = args.filenames
            plot_graphs(make_df(csv_path))

    # # ファイル名の指定がない場合は、テスト用のデータを実行
    # else:
    #     # print('csvファイルの指定がないため、テストファイルによるプロットを実行します', end='')
    #     BASE_DIR = Path(__file__).resolve().parent
    #     args.plot_all_graphs = True

    #     # テスト用のグラフ画像のファイル出力先を適宜指定
    #     args.imgdir = BASE_DIR / 'images'

    #     if not args.imgdir.parent.is_dir():
    #         args.imgdir.parent.mkdir(parents=True)
    #     csv_path = BASE_DIR / 'test_csv_files\Silent_garden_B.csv'
    #     plot_graphs(make_df(str(csv_path)))

    print('\r処理完了        ')
