開発中

## csv2graph.py

[fgosccnt.py](https://github.com/fgosc/fgosccnt) から作成した csv ファイルを元に、グラフを作成する

作成できるグラフと機能
- ドロップアイテム
  - データテーブル
  - 箱ひげ図
  - 平行座標
  - ヒストグラム or 棒グラフ
  - ヴァイオリンプロット
  - イベントアイテムのイベントボーナス毎の線グラフと表 (予定)
- 周回数毎のアイテムのドロップ数
- 周回数毎のアイテムのドロップ率
- 統計データの出力 (予定)

### 使い方

仕様が固まっていないため、変更になる可能性があります

```
positional arguments:
  filenames             入力ファイル

optional arguments:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
  -i IMGDIR, --imgdir IMGDIR
                        画像ファイルの出力フォルダ
  -a, --all             全てのプロットを作成
  -v, --violine         ヴァイオリンプロットを作成
  -b, --box             ボックスプロット(箱ひげ図)を作成
  -p, --pc              平行座標を作成
  -t, --table           表を作成
  -d, --drop            周回数ごとのドロップ数を作成
  -r, --rate            周回数ごとの累積素材ドロップ率を作成
  -e, --event           イベントアイテムのプロットを作成
```

使用例
```
$ python csv2graph.py <csv_path> -i <images_output_dir> -a
```

## mergecsv.py (おまけ)

複数ページに分かれている(ドロ数20+とその次のドロ数) fgosccnt.py の出力を、
周回数毎のデータに統合して書き出すプログラム
    
### 使い方
```
$ python mergecsv.py [-h] [--version] [-c CHARACTER_CODE] [-o OUTPUT_PATH] [-f OUTPUT_FOLDER] [-r] [filenames [filenames ...]]

positional arguments:
  filenames             入力ファイル

optional arguments:
  -h, --help            show help message and exit
  --version             show program's version number and exit
  -c CHARACTER_CODE, --character_code CHARACTER_CODE
                        出力時の文字コード EUC-8 shift-jis など
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        ファイル名までの出力先パスを指定
  -f OUTPUT_FOLDER, --output_folder OUTPUT_FOLDER
                        出力先フォルダ
  -r, --remove_total_row
                        合計行を削除する
```

#### 使用例

csvファイルを指定する
出力先フォルダを指定していない場合は、プログラムのフォルダに同名で出力します
```
$ python mergecsv.py C:\Silent_garden_B.csv
```
出力先フォルダを指定する場合
出力先フォルダに同名で出力します
```
$ python mergecsv.py -f R:\ C:\Silent_garden_B.csv
```
出力パスを指定する場合
出力ファイル名を含めて指定することができます
出力パスを指定する場合、複数ファイルの指定はできません
```
$ python mergecsv.py -o R:\Silent_garden_B.csv C:\Silent_garden_B.csv
```
合計行を削除する場合
他のソフトで結果の解析をしたい場合などに合計の行を削除すると都合がいい可能性があります
csv2counter.py との互換性はなくなります
```
$ python mergecsv.py -r -o R:\Silent_garden_B.csv C:\Silent_garden_B.csv
```

