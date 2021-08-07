@echo off
rem+--------------------------------------------------------+
rem csv2graph drag and execute
rem
rem You can run csv2graph on Windows by dragging and dropping
rem the csv file onto the icon for this batch file.
rem
rem In order to use this file, describe the path of the
rem python execute file, the path of csv2graph.py, and
rem the path of the directory that outputs the graph image.
rem 
rem                                                   @ame54
rem+--------------------------------------------------------+

rem Python execute file path
set python="C:\path\to\python.exe"

rem csv2graph path
set csv2graph.py="C:\path\to\csv2graph.py"

rem Directory path to output the graph image
set graphs_dir="C:\path\to\graphs"

set yyyy=%date:~0,4%
set mm=%date:~5,2%
set dd=%date:~8,2%
set time2=%time: =0%
set hh=%time2:~0,2%
set mn=%time2:~3,2%
set ss=%time2:~6,2%
set time=%yyyy%_%mm%_%dd%_%hh%_%mn%_%ss%
echo グラフのプロットを開始します [%date% %time%]

rem -a Output all kinds of graphs
rem -w Outputs a graph that displays the value when the cursor is placed on the browser
@echo on
%python% %csv2graph.py% %*  -i %graphs_dir% -a

pause