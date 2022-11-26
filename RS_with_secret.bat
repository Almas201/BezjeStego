@echo off

set bmp_path=C:\Users\User01\Desktop\BezjeStego\res-img
set log_file="RS_interpolated_with_secrect.txt"


for %%i in (%bmp_path%\*.bmp) do java RS.RSAnalysis %%i >>%log_file%



