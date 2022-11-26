@echo off

set bmp_path=C:\Users\User01\Desktop\BezjeStego\inter-img
set log_file="RS_interpolated_without-secret.txt"


for %%i in (%bmp_path%\*.bmp) do java RS.RSAnalysis %%i >>%log_file%



