@echo off

if not exist bin\%1\data (
  xcopy /E /I /Y /Q data bin\%1\data
)

if not exist bin\%1\data\dll (
  xcopy /E /I /Y /Q platform\windows\dll bin\%1\data\dll
)

if not exist bin\%1\valo.exe.config (
  xcopy /Y /Q platform\windows\valo.exe.config bin\%1
)

if not exist bin\%1\valo.ini (
  xcopy /Y /Q misc\valo.ini bin\%1
)
