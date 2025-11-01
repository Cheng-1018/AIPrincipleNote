@echo off
set http_proxy=http://127.0.0.1:7897
set https_proxy=http://127.0.0.1:7897
git add .
git commit -m "Auto commit"
git push
pause