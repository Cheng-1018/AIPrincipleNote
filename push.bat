@echo off
set http_proxy=http://127.0.0.1:7897
set https_proxy=http://127.0.0.1:7897
git add .
git commit -m "站点发布公式渲染"
git push
pause