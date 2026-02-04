@echo off
chcp 65001 > nul
title Milestone Tracker Server

cd /d "%~dp0"
python backend.py

pause
