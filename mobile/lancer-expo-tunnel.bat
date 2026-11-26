@echo off
rem Comme lancer-expo.bat, mais passe par Internet (pour les Wi-Fi qui isolent les appareils).
title EpiCare - Expo (tunnel)
set "PATH=C:\Program Files\nodejs;%PATH%"
cd /d "%~dp0"
echo.
echo  EpiCare : demarrage d'Expo en mode tunnel...
echo  Si Expo demande d'installer @expo/ngrok, tapez Y puis Entree.
echo.
call npm.cmd run tunnel
echo.
echo  Expo s'est arrete. Copiez le message ci-dessus si une erreur s'affiche.
pause
