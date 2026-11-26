@echo off
rem Lance l'app EpiCare pour Expo Go (affiche le QR code). Double-cliquez sur ce fichier.
title EpiCare - Expo
set "PATH=C:\Program Files\nodejs;%PATH%"
cd /d "%~dp0"
echo.
echo  EpiCare : demarrage d'Expo... le QR code va s'afficher ci-dessous.
echo  Scannez-le avec l'appareil photo de l'iPhone (meme Wi-Fi que ce PC).
echo  Si l'iPhone ne se connecte pas, fermez cette fenetre et lancez lancer-expo-tunnel.bat
echo.
call npm.cmd start
echo.
echo  Expo s'est arrete. Copiez le message ci-dessus si une erreur s'affiche.
pause
