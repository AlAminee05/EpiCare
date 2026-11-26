@echo off
rem Connecte Expo (sur ce PC) au meme compte que l'application Expo Go de l'iPhone.
title EpiCare - Connexion Expo
set "PATH=C:\Program Files\nodejs;%PATH%"
cd /d "%~dp0"
echo.
echo  Connexion a votre compte Expo.
echo  Tapez votre identifiant Expo (ex. mariiieeeemmmm), Entree, puis votre mot de passe, Entree.
echo  Le mot de passe ne s'affiche pas pendant la frappe : c'est normal.
echo.
call npx.cmd expo login
echo.
call npx.cmd expo whoami
echo.
echo  Si votre identifiant s'affiche ci-dessus, c'est bon :
echo  fermez la fenetre "EpiCare - Expo" puis relancez lancer-expo.bat.
pause
