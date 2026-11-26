@echo off
rem Relie ce projet a votre projet Expo (expo.dev) : connexion puis eas init.
title EpiCare - EAS init
set "PATH=C:\Program Files\nodejs;%APPDATA%\npm;%PATH%"
cd /d "%~dp0"
echo.
echo  1/2  Connexion a votre compte Expo
echo       Tapez votre identifiant (ex. mariiieeeemmmm), Entree, puis votre mot de passe, Entree.
echo       Le mot de passe ne s'affiche pas pendant la frappe : c'est normal.
echo.
call eas.cmd whoami >nul 2>&1 || call eas.cmd login
call eas.cmd whoami || goto fail
echo.
echo  2/2  Liaison du projet (eas init)
echo       Repondez Y (oui) aux questions.
echo.
call eas.cmd init --id d321ff73-33b9-479c-af6e-7625aa3c21e1 || goto fail
echo.
echo  Termine : le projet est relie a expo.dev. Vous pouvez fermer cette fenetre.
pause
exit /b 0
:fail
echo.
echo  Une etape a echoue. Copiez le message ci-dessus.
pause
exit /b 1
