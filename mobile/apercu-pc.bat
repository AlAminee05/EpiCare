@echo off
rem Ouvre EpiCare sur ce PC dans une fenetre au format telephone (meme app qu'Expo Go).
set "PATH=C:\Program Files\nodejs;%PATH%"
cd /d "%~dp0"
call node scripts/build-web.js
node -e "const fs=require('fs');const s=fs.readFileSync('web/epicare-html.js','utf8');fs.writeFileSync('web/EpiCare-apercu.html', JSON.parse(s.slice(s.indexOf('export default ')+15).trim().replace(/;$/,'')))"
start "" "C:\Program Files\Google\Chrome\Application\chrome.exe" --app="file:///%~dp0web/EpiCare-apercu.html" --window-size=410,860 --user-data-dir="%TEMP%\epicare-apercu"
