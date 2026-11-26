# EpiCare native app (Expo)

iOS and Android app built with Expo. Its screens are the EpiCare web app (`../app`), embedded
offline by `scripts/build-web.js` and shown full screen in a WebView (`App.js`). Edit the
screens in `../app`; this folder only packages them.

## Try it on a phone with Expo Go

```bash
cd mobile
npm install
npm start              # rebuilds the embedded page, then starts Expo
```

Scan the QR code with the iPhone camera (or the Expo Go app on Android). Phone and PC must be on
the same Wi-Fi. If the network blocks devices from seeing each other (university or office
Wi-Fi), use `npm run tunnel` instead.

## Update the results shown

`npm start` re-embeds `../app/data/app_data.json` (and `../app/data/branch2.json` if present).
After `python -m epicare_b1 export-app`, restart `npm start` and reload the app.

## Publish on the App Store and Google Play

Builds run in the cloud with EAS, no Mac needed:

```bash
npx eas-cli@latest login
npx eas-cli@latest build -p android --profile preview   # .apk to share by link
npx eas-cli@latest build -p ios                         # needs an Apple Developer account (99 $/year)
npx eas-cli@latest submit -p ios                        # or -p android (Google Play: 25 $ once)
```

App id: `com.epicare.app` (in `app.json`; change it before the first store upload if needed).
Health apps are reviewed closely: present EpiCare as a research / information tool, not a
medical device, and provide a privacy policy.
