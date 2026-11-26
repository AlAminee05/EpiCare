/**
 * EpiCare native shell (iOS / Android, Expo).
 *
 * The screens are the EpiCare web app (../app), embedded offline as one HTML page by
 * scripts/build-web.js. This file only hosts it full screen and passes the safe-area
 * insets (notch, home bar) so the web layout clears them on both platforms.
 */
import { StatusBar } from 'expo-status-bar';
import { useMemo } from 'react';
import { Platform, StyleSheet, View, useColorScheme } from 'react-native';
import { SafeAreaProvider, useSafeAreaInsets } from 'react-native-safe-area-context';
import { WebView } from 'react-native-webview';

import EPICARE_HTML from './web/epicare-html';

function EpiCareScreen() {
  const insets = useSafeAreaInsets();
  const dark = useColorScheme() === 'dark';

  // Android WebViews report env(safe-area-inset-*) as 0: give the real values to the CSS.
  const injected = useMemo(() => `
    (function () {
      var s = document.createElement('style');
      s.textContent = ':root{--safe-top:${Math.round(insets.top)}px;--safe-bottom:${Math.round(insets.bottom)}px}';
      (document.head || document.documentElement).appendChild(s);
    })();
    true;`, [insets.top, insets.bottom]);

  return (
    <View style={[styles.root, { backgroundColor: dark ? '#0a0e15' : '#f3f5f9' }]}>
      <StatusBar style="auto" />
      <WebView
        key={injected /* re-inject if the insets change (rotation) */}
        originWhitelist={['*']}
        source={{ html: EPICARE_HTML, baseUrl: 'https://epicare.app/' }}
        injectedJavaScriptBeforeContentLoaded={injected}
        style={styles.web}
        javaScriptEnabled
        domStorageEnabled
        allowFileAccess
        bounces={false}
        overScrollMode="never"
        contentInsetAdjustmentBehavior="never"
        automaticallyAdjustContentInsets={false}
        setSupportMultipleWindows={false}
        allowsBackForwardNavigationGestures={Platform.OS === 'ios'}
        decelerationRate="normal"
      />
    </View>
  );
}

export default function App() {
  return (
    <SafeAreaProvider>
      <EpiCareScreen />
    </SafeAreaProvider>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1 },
  web: { flex: 1, backgroundColor: 'transparent' },
});
